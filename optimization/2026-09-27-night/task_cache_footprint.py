"""Task-state tensors versus reusable ICL prefixes; no GPU work."""
import hashlib
import json
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from common import HERE,ALL_TASKS,make_dict
from cass.config import MODEL_PATHS
from cass.tasks import zs_prompt,icl_prompt,load_task
from fast_solver import GramSolver
from efficient_ops import selected_zlist
from summarize_night import read_rows
from analyze_broader import family


def main():
    layers=[12,16];G={l:{} for l in layers}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:G[l][name]=b['G_by_layer'][l].float().numpy()
    D=make_dict(G,layers);config=AutoConfig.from_pretrained(MODEL_PATHS['llama31-8b'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(MODEL_PATHS['llama31-8b'],local_files_only=True)
    head_dim=getattr(config,'head_dim',None) or config.hidden_size//config.num_attention_heads
    per_token=config.num_hidden_layers*2*config.num_key_value_heads*head_dim*2
    def prefix_length(prefix,prompts):
        ids=tok.encode(prefix);complete=[tok.encode(p) for p in prompts]
        while ids and any(p[:len(ids)]!=ids for p in complete):ids=ids[:-1]
        assert ids
        return len(ids)
    results=[];sources={};checks=[]
    timing=HERE/'matched_engine_latency.json'
    if timing.exists():
        measured=json.loads(timing.read_text())
        if measured.get('complete'):
            for rec in measured['caches']:
                name=rec['task'];current=make_dict(G,layers,exclude=name)
                data=torch.load(HERE/f'cache_signature_confirm_{name}_20.pt',weights_only=True)
                code=GramSolver(current).solve(selected_zlist(current,data['null']))
                ranks=[min(current.d,sum(current.per_layer[l].bases[n].shape[1] for n in code.support)) for l in layers]
                compiled_bytes=sum((current.d*r+current.d+1)*4 for r in ranks)
                assert compiled_bytes==rec['compiled_operator_tensor_bytes']
                for mode in ['cass','icl4']:assert rec[mode]['kv_bytes']==rec[mode]['tokens']*per_token
                checks.append(dict(task=name,compiled_operator_bytes_match=True,prefix_kv_bytes_match=True,
                                   uncompiled_operator_bytes=sum((current.d*r+2*current.d)*4 for r in ranks)))
    for stem in ['extension_confirm','broader_extensions']:
        path=HERE/(stem+'.jsonl')
        if not path.exists():continue
        rows=read_rows(stem);sources[stem]=hashlib.sha256(path.read_bytes()).hexdigest()
        for row in rows:
            if row.get('config',row.get('method'))!='combined':continue
            if stem=='extension_confirm' and row['suite']=='loto':continue
            assert row['config_parameters' if stem=='extension_confirm' else 'variant_config']['null_fraction']==1.
            # The retained QR width is min(d, sum of selected basis widths).
            ranks=[min(D.d,sum(D.per_layer[l].bases[n].shape[1] for n in row['support'])) for l in layers]
            operator_bytes=sum((D.d*r+2*D.d)*4 for r in ranks)
            examples=row['demos'];queries=row['queries']
            p0=[zs_prompt(x) for x,y in queries];p4=[icl_prompt(examples,x) for x,y in queries]
            n0=prefix_length('Q:',p0)
            n4=prefix_length('\n\n'.join(f'Q: {x}\nA: {y}' for x,y in examples)+'\n\n',p4)
            total=operator_bytes+n0*per_token;icl=n4*per_token
            results.append(dict(task=row['task'],seed=row['seed'],suite=row.get('suite',family(row['task'])),source=stem,
                ranks=ranks,operator_float32_bytes=operator_bytes,cass_prefix_tokens=n0,icl4_prefix_tokens=n4,
                cass_cached_task_state_bytes=total,icl4_cached_task_state_bytes=icl,icl4_to_cass_ratio=icl/total))
    groups=[]
    for suite in sorted({r['suite'] for r in results}):
        rows=[r for r in results if r['suite']==suite]
        groups.append(dict(suite=suite,n_tasks=len({r['task'] for r in rows}),n_task_seeds=len(rows),
            median_cass_mib=float(np.median([r['cass_cached_task_state_bytes'] for r in rows])/2**20),
            median_icl4_mib=float(np.median([r['icl4_cached_task_state_bytes'] for r in rows])/2**20),
            median_per_task_seed_ratio=float(np.median([r['icl4_to_cass_ratio'] for r in rows]))))
    result=dict(kv_bytes_per_token=per_token,sources=sources,rows=results,groups=groups,measured_tensor_checks=checks,
        scope='Float32 operator tensor-shape accounting (Q, direction, anchor) plus unquantized BF16 reusable prefix KV. '
              'Excludes shared model weights, the global offline dictionary, Python metadata, allocator overhead and transient query KV. '
              'No claim of measured multi-task throughput or total-system memory reduction; task states are assumed individually retained.')
    (HERE/'task_cache_footprint.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
