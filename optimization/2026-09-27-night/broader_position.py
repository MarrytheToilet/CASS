"""Frozen position refinement on every predeclared broader task."""
import json
import hashlib
import argparse
import os
import time
import numpy as np
import torch
import position_refine as position
from common import HERE, make_dict, generate
from fast_solver import GramSolver
from broader_tasks import load_broader, score, clean_answer
from cass.models import HookedLM
from cass.tasks import zs_prompt
from freeze_extensions import freeze as base_selection


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--projector',action='store_true');args=parser.parse_args()
    if args.projector:
        import projector_refine as refinement
        while not (HERE/'projector_dev_done.json').exists():
            print('Waiting for frozen projector-development selection; no model loaded.',flush=True)
            time.sleep(10)
    else:refinement=position
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2', '3']
    selected = refinement.freeze()
    G, clean = refinement.assets()
    D = make_dict(G, refinement.LAYERS)
    solver = GramSolver(D)
    hlm = HookedLM('llama31-8b')
    reference = {}
    contents=(HERE/'broader_extensions.jsonl').read_bytes()
    source_hash=hashlib.sha256(contents).hexdigest()
    for line in contents.decode().splitlines():
        row = json.loads(line)
        reference[row['task'], row['seed'], row['method']] = row
    assert (HERE/'broader_extensions_done.json').exists()
    assert (HERE/'answer_stop_verified.json').exists()
    stem='broader_projector' if args.projector else 'broader_position'
    frozen_context=dict(base_selection()['combined']['config'],pipeline='context',positions=1)
    checked_reuse=set()
    path = HERE/(stem+'.jsonl')
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done.add((row['task'], row['seed'], row['method']))
    with path.open('a') as out:
        for name, task in load_broader().items():
            queries = task.eval_queries
            limit = 128 if name.startswith(('code-', 'long-')) else 32
            for seed in [20, 21, 22]:
                methods = ['best_correction', 'matched_no_correction', 'best_overall',
                           'dictionary_free', 'frozen_cass', 'icl1', 'icl4']
                if seed == 20:
                    methods.append('zero')
                if all((name, seed, m) in done for m in methods):
                    continue
                rng = np.random.default_rng(100*seed+4)
                examples = [task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool), 4, replace=False)]
                raw = torch.load(HERE/f'cache_broader_llama31-8b_{name}_{seed}.pt', weights_only=True)
                null = torch.load(HERE/f'cache_broader_null_llama31-8b_{name}_{seed}.pt', weights_only=True)
                data = dict(raw={l:raw[:, l] for l in refinement.LAYERS}, null=null)
                prepared = refinement.prepare(D, solver, data, clean)
                prompts = [zs_prompt(x) for x, y in queries]
                generated = {}
                for method in methods:
                    if (name, seed, method) in done:
                        continue
                    if method in ['frozen_cass', 'icl1', 'icl4', 'zero']:
                        source_method = 'combined' if method == 'frozen_cass' else method
                        row = reference[name, seed, source_method].copy()
                        assert row['queries'] == [list(q) for q in queries]
                        assert row['demos'] == [list(e) for e in examples]
                        row.update(method=method, reused_baseline_source='broader_extensions.jsonl',
                                   reused_baseline_method=source_method)
                    else:
                        cfg = selected['best_correction' if method == 'matched_no_correction' else method]
                        off = method == 'matched_no_correction'
                        identity = json.dumps([cfg, off], sort_keys=True)
                        if cfg==frozen_context:
                            source_method='combined_no_correction' if off else 'combined'
                            ref=reference[name,seed,source_method]
                            assert ref['queries']==[list(q) for q in queries] and ref['demos']==[list(e) for e in examples]
                            assert dict(ref['variant_config'],pipeline='context',positions=1)==cfg
                            assert ref['max_new_tokens']==limit and ref['answer_stop_verified']
                            predictions=ref['predictions']
                            if source_method not in checked_reuse:
                                ops,layers=refinement.operators(cfg,prepared,data,hlm,examples,off)
                                assert generate(hlm,prompts,ops,layers,cfg['schedule'],positions=1,batch_size=4,
                                                max_new_tokens=limit,stop_after_answer=True)==predictions
                                checked_reuse.add(source_method)
                        elif identity in generated:
                            predictions = generated[identity]
                        else:
                            ops, layers = refinement.operators(cfg, prepared, data, hlm, examples, off)
                            predictions = generate(hlm, prompts, ops, layers, cfg['schedule'],
                                                   positions=cfg['positions'], batch_size=4,
                                                   max_new_tokens=limit, stop_after_answer=True)
                            generated[identity] = predictions
                        values = [score(name, pred, y) for pred, (x, y) in zip(predictions, queries)]
                        row = dict(task=name, seed=seed, method=method, family=task.family,
                                   score=float(np.mean(values)), per_example_score=values,
                                   queries=queries, predictions=predictions, demos=examples,
                                   variant_config=cfg, correction_disabled=off,
                                   dictionary_used=cfg['pipeline']!='dictionary_free',
                                   development_source_sha256=selected['source_sha256'],
                                   answer_stop_verified=True, max_new_tokens=limit,
                                   support=prepared[2].support, residual=prepared[2].residual,
                                   mean_input_tokens=float(np.mean([len(hlm.tok.encode(p)) for p in prompts])))
                        if cfg==frozen_context:
                            row.update(reused_baseline_source='broader_extensions.jsonl',
                                       reused_baseline_method=source_method,reused_baseline_sha256=source_hash,
                                       reused_identical_configuration=True)
                        if name.startswith('long-'):
                            row['position_accuracy'] = float(np.mean([
                                sum(a == b for a, b in zip(clean_answer(pred).split(), y.split()))/len(y.split())
                                for pred, (x, y) in zip(predictions, queries)]))
                    out.write(json.dumps(row, ensure_ascii=False)+'\n')
                    out.flush()
                    done.add((name, seed, method))
                    print(name, seed, method, round(row['score'], 4), flush=True)
    assert len(done) == 374
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


if __name__ == '__main__':
    main()
