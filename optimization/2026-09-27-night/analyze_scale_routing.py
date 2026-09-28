"""Portability of submitted thresholds under all completed dictionary expansions."""
import hashlib
import json
import numpy as np
from common import HERE
from cass.evaluate import accuracy
from summarize_night import paired


def read(stem):
    return [json.loads(s) for s in (HERE/(stem+'.jsonl')).read_text().splitlines()]


def main():
    icl={(r['task'],r['seed']):r for r in read('extension_confirm') if r['config']=='icl4'}
    replacement={(r['task'],r['seed']):r for r in read('signature_confirm') if r['config']=='replace'}
    output=dict(protocol='Post-hoc portability check: norm<5.125 replaces, residual>0.7 escalates; thresholds never retuned; escalation precedes replacement in combined rule.',
                scope='Operational branch quality and stability, not calibrated failure probabilities. No timing claim is inherited.',
                sources={},results=[],per_task_seed=[])
    for stem in ['scale_eval','scale_eval_fixed_shared','scale_context','scale_context_fixed_shared']:
        assert (HERE/(stem+'_done.json')).exists()
        source=HERE/(stem+'.jsonl');digest=hashlib.sha256(source.read_bytes()).hexdigest()
        audit=json.loads((HERE/(stem+'_analysis.json')).read_text())
        assert audit['source_sha256']==digest and audit['generation_metrics_verified']
        output['sources'][stem]=digest
        raw=read(stem);tasks=sorted({r['task'] for r in raw});assert len(tasks)==25
        rows=[]
        for r in raw:
            key=(r['task'],r['seed']);reference=icl[key];repl=replacement[key]
            assert r['queries']==reference['queries']==repl['queries']
            if 'demos' in repl:assert repl['demos']==reference['demos']
            for policy in ['steering','norm_only','residual_only','combined']:
                branch='steering'
                if policy in ['norm_only','combined'] and r['znorm']<5.125:branch='replacement'
                if policy in ['residual_only','combined'] and r['residual']>.7:branch='icl4'
                preds={'steering':r,'replacement':repl,'icl4':reference}[branch]['predictions']
                value=accuracy(preds,[y for x,y in r['queries']],case_sensitive='+' in r['task'])
                rows.append(dict(stem=stem,size=r['size'],mode=r['solver_mode'],task=r['task'],seed=r['seed'],
                                 policy=policy,branch=branch,accuracy=value,icl4_accuracy=reference['acc']))
        output['per_task_seed'].extend(rows)
        for mode in ['full','shortlist32']:
            for policy in ['steering','norm_only','residual_only','combined']:
                base={(r['task'],r['seed']):r for r in rows if r['mode']==mode and r['policy']==policy and r['size']==32}
                for size in [32,64,128,256]:
                    part=[r for r in rows if r['mode']==mode and r['policy']==policy and r['size']==size]
                    assert len(part)==75
                    means=[np.mean([r['accuracy'] for r in part if r['task']==t]) for t in tasks]
                    base_means=[np.mean([r['accuracy'] for (n,s),r in base.items() if n==t]) for t in tasks]
                    icl_means=[np.mean([r['icl4_accuracy'] for r in part if r['task']==t]) for t in tasks]
                    output['results'].append(dict(stem=stem,size=size,mode=mode,policy=policy,
                        accuracy=float(np.mean(means)),
                        fractions={b:sum(r['branch']==b for r in part)/len(part) for b in ['steering','replacement','icl4']},
                        route_change_from32=float(np.mean([r['branch']!=base[r['task'],r['seed']]['branch'] for r in part])),
                        change_from32=paired(means,base_means),difference_from_icl4=paired(means,icl_means)))
    for stem in ['extension_confirm','signature_confirm']:
        output['sources'][stem]=hashlib.sha256((HERE/(stem+'.jsonl')).read_bytes()).hexdigest()
    target=HERE/'scaling_routing_analysis.json';target.write_text(json.dumps(output,indent=2)+'\n');print(target)


if __name__=='__main__':main()
