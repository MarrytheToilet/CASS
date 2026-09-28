"""Unretuned submitted residual threshold on both contextual confirmation splits."""
import hashlib
import json
import numpy as np
from common import HERE,DEV_TASKS
from cass.evaluate import accuracy
from robust_metrics import score
from summarize_night import paired


def main():
    result=dict(protocol='Post-hoc portability of the submitted residual>0.7 threshold, no new fitting. The existing norm<5.125 replacement branch is unused for every contextual signature in these two splits.',
                sources={},splits={})
    for split in ['confirm','fresh']:
        path=HERE/('extension_'+split+'.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()]
        indexed={(r['task'],r['seed'],r['config']):r for r in rows}
        combined=[r for r in rows if r['config']=='combined'];assert len(combined)==171
        assert all(r['znorm']>=5.125 for r in combined)
        result['sources'][split]=hashlib.sha256(path.read_bytes()).hexdigest()
        cells=[]
        for r in combined:
            name,seed=r['task'],r['seed'];off=indexed[name,seed,'combined_no_correction'];icl=indexed[name,seed,'icl4']
            assert r['queries']==off['queries']==icl['queries'] and r['demos']==off['demos']==icl['demos']
            branch=r['residual']>.7
            for metric in ['submitted_prefix','task_case_literal_line']:
                def evaluate(predictions):
                    if metric=='submitted_prefix':return accuracy(predictions,[y for x,y in r['queries']],case_sensitive='+' in name)
                    return float(np.mean([score(p,y,name,metric) for p,(x,y) in zip(predictions,r['queries'])]))
                cells.append(dict(task=name,seed=seed,suite=r['suite'],metric=metric,uses_icl=branch,
                    routed=evaluate(icl['predictions'] if branch else r['predictions']),
                    routed_off=evaluate(icl['predictions'] if branch else off['predictions']),
                    steering=evaluate(r['predictions']),icl4=evaluate(icl['predictions'])))
        result['splits'][split]=dict(per_task_seed=cells,suites={})
        for suite in ['heldout_known','loto','novel','compound']:
            if suite=='heldout_known':selected=[r for r in cells if r['suite']=='loto' and r['task'] not in DEV_TASKS]
            else:selected=[r for r in cells if r['suite']==suite]
            entry={}
            for metric in ['submitted_prefix','task_case_literal_line']:
                part=[r for r in selected if r['metric']==metric];names=sorted({r['task'] for r in part})
                assert len(part)==3*len(names)>0
                means={m:[np.mean([r[m] for r in part if r['task']==n]) for n in names] for m in ['routed','routed_off','steering','icl4']}
                entry[metric]=dict(n_tasks=len(names),icl4_fraction=float(np.mean([r['uses_icl'] for r in part])),
                    means={m:float(np.mean(v)) for m,v in means.items()},
                    routed_vs_icl4=paired(means['routed'],means['icl4']),
                    routed_vs_steering=paired(means['routed'],means['steering']),
                    routed_correction=paired(means['routed'],means['routed_off']))
            result['splits'][split]['suites'][suite]=entry
    target=HERE/'extension_routing_analysis.json';target.write_text(json.dumps(result,indent=2)+'\n');print(target)


if __name__=='__main__':main()
