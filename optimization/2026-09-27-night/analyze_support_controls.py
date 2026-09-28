"""Summarize all predeclared support interventions with matched-task intervals."""
import json
import numpy as np
import pandas as pd
from common import HERE,ROOT
from summarize_night import read_rows,paired
from cass.evaluate import accuracy


def main():
    rows=read_rows('support_controls')
    if not rows:return
    by={};seen=set()
    for r in rows:
        key=(r['task'],r['seed'],r['config']);assert key not in seen;seen.add(key)
        assert abs(accuracy(r['predictions'],[y for x,y in r['queries']],case_sensitive=True)-r['acc'])<1e-12
        by.setdefault(r['task'],{}).setdefault(r['seed'],{})[r['config']]=r
    arms=['inferred','no_correction','true_correction','true_correction_norm','true_support']
    tasks={};matching=[]
    for name,seeds in by.items():
        if set(seeds)!={20,21,22}:continue
        values={a:[] for a in arms+['wrong_matched','wrong_norm_matched']}
        complete=True
        for seed,records in seeds.items():
            if not set(arms)<=set(records):complete=False;break
            for a in arms:values[a].append(records[a]['acc'])
            for norm,label in [(False,'wrong_matched'),(True,'wrong_norm_matched')]:
                available=[r for k,r in records.items() if k.startswith('wrong') and r['correction_norm_matched']==norm]
                valid=[r['acc'] for r in available if r['rank_coherence_matched']]
                matching.append(dict(task=name,seed=seed,norm_matched=norm,candidates=len(available),passed=len(valid)))
                if valid:values[label].append(float(np.mean(valid)))
        if not complete:continue
        tasks[name]={a:float(np.mean(v)) for a,v in values.items() if len(v)==3}
    summary=dict(complete_tasks=len(tasks),per_task=tasks,matching=matching,means={},paired={},
                 scope='Equal task weights after averaging seeds; matched alternatives averaged within seed first; 50000 paired task-bootstrap draws.')
    for a in arms+['wrong_matched','wrong_norm_matched']:
        vals=[v[a] for v in tasks.values() if a in v]
        if vals:summary['means'][a]=dict(n_tasks=len(vals),accuracy=float(np.mean(vals)))
    for a,b in [('inferred','no_correction'),('inferred','wrong_matched'),('inferred','wrong_norm_matched'),
                ('true_correction','inferred'),('true_correction_norm','inferred'),('true_support','inferred')]:
        names=[n for n,v in tasks.items() if a in v and b in v]
        if names:summary['paired'][a+'__'+b]=paired([tasks[n][a] for n in names],[tasks[n][b] for n in names])
    # Example selection is descriptive and deterministic, never a metric selector.
    examples=[]
    for name,seeds in by.items():
        r=seeds.get(20,{})
        if not set(arms)<=set(r):continue
        for i,(x,y) in enumerate(r['inferred']['queries']):
            p={a:r[a]['predictions'][i] for a in arms}
            ok={a:bool(accuracy([p[a]],[y],case_sensitive=True)) for a in arms}
            if (ok['inferred'] and not ok['no_correction']) or (not ok['inferred'] and not ok['true_support']):
                examples.append(dict(task=name,input=x,target=y,predictions=p,correct=ok));break
    summary['representative_examples']=examples
    # Apply exactly the submitted task-level routing decisions to every arm.
    # Replacement outputs are the same new-seed outputs in signature_confirm.
    route_path=ROOT/'rebuttal/2026-09-27/compound_fixed_route_per_task.csv'
    if route_path.exists():
        routes=pd.read_csv(route_path).set_index('task').route.to_dict()
        replacements={}
        for r in read_rows('signature_confirm'):
            if r['suite']=='compound' and r['config']=='replace':
                replacements.setdefault(r['task'],{})[r['seed']]=r
        routed={}
        for name,values in tasks.items():
            assert routes[name] in ['compose','replace']
            if routes[name]=='compose':routed[name]=values.copy();continue
            assert set(replacements[name])=={20,21,22}
            for seed in [20,21,22]:
                assert replacements[name][seed]['queries']==by[name][seed]['inferred']['queries']
            replacement=float(np.mean([r['acc'] for r in replacements[name].values()]))
            routed[name]={a:replacement for a in values}
        entry=dict(policy_source=str(route_path.relative_to(ROOT)),
                   replacement_source='signature_confirm.jsonl',
                   policy='Original task-level routing frozen across every intervention arm; '
                          'all new-seed replacement predictions identical across arms.',
                   per_task=routed,means={},paired={})
        for a in arms+['wrong_matched','wrong_norm_matched']:
            vals=[v[a] for v in routed.values() if a in v]
            if vals:entry['means'][a]=dict(n_tasks=len(vals),accuracy=float(np.mean(vals)))
        for a,b in [('inferred','no_correction'),('inferred','wrong_matched'),('inferred','wrong_norm_matched'),
                    ('true_correction','inferred'),('true_correction_norm','inferred'),('true_support','inferred')]:
            names=[n for n,v in routed.items() if a in v and b in v]
            if names:entry['paired'][a+'__'+b]=paired([routed[n][a] for n in names],[routed[n][b] for n in names])
        summary['common_submitted_routing']=entry
    (HERE/'support_controls_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(complete_tasks=len(tasks),means=summary['means'],paired=summary['paired']),indent=2))

if __name__=='__main__':main()
