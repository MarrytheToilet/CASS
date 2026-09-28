"""Summarize complete, fresh remote experiments without pooling old generations.

Inference uses paired task bootstrap after averaging demonstration seeds within
task. The original CASS route is frozen; every arm shares fresh replacements.
Wrong supports enter the matched comparison only when geometry tolerances pass.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_csv(name):
    with (HERE/name).open() as f:
        return list(csv.DictReader(f))


def interval(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    rng = np.random.default_rng(20260927)
    draws = v[rng.integers(0, len(v), size=(50000, len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()), ci95=np.quantile(draws, [.025,.975]).tolist(),
                n_tasks=len(v))


def digest(name):
    return hashlib.sha256((HERE/name).read_bytes()).hexdigest()


def compound():
    rows = read_csv('compound_controls.csv')
    table = defaultdict(dict)
    for row in rows:
        key = (row['task'], int(row['seed']))
        assert row['condition'] not in table[key], ('duplicate',key,row['condition'])
        table[key][row['condition']] = row
    tasks = sorted({t for t,s in table})
    expected = {'inferred','no_correction','true_correction','z_only','true_support','replace'}
    assert len(tasks)==10
    for task in tasks:
        for seed in [0,1,2]:
            assert expected <= table[task,seed].keys(), ('incomplete',task,seed)
    routes = {r['task']:r['route'] for r in read_csv('compound_fixed_route_per_task.csv')}
    out = dict(source_sha256=digest('compound_controls.csv'), n_tasks=10, seeds=[0,1,2],
               uncertainty='50,000 paired task bootstrap draws; seeds averaged within task; conditional on frozen original CASS policy',
               route_source_sha256=digest('compound_fixed_route_per_task.csv'),
               comparison_scope='Fresh generations only; no original-runtime scores are pooled',
               fixed_route_counts={route:sum(routes[t]==route for t in tasks) for route in ['compose','replace']})
    all_wrong = [r for r in rows if r['condition'].startswith('wrong_')]
    out['coherence_matching'] = dict(n_candidates_evaluated=len(all_wrong),
                                   n_pass=sum(r['matched_coherence']=='True' for r in all_wrong),
                                   tolerances=dict(mean=.02, maximum=.03))
    values = {mode:{} for mode in ['unrouted','fixed_route']}
    comparisons = {mode:{} for mode in values}
    per_task = []
    for task in tasks:
        by_condition = {c:[] for c in expected}
        wrong_values=[]
        wrong_seed_count=0
        for seed in [0,1,2]:
            arm = table[task,seed]
            for c in expected:
                by_condition[c].append(float(arm[c]['acc']))
            valid=[float(r['acc']) for c,r in arm.items() if c.startswith('wrong_') and r['matched_coherence']=='True']
            if valid:
                wrong_values.append(float(np.mean(valid)))
                wrong_seed_count+=1
        means={c:float(np.mean(a)) for c,a in by_condition.items()}
        # Three seeds must all have a matched wrong control to compare this task.
        if wrong_seed_count==3:
            means['wrong_matched']=float(np.mean(wrong_values))
        for mode in values:
            armmeans={c:(means['replace'] if mode=='fixed_route' and routes[task]=='replace' else a) for c,a in means.items()}
            values[mode][task]=armmeans
            per_task.append(dict(task=task,mode=mode,route=routes[task],**armmeans))
    for mode, by_task in values.items():
        out[mode]={c:interval([v[c] for v in by_task.values() if c in v]) for c in expected|{'wrong_matched'}}
        for left,right in [('inferred','z_only'),('inferred','no_correction'),
                           ('true_support','inferred'),('true_correction','inferred'),
                           ('inferred','wrong_matched'),('true_correction','no_correction')]:
            tt=[t for t,v in by_task.items() if left in v and right in v]
            result=interval([by_task[t][left]-by_task[t][right] for t in tt])
            comparisons[mode][left+' minus '+right]=dict(result=result,tasks=tt)
    out['paired_differences']=comparisons
    out['gates']=dict(minimum=min(float(r['gate']) for r in rows if r['condition']=='inferred'),
                      maximum=max(float(r['gate']) for r in rows if r['condition']=='inferred'),
                      zero_count=sum(float(r['gate'])==0 for r in rows if r['condition']=='inferred'))
    fields=['task','mode','route']+sorted(expected|{'wrong_matched'})
    with (HERE/'fresh_compound_per_task.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(per_task)
    (HERE/'fresh_compound_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def latency():
    data=json.loads((HERE/'latency_results.json').read_text())
    assert data['cuda_visible_devices']=='2', 'Only authorized remote GPU measurements are used'
    rows=data['measurements']
    assert len(rows)==33, ('incomplete',len(rows))
    out={k:data[k] for k in ['gpu','torch','dtype','attention','scope','energy_scope','model_load_seconds','dictionary_rebuild_seconds','dictionary_policy','held_out_dictionary_rebuild_seconds']}
    out['source_sha256']=digest('latency_results.json')
    out['by_task']={}
    for task in data['tasks']:
        rr=[r for r in rows if r['task']==task]
        adaptation=next(r for r in rr if r['measurement']=='adaptation_end_to_end')
        result=dict(adaptation_seconds=adaptation['median_seconds'],
                    adaptation_gpu_board_joules=adaptation['median_gpu_board_joules'],serving={})
        for bs in [1,8,25]:
            ca=next(r for r in rr if r['measurement']=='serve_cass' and r['batch_size']==bs)
            ic=next(r for r in rr if r['measurement']=='serve_icl4' and r['batch_size']==bs)
            delta=(ic['median_seconds']-ca['median_seconds'])/bs
            ej=[ca['median_gpu_board_joules'],ic['median_gpu_board_joules'],adaptation['median_gpu_board_joules']]
            de=(ej[1]-ej[0])/bs if all(x is not None for x in ej) else None
            result['serving'][str(bs)]=dict(cass_batch_seconds=ca['median_seconds'],icl4_batch_seconds=ic['median_seconds'],
                 cass_gpu_board_joules=ej[0],icl4_gpu_board_joules=ej[1],
                 latency_break_even_queries=float(adaptation['median_seconds']/delta) if delta>0 else None,
                 energy_break_even_queries=float(ej[2]/de) if de is not None and de>0 else None)
        out['by_task'][task]=result
    (HERE/'latency_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def qwen():
    rows=read_csv('qwen_probe.csv')
    tasks=sorted({r['task'] for r in rows}); methods=sorted({r['method'] for r in rows})
    assert len(rows)==56 and len(tasks)==8 and len(methods)==7
    pairs={(r['task'],r['method']):float(r['acc']) for r in rows}
    assert len(pairs)==56
    out=dict(source_sha256=digest('qwen_probe.csv'),scope='Eight tasks, one fixed configuration grid; descriptive model/intervention diagnosis',
             means={m:float(np.mean([pairs[t,m] for t in tasks])) for m in methods},
             differences_to_pair_last1={m:interval([pairs[t,m]-pairs[t,'pair_last1'] for t in tasks]) for m in methods if m!='pair_last1'})
    (HERE/'qwen_probe_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


if __name__=='__main__':
    import sys
    functions={'compound':compound,'latency':latency,'qwen':qwen}
    for kind in sys.argv[1:] or functions:
        print(json.dumps({kind:functions[kind]()},indent=2))
