"""Metric-homogeneous broader summaries and a frozen submitted-routing check."""
import json
import hashlib
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from common import HERE
from summarize_night import read_rows,paired
from broader_tasks import score
from cass.config import MODEL_PATHS
from cass.tasks import build_fewshot_pair_prompts,zs_prompt,icl_prompt


def family(task):
    if task.startswith('code-sort3-'):return 'code_multistep_execution'
    if task.startswith('code-'):return 'code_expression_execution'
    if task.startswith('long-'):return 'long_structured_whole_answer'
    if task.startswith('entity-'):return 'entity_token_f1'
    if task=='reading-comprehension':return 'reading_token_f1'
    return 'classification_accuracy'


def main():
    all_rows=[];by={};identities={};sources={}
    for stem in ['broader_llama31-8b','broader_signature_llama31-8b','broader_extensions','broader_native','broader_position','broader_projector']:
        rows=read_rows(stem);seen=set()
        if rows:sources[stem]=hashlib.sha256((HERE/(stem+'.jsonl')).read_bytes()).hexdigest()
        for r in rows:
            key=(r['task'],r['seed'],r['method']);assert key not in seen;seen.add(key)
            values=[score(r['task'],p,y) for p,(x,y) in zip(r['predictions'],r['queries'])]
            assert abs(float(np.mean(values))-r['score'])<1e-12
            np.testing.assert_allclose(values,r['per_example_score'],atol=1e-12,rtol=0.)
            identity=(r['queries'],r['demos']);ts=key[:2]
            if ts in identities:assert identity==identities[ts]
            else:identities[ts]=identity
            method=r['method']
            if stem=='broader_native':method='native_'+method
            if stem=='broader_position':method='position_'+method
            if stem=='broader_projector':method='projector_'+method
            if method in ['cass','z_only']:
                method=('original_' if stem=='broader_llama31-8b' else 'mixed_')+method
            out=dict(r,method=method,source=stem,metric_family=family(r['task']))
            key=(r['task'],r['seed'],method)
            if key in by:
                assert by[key]['predictions']==r['predictions'],'Reused baseline must be identical'
                continue
            by[key]=out;all_rows.append(out)
    if not all_rows:return
    tok=AutoTokenizer.from_pretrained(MODEL_PATHS['llama31-8b'],local_files_only=True)
    token_cache={}
    def adaptation_tokens(row,context,route):
        key=(row['task'],row['seed'],context,route)
        if key in token_cache:return token_cache[key]
        ex=row['demos'];cp,cn=build_fewshot_pair_prompts(ex,random.Random(9000+row['seed']),n_reps=6)
        prompts=cp+cn
        if context and route=='steer':prompts += [zs_prompt(x) for x,y in ex]
        if route=='replace':prompts += [icl_prompt([e for i,e in enumerate(ex) if i!=j],x) for j,(x,y) in enumerate(ex)]
        result=(sum(len(tok.encode(p)) for p in prompts),len(prompts));token_cache[key]=result
        return result
    routing=[]
    for task,seed,method in list(by):
        if method!='original_cass':continue
        source=by[task,seed,method]
        for context in [False,True]:
            steering='combined' if context else 'original_cass'
            if (task,seed,steering) not in by:continue
            for policy in ['norm_only','norm_plus_residual']:
                route=('icl4' if policy=='norm_plus_residual' and source['residual']>.7 else
                       'replace' if source['znorm']<5.125 else 'steer')
                used=steering if route=='steer' else route
                if (task,seed,used) not in by:continue
                row=by[task,seed,used];tokens,sequences=adaptation_tokens(source,context,route)
                routing.append(dict(task=task,seed=seed,operator='contextual' if context else 'submitted',policy=policy,
                    route=route,score=row['score'],metric_family=family(task),prompt_tokens=row['mean_input_tokens'],
                    adaptation_input_tokens=tokens,adaptation_sequences=sequences,
                    input_tokens_per_query_at1000=row['mean_input_tokens']+tokens/1000.,
                    residual=source['residual'],znorm=source['znorm']))
    df=pd.DataFrame(all_rows);summary=dict(n_tasks=int(df.task.nunique()),n_cells=len(df),sources=sources,
        scope='All predeclared tasks; seeds averaged within task, then equal task weights within the stated metric family. '
              'Zero-shot uses its single deterministic generation set. No global average mixes F1 with accuracy.',
        per_task=[],per_task_comparisons=[],families=[],routing=dict(per_seed=routing,per_task=[],families=[],
            policy='Thresholds5.125 and0.7 frozen from submitted norm policy and residual dial; '
                   'all variants share original extraction signals and byte-identical fallback predictions.',
            token_scope='Input-token accounting at1000 queries; includes branch-dependent online extraction, '
                        'excludes offline mining, output tokens and reusable-prefix caching; not a latency or energy estimate.'))
    for (task,method),part in df.groupby(['task','method']):
        expected=1 if method in ['zero','native_zero','position_zero','projector_zero'] else 3
        if part.seed.nunique()!=expected:continue
        summary['per_task'].append(dict(task=task,method=method,metric_family=family(task),
                                      score=float(part.score.mean()),n_seeds=expected,prompt_tokens=float(part.mean_input_tokens.mean())))
    per=pd.DataFrame(summary['per_task'])
    contrasts=[('mixed_cass','original_cass'),('mixed_cass','mixed_z_only'),('mixed_cass','zero'),
               ('combined','original_cass'),('combined','combined_no_correction'),('combined','dictionary_free'),
               ('combined','zero'),('oneshot','icl1'),('oneshot','oneshot_no_correction'),
               ('native_best_correction','native_original'),('native_best_correction','native_matched_no_correction'),
               ('native_best_correction','native_dictionary_free'),('native_best_correction','native_zero'),
               ('native_best_correction','combined'),
               ('position_best_correction','position_matched_no_correction'),
               ('position_best_correction','position_dictionary_free'),
               ('position_best_correction','position_frozen_cass'),
               ('position_best_correction','position_zero'),
               ('projector_best_correction','projector_matched_no_correction'),
               ('projector_best_correction','projector_dictionary_free'),
               ('projector_best_correction','position_best_correction'),
               ('projector_best_correction','projector_frozen_cass'),
               ('projector_best_correction','projector_zero')]
    for task in sorted(df.task.unique()):
        for a,b in contrasts:
            ar=[by.get((task,seed,a)) for seed in [20,21,22]]
            br=[by.get((task,20 if b in ['zero','native_zero','position_zero','projector_zero'] else seed,b)) for seed in [20,21,22]]
            if any(r is None for r in ar+br):continue
            for x,y in zip(ar,br):assert x['queries']==y['queries']
            av=np.asarray([r['per_example_score'] for r in ar]);bv=np.asarray([r['per_example_score'] for r in br])
            diff=(av-bv).mean(0)
            rng=np.random.default_rng(20260927)
            samples=diff[rng.integers(len(diff),size=(50000,len(diff)))].mean(1)
            summary['per_task_comparisons'].append(dict(task=task,metric_family=family(task),a=a,b=b,
                n_queries=len(diff),n_demo_seeds=3,diff=float(diff.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                uncertainty='Paired query bootstrap after averaging the three fixed demonstration seeds; '
                            'conditional on those demonstrations, not a cross-task interval.'))
    for metric,part in per.groupby('metric_family'):
        table=part.pivot(index='task',columns='method',values='score')
        entry=dict(metric_family=metric,means={},paired={})
        for method in table:
            vals=table[method].dropna();entry['means'][method]=dict(n_tasks=len(vals),score=float(vals.mean()))
        for a,b in contrasts:
            if a not in table or b not in table:continue
            pairs=table[[a,b]].dropna()
            if len(pairs)>=2:entry['paired'][a+'__'+b]=paired(pairs[a],pairs[b])
        summary['families'].append(entry)
    if routing:
        rdf=pd.DataFrame(routing)
        for (task,operator,policy),part in rdf.groupby(['task','operator','policy']):
            if part.seed.nunique()!=3:continue
            summary['routing']['per_task'].append(dict(task=task,operator=operator,policy=policy,metric_family=family(task),
                score=float(part.score.mean()),icl_fraction=float((part.route=='icl4').mean()),
                replace_fraction=float((part.route=='replace').mean()),steer_fraction=float((part.route=='steer').mean()),
                input_tokens_per_query_at1000=float(part.input_tokens_per_query_at1000.mean())))
        tab=pd.DataFrame(summary['routing']['per_task'])
        if len(tab):
            for (metric,operator,policy),part in tab.groupby(['metric_family','operator','policy']):
                summary['routing']['families'].append(dict(metric_family=metric,operator=operator,policy=policy,n_tasks=len(part),
                    **{c:float(part[c].mean()) for c in ['score','icl_fraction','replace_fraction','steer_fraction','input_tokens_per_query_at1000']}))
    per.to_csv(HERE/'broader_per_task.csv',index=False)
    (HERE/'broader_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['families'],indent=2))


if __name__=='__main__':main()
