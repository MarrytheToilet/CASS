"""Post-hoc scoring robustness, never used to choose configurations."""
import json
import numpy as np
import pandas as pd
from common import HERE,DEV_TASKS
from cass.evaluate import exact_match,_words
from summarize_night import read_rows,paired

CASE_TASKS={'capitalize','capitalize-first-letter','capitalize-last-letter',
            'lowercase-first-letter','next-capital-letter','element-symbol'}


def score(pred,target,task,metric):
    sensitive='+' in task or task in CASE_TASKS
    if metric=='task_case_prefix':return float(exact_match(pred,target,sensitive))
    if metric=='task_case_full_line':return float(_words(pred,sensitive)==_words(target,sensitive) and bool(_words(target,sensitive)))
    if metric=='all_case_full_line':return float(_words(pred,True)==_words(target,True) and bool(_words(target,True)))
    if metric=='task_case_literal_line':
        def normalize(s):
            line=' '.join(s.strip().split('\n')[0].split())
            return line if sensitive else line.lower()
        return float(bool(normalize(target)) and normalize(pred)==normalize(target))
    raise ValueError(metric)


def main():
    result={'scope':'Supplementary metric robustness only; not used for selecting or tuning configurations.',
            'case_sensitive_tasks':sorted(CASE_TASKS),'experiments':{}}
    for stem in ['signature_confirm','extension_confirm','extension_fresh','qwen_confirm',
                 'qwen3-4b_contrast_confirm','qwen25-3b_contrast_confirm','adaptive_combined',
                 'qwen3-4b_contrast_fresh','qwen25-3b_contrast_fresh',
                 'llama_native_confirm','llama_native_fresh','position_confirm','position_fresh',
                 'projector_confirm','projector_fresh',
                 'teacher_route_confirm','teacher_route_fresh',
                 'shared_rank_confirm','shared_rank_fresh',
                 'exhaustive_orders_confirm','exhaustive_orders_fresh']:
        raw=read_rows(stem)
        if not raw:continue
        records=[]
        for r in raw:
            for metric in ['task_case_prefix','task_case_full_line','all_case_full_line','task_case_literal_line']:
                acc=np.mean([score(p,y,r['task'],metric) for p,(x,y) in zip(r['predictions'],r['queries'])])
                records.append(dict(task=r['task'],seed=r['seed'],method=r['config'],suite=r['suite'],metric=metric,acc=float(acc)))
        df=pd.DataFrame(records);suites=list(df.groupby('suite'))
        if 'loto' in df.suite.values:suites.append(('heldout_known',df[(df.suite=='loto')&~df.task.isin(DEV_TASKS)]))
        stats={}
        for suite,part in suites:
            stats[suite]={}
            for metric,group in part.groupby('metric'):
                table=group.pivot(index=['task','seed'],columns='method',values='acc').dropna()
                counts=table.groupby(level='task').size();table=table.loc[table.index.get_level_values('task').isin(counts[counts==3].index)]
                if not len(table):continue
                means=table.groupby(level='task').mean();entry=dict(n_tasks=len(means),means=means.mean().to_dict(),paired={})
                for a,b in [('combined','combined_no_correction'),('combined','dictionary_free'),('combined','original'),
                            ('best_correction','original'),('best_correction','matched_z'),
                            ('best_correction','plain_original'),('best_correction','matched_no_correction'),
                            ('best_correction','native_original'),('best_correction','dictionary_free'),
                            ('best_correction','frozen_cass'),
                            ('best_correction','position_cass'),
                            ('exhaustive_cass','frozen_cass'),('exhaustive_cass','exhaustive_off'),
                            ('teacher_route','frozen_cass'),('teacher_route','residual_route'),
                            ('teacher_route','teacher_route_off'),('teacher_route','icl4'),
                            ('rank0','rank1'),('rank2','rank1'),('rank4','rank1'),
                            ('rank0','rank0_off'),('rank1','rank1_off'),
                            ('rank2','rank2_off'),('rank4','rank4_off'),
                            ('oneshot','icl1'),('chat_selected_cass','plain_original_cass'),
                            ('chat_selected_cass','chat_z'),('adaptive_cass','fixed_policy_no_correction')]:
                    if a in means and b in means:entry['paired'][a+'__'+b]=paired(means[a],means[b])
                stats[suite][metric]=entry
        result['experiments'][stem]=stats
    (HERE/'metric_robustness.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Metric robustness written for',len(result['experiments']),'experiments.')

if __name__=='__main__':main()
