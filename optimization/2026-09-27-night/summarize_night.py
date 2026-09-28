"""Live summaries plus generation-to-metric consistency checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,DEV_TASKS
from cass.evaluate import accuracy
from broader_tasks import score


def read_rows(stem):
    path=HERE/f'{stem}.jsonl'
    if not path.exists():return []
    lines=path.read_text().splitlines();rows=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1:break  # a live writer may be between write and flush
            raise
    return rows


def paired(a,b):
    diff=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(20260927)
    boots=diff[rng.integers(len(diff),size=(50000,len(diff)))].mean(1)
    return dict(n_tasks=len(diff),diff=float(diff.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())


def evaluated(stem,contrasts):
    rows=read_rows(stem)
    if not rows:return None
    for row in rows:
        value=accuracy(row['predictions'],[y for x,y in row['queries']],case_sensitive='+' in row['task'])
        assert abs(value-row['acc'])<1e-12,(stem,row['task'],row['seed'],row['config'])
    df=pd.DataFrame(rows)
    assert not df.duplicated(['task','seed','config']).any()
    out=dict(n_cells=len(df),n_tasks=int(df.task.nunique()),generation_metrics_verified=True,suites={})
    groups=list(df.groupby('suite'))
    if 'loto' in df.suite.values:
        groups.append(('heldout_known',df[(df.suite=='loto') & ~df.task.isin(DEV_TASKS)]))
    for suite,part in groups:
        if not len(part):continue
        table=part.pivot(index=['task','seed'],columns='config',values='acc')
        # Whole tasks with all three matched seeds in all current conditions.
        table=table.dropna();counts=table.groupby(level='task').size()
        table=table.loc[table.index.get_level_values('task').isin(counts[counts==3].index)]
        if not len(table):continue
        means=table.groupby(level='task').mean()
        rec=dict(n_complete_tasks=len(means),means={k:float(v) for k,v in means.mean().items()},paired={})
        for a,b in contrasts:
            if a in means and b in means:rec['__'.join([a,b])]=paired(means[a],means[b])
        out['suites'][suite]=rec
    return out


def main():
    result={}
    for stem in ['llama_dev','qwen_oracle_dev','qwen_chat_oracle_dev','llama_variants_dev','signature_dev','combined_dev','dictionary_free_dev','llama_native_dev','position_dev','projector_dev','qwen3-4b_contrast_dev','qwen25-3b_contrast_dev']:
        rows=read_rows(stem)
        if not rows:continue
        df=pd.DataFrame(rows)
        stats=df.groupby('config').agg(mean=('acc','mean'),n=('acc','size'))
        stats=stats[stats.n==stats.n.max()].sort_values('mean',ascending=False)
        result[stem]=dict(n_cells=len(df),n_tasks=int(df.task.nunique()),top=stats.head(12).reset_index().to_dict('records'))
    for stem,contrasts in [
        ('llama_confirm',[('best_correction','original'),('best_correction','matched_z'),('best_correction','matched_no_correction'),('best_correction','best_overall')]),
        ('signature_confirm',[('best_correction','original'),('best_correction','matched_z'),('best_correction','matched_no_correction'),('best_correction','best_overall')]),
        ('qwen_confirm',[('chat_selected_cass','plain_original_cass'),('chat_selected_cass','plain_selected_cass'),('chat_selected_cass','chat_z'),('chat_selected_cass','chat_icl4')]),
        ('extension_confirm',[('combined','combined_no_correction'),('combined','dictionary_free'),('oneshot','oneshot_no_correction'),('oneshot','icl1'),('combined','icl4')]),
        ('extension_fresh',[('combined','combined_no_correction'),('combined','dictionary_free'),('combined','original'),('oneshot','icl1'),('oneshot','oneshot_no_correction')]),
        ('adaptive_combined',[('adaptive_cass','frozen_cass'),('adaptive_cass','fixed_policy_no_correction'),('adaptive_cass','adaptive_z')]),
        ('teacher_route_confirm',[('teacher_route','frozen_cass'),('teacher_route','residual_route'),('teacher_route','teacher_route_off'),('teacher_route','icl4')]),
        ('teacher_route_fresh',[('teacher_route','frozen_cass'),('teacher_route','residual_route'),('teacher_route','teacher_route_off'),('teacher_route','icl4')]),
        ('shared_rank_confirm',[(f'rank{r}','rank1') for r in [0,2,4]]+[(f'rank{r}',f'rank{r}_off') for r in [0,1,2,4]]),
        ('shared_rank_fresh',[(f'rank{r}','rank1') for r in [0,2,4]]+[(f'rank{r}',f'rank{r}_off') for r in [0,1,2,4]]),
        ('llama_native_confirm',[('best_correction','native_original'),('best_correction','matched_no_correction'),('best_correction','dictionary_free'),('best_correction','best_overall'),('best_correction','icl4')]),
        ('llama_native_fresh',[('best_correction','native_original'),('best_correction','matched_no_correction'),('best_correction','dictionary_free'),('best_correction','best_overall'),('best_correction','icl4')]),
        ('position_confirm',[('best_correction','frozen_cass'),('best_correction','matched_no_correction'),('best_correction','dictionary_free'),('best_correction','icl4')]),
        ('position_fresh',[('best_correction','frozen_cass'),('best_correction','matched_no_correction'),('best_correction','dictionary_free'),('best_correction','icl4')]),
        ('projector_confirm',[('best_correction','position_cass'),('best_correction','matched_no_correction'),('best_correction','dictionary_free'),('best_correction','frozen_cass'),('best_correction','icl4')]),
        ('projector_fresh',[('best_correction','position_cass'),('best_correction','matched_no_correction'),('best_correction','dictionary_free'),('best_correction','frozen_cass'),('best_correction','icl4')]),
        ('exhaustive_orders_confirm',[('exhaustive_cass','frozen_cass'),('exhaustive_cass','exhaustive_off'),('exhaustive_cass','icl4')]),
        ('exhaustive_orders_fresh',[('exhaustive_cass','frozen_cass'),('exhaustive_cass','exhaustive_off'),('exhaustive_cass','icl4')]),
        ('qwen3-4b_contrast_confirm',[('best_correction','plain_original'),('best_correction','native_shuffle24'),('best_correction','matched_no_correction'),('best_correction','best_overall')]),
        ('qwen25-3b_contrast_confirm',[('best_correction','plain_original'),('best_correction','native_shuffle24'),('best_correction','matched_no_correction'),('best_correction','best_overall')]),
        ('qwen3-4b_contrast_fresh',[('best_correction','plain_original'),('best_correction','native_shuffle24'),('best_correction','matched_no_correction'),('best_correction','best_overall')]),
        ('qwen25-3b_contrast_fresh',[('best_correction','plain_original'),('best_correction','native_shuffle24'),('best_correction','matched_no_correction'),('best_correction','best_overall')]),
        ('adaptive_eval',[('adaptive_cass','original_cass'),('adaptive_cass','fixed_policy_z'),('adaptive_cass','adaptive_z'),('adaptive_cass','fixed_policy_no_correction')])]:
        row=evaluated(stem,contrasts)
        if row:result[stem]=row
    for broad_stem in ['broader_llama31-8b','broader_signature_llama31-8b','broader_extensions','broader_native','broader_position','broader_projector']:
        rows=read_rows(broad_stem)
        if not rows:continue
        for r in rows:
            vals=[score(r['task'],p,y) for p,(x,y) in zip(r['predictions'],r['queries'])]
            assert abs(np.mean(vals)-r['score'])<1e-12
        df=pd.DataFrame(rows)
        result[broad_stem]=dict(n_cells=len(df),per_task=df.groupby(['task','method']).score.agg(['mean','count']).reset_index().to_dict('records'),
                               generation_metrics_verified=True,scope='Live partial rows; inspect count before comparing seeds.')
    (HERE/'live_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,value in result.items():
        print(key,value.get('n_cells'))
        if 'suites' in value:print(json.dumps(value['suites'],indent=2))
    return result


if __name__=='__main__':main()
