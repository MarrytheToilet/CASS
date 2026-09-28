"""Report complete frozen follow-up variants without per-suite winner mixing."""
import hashlib
import json
from common import HERE,ROOT,DEV_TASKS

SUITES=[('heldout_known','Non-development LOTO20',20),('novel','Novel15',15),('compound','Compound10',10)]
BRANCHES=[
 ('llama_native','Native-template Llama',
  ['best_correction','matched_no_correction','best_overall','dictionary_free','native_original','icl1','icl4'],
  ['Correction setting','Correction off','Overall context setting','Dictionary-free','Native submitted operator','ICL1','ICL4']),
 ('position','Prompt-position refinement',
  ['best_correction','matched_no_correction','best_overall','dictionary_free','frozen_cass','icl4'],
  ['Correction setting','Correction off','Overall context setting','Dictionary-free','Original contextual setting','ICL4']),
 ('projector','Weighted/rank correction refinement',
  ['best_correction','matched_no_correction','best_overall','dictionary_free','position_cass','frozen_cass','icl4'],
  ['Correction setting','Correction off','Overall context setting','Dictionary-free','Position setting','Original contextual setting','ICL4']),
 ('exhaustive_orders','All six demonstration orders',
  ['exhaustive_cass','exhaustive_off','frozen_cass','frozen_off','icl4'],
  ['All orders','Correction off','Original contextual setting','Original correction off','ICL4']),
 ('teacher_route','Demonstration-only serving-tier selection',
  ['teacher_route','teacher_route_off','residual_route','frozen_cass','icl4'],
  ['Demo-loss route','Same route, correction off','Residual0.7 route','Pure contextual steering','ICL4']),
 ('shared_rank','Shared-rank mechanism ablation',
  ['rank0','rank0_off','rank1','rank1_off','rank2','rank2_off','rank4','rank4_off','icl4'],
  ['Rank0','Rank0 off','Rank1','Rank1 off','Rank2','Rank2 off','Rank4','Rank4 off','ICL4']),
]


def interval(value):
    lo,hi=value['ci95']
    return f"{100*value['diff']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"


def main():
    live=json.loads((HERE/'live_summary.json').read_text())
    robust=json.loads((HERE/'metric_robustness.json').read_text())['experiments']
    lines=['# Separately frozen follow-up variants', '',
        'Each development-selected variant transfers one fixed setting to every '
        'target. All-order enumeration is parameter-free given the original '
        'contextual setting. Results remain separate from the initial '
        'contextual confirmation and from each other. The same four examples '
        'and three demonstration seeds are used in compared arms. No table '
        'combines the best method from each suite or task.', '',
        'The native-template condition adds the same generic system instruction '
        'to both steering and prompting baselines; its native submitted-operator '
        'arm also remakes the dictionary under that prompt format. Correction-off '
        'keeps dictionary-based direction calibration. The independently tuned '
        'dictionary-free arm omits the library. “Overall context setting” denotes '
        'the best setting within the contextual search, including zero correction; '
        'it does not select across unrelated methods on evaluation labels.', '',
        'Prefix and strict case-aware literal-first-line metrics are reported '
        'separately. Paired intervals use task means over three seeds and 50,000 '
        'bootstrap draws. The additional-query split has50 disjoint target queries '
        'per task. Timing and memory measurements for the earlier operator are '
        'not automatically assigned to a different prompt or implementation.', '']
    sources={}
    for prefix,label,methods,headers in BRANCHES:
        lines += ['## '+label, '']
        if prefix=='shared_rank':
            lines += ['This is a fixed-setting mechanism ablation of the contextual '
                'extension. The original null contrast, gain, schedule and correction '
                'rule stay frozen while the number of removed shared directions '
                'changes at each selected layer. Bases, sparse supports and norm calibration are rebuilt '
                'consistently for each rank. No evaluation label selects a rank, '
                'and the ranks do not each receive separately tuned hyperparameters. '
                'The rank1 predictions are the main contextual results, with '
                'matched input checks and a numerical reconstruction check.', '']
        if prefix=='teacher_route':
            lines += ['Compare the mean fourfold answer NLL of fixed CASS with '
                'three-shot ICL, using only the supplied four demonstrations. '
                'Route to ICL4 only when its ICL3 validation loss is strictly '
                'lower. There is no fitted margin or evaluation-label selection. '
                'The fixed CASS setting is used, not the adaptive grid minimum. '
                'This separate variant requires112 logical extraction sequences '
                'and eight teacher-forced forwards; it does not inherit the '
                'earlier adaptation timing. Its residual-policy comparator uses '
                'the submitted0.7 threshold without retuning.', '']
        selected=HERE/(prefix+'_selected.json')
        if selected.exists():
            settings=json.loads(selected.read_text())
            lines += ['Development-frozen settings:', '', '```json',
                      json.dumps({k:v for k,v in settings.items() if k not in ['all_development_means']},indent=2), '```', '']
        for split,split_label in [('confirm','Original queries'),('fresh','Additional queries')]:
            stem=prefix+'_'+split
            if not (HERE/(stem+'_done.json')).exists() or stem not in live or stem not in robust:
                lines += [split_label+': pending complete confirmation and analysis.', ''];continue
            path=HERE/(stem+'.jsonl');sources[stem]=hashlib.sha256(path.read_bytes()).hexdigest()
            if prefix=='teacher_route':
                records=[json.loads(s) for s in path.read_text().splitlines()]
                lines += ['| Suite | Demo-loss ICL4 fraction | Residual-policy ICL4 fraction |',
                          '|---|---:|---:|']
                for suite,name,n in SUITES:
                    if suite=='heldout_known':current=[r for r in records if r['suite']=='loto' and r['task'] not in DEV_TASKS]
                    else:current=[r for r in records if r['suite']==suite]
                    fractions=[]
                    for method in ['teacher_route','residual_route']:
                        arm=[r for r in current if r['config']==method];assert len(arm)==n*3
                        fractions.append(sum(r['uses_icl'] for r in arm)/len(arm))
                    lines.append(f'| {name} | {fractions[0]:.3f} | {fractions[1]:.3f} |')
                lines.append('')
            for metric,metric_label in [('submitted_prefix','Submitted prefix metric'),('task_case_literal_line','Case-aware literal-first-line metric')]:
                lines += ['### '+split_label+' / '+metric_label, '',
                          '| Suite | '+' | '.join(headers)+' |','|---|'+'---:|'*len(methods)]
                entries={}
                for suite,name,n in SUITES:
                    if metric=='submitted_prefix':
                        entry=live[stem]['suites'][suite];assert entry['n_complete_tasks']==n
                        pairs={k:v for k,v in entry.items() if '__' in k}
                    else:
                        entry=robust[stem][suite][metric];assert entry['n_tasks']==n;pairs=entry['paired']
                    assert all(m in entry['means'] for m in methods)
                    lines.append('| '+name+' | '+' | '.join(f"{entry['means'][m]:.4f}" for m in methods)+' |')
                    entries[name]=pairs
                lines += ['', '| Suite | Paired comparison | Difference, pp [95% CI] |', '|---|---|---:|']
                for name,pairs in entries.items():
                    for key,value in pairs.items():
                        lines.append('| '+name+' | '+key.replace('__',' − ')+' | '+interval(value)+' |')
                lines.append('')
    lines += ['## Source generation hashes', '', '```json',json.dumps(sources,indent=2),'```','']
    target=ROOT/'rebuttal/2026-09-27-optimized/followup_evidence.md'
    target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
