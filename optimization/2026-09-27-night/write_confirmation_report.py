"""Complete frozen-configuration confirmation on both query splits and metrics."""
import json
from common import HERE, ROOT


def main():
    data=json.loads((HERE/'extension_comparisons.json').read_text())
    suites=[('nondevelopment_loto','Non-development LOTO20'),('novel','Novel15'),('compound','Compound10')]
    methods=['original','combined','combined_no_correction','dictionary_free_refined','oneshot','icl1','icl4']
    headers=['Submitted operator','Contextual','Correction off','Dictionary-free','One-demo CASS','ICL1','ICL4']
    lines=['# Frozen-configuration confirmation', '',
        'Both query splits use the same frozen development-selected settings and '
        'four demonstrations drawn with seeds 20/21/22. The additional split '
        'contains 50 target queries disjoint from that target’s original evaluation '
        'and demonstration/development inputs. Target-task mining is excluded '
        'before building each LOTO dictionary. Development can use activations '
        'of other known skills, and the original benchmark had already been '
        'inspected earlier in the research process. These are explicitly '
        'non-development targets and disjoint queries, not a claim of a wholly '
        'unseen research benchmark.', '',
        'Every row below is steering-only with no routing. The correction-off '
        'arm retains dictionary-based direction scaling. The separate dictionary-free '
        'baseline is selected from 552 development settings. The one-demo tier '
        'retains one demonstration in its query prompt and has a different '
        'serving cost. The LOTO20 split is primary; LOTO32 summaries remain '
        'available in the source JSON.', '']
    for split,split_label in [('confirm','Original evaluation queries'),('fresh','Additional 50 queries per task')]:
        for metric,metric_label in [('submitted_prefix','Submitted prefix metric'),
                                    ('task_case_literal_line','Case-aware literal-first-line metric')]:
            lines += [f'## {split_label}: {metric_label}', '',
                      '| Suite | '+' | '.join(headers)+' |', '|---|'+'---:|'*len(methods)]
            for suite,label in suites:
                entry=data['splits'][split][suite][metric]
                assert all(entry['means'][m]['n_tasks']=={'nondevelopment_loto':20,'novel':15,'compound':10}[suite] for m in methods)
                lines.append('| '+label+' | '+' | '.join(f"{entry['means'][m]['accuracy']:.4f}" for m in methods)+' |')
            lines += ['', '| Suite | Contextual minus submitted, pp (95% CI) | Correction gain, pp (95% CI) |',
                      '|---|---:|---:|']
            for suite,label in suites:
                entry=data['splits'][split][suite][metric]['comparisons']
                cells=[]
                for key in ['combined__original','combined__combined_no_correction']:
                    r=entry[key];lo,hi=r['ci95']
                    cells.append(f"{100*r['diff']:+.2f} [{100*lo:.2f}, {100*hi:.2f}]")
                lines.append('| '+label+' | '+' | '.join(cells)+' |')
            lines.append('')
    routing_path=HERE/'extension_routing_analysis.json'
    if routing_path.exists():
        routing=json.loads(routing_path.read_text())
        lines += ['## Portability of the submitted routing threshold', '',
            'This post-hoc check applies residual>0.7 unchanged after both query '
            'splits are complete. All contextual signature norms exceed the '
            'submitted5.125 threshold, so the replacement branch is unused. '
            'The full and correction-off arms use identical ICL4 outputs on '
            'every escalated case. Accuracy gains over steering-only therefore '
            'include payment for four-example prompting on the stated fraction '
            'of task/seed cases. This is an operational policy check, separate '
            'from a calibrated failure-probability claim.', '']
        for split,label in [('confirm','Original queries'),('fresh','Additional queries')]:
            for metric,mlabel in [('submitted_prefix','Submitted prefix'),('task_case_literal_line','Literal first line')]:
                lines += ['### '+label+' / '+mlabel, '',
                    '| Suite | Routed contextual | Same route, correction off | ICL4 | ICL4 fraction | Routed correction gain, pp [95% CI] |',
                    '|---|---:|---:|---:|---:|---:|']
                for suite,name in [('heldout_known','Non-development LOTO20'),('novel','Novel15'),('compound','Compound10')]:
                    r=routing['splits'][split]['suites'][suite][metric];v=r['means'];lo,hi=r['routed_correction']['ci95']
                    lines.append(f"| {name} | {v['routed']:.4f} | {v['routed_off']:.4f} | {v['icl4']:.4f} | "
                        f"{r['icl4_fraction']:.3f} | {100*r['routed_correction']['diff']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
                lines.append('')
        lines += ['All task/seed decisions and paired comparisons are retained '
                  'in `extension_routing_analysis.json` with source hashes.', '']
    lines += ['## Interpretation and provenance', '',
        'The dictionary-free rows above belong to the initial 552-configuration '
        'search. A further development-only prompt-position sweep strengthens '
        'that control: original-query scores become0.4379/0.6487/0.3055 on '
        'LOTO20/Novel15/Compound10, and additional-query scores become '
        '0.4757/0.6444/0.3180. The contextual development winner stays unchanged. '
        'Complete matched comparisons and intervals for this stronger control '
        'are in `followup_evidence.md`; the figures with the '
        '`_position_control` suffix show it explicitly. The Novel difference '
        'from a separately tuned dictionary-free system is much smaller than '
        'the matched correction-off intervention, which asks a different question.', '',
        'The overall extraction/intervention improvement over the submitted '
        'operator persists on the additional inputs in all three suites. The '
        'smaller projected-correction gain has positive intervals on LOTO and '
        'Novel under both metrics. For Compound on additional queries, its '
        'submitted-prefix interval includes zero, while the stricter '
        'literal-first-line interval is positive. Both are retained; significance '
        'from one metric is not substituted for the other.', '',
        'The submitted metric matches the target at the beginning of the first '
        'answer line after word-edge punctuation normalization. The literal '
        'metric requires the entire whitespace-normalized first line to equal '
        'the target, preserving punctuation and signs and preserving case for '
        'case-dependent tasks. Neither metric requires all subsequent explanatory '
        'lines to be empty. Supplementary metric checks did not select settings.', '',
        'Intervals use 50,000 paired task-bootstrap draws, averaging the three '
        'demonstration seeds before resampling tasks. They describe each stated '
        'suite, not simultaneous family-wise coverage across all exploratory '
        'comparisons. Source hashes, saved predictions and all paired baseline '
        'comparisons are in `extension_comparisons.json`; input-split and '
        'generation-to-score checks are in `prediction_ledger_validation.json`.', '']
    target=ROOT/'rebuttal/2026-09-27-optimized/confirmation_evidence.md'
    target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
