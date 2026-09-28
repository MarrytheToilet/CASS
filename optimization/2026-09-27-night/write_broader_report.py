"""Render all metric families and tasks, keeping method configurations separate."""
import json
from common import HERE, ROOT

LABELS = {
    'classification_accuracy': 'Classification accuracy',
    'entity_token_f1': 'Entity extraction token F1',
    'reading_token_f1': 'Reading comprehension token F1',
    'long_structured_whole_answer': 'Long structured first-answer-line accuracy',
    'code_expression_execution': 'Code-expression execution accuracy',
    'code_multistep_execution': 'Composed code-expression execution accuracy',
}


def main():
    data = json.loads((HERE/'broader_analysis.json').read_text())
    methods = ['original_cass', 'combined', 'combined_no_correction', 'dictionary_free', 'oneshot', 'icl1', 'icl4']
    headers = ['Submitted', 'Contextual', 'Correction off', 'Dictionary-free', 'One-demo CASS', 'ICL1', 'ICL4']
    lines = ['# Broader-task evidence', '',
        'All 17 predeclared tasks are included. Settings were selected on the '
        'original 12 development targets and frozen before these evaluations. '
        'Each task uses four adaptation examples and three demonstration seeds. '
        'The one-demo tier retains the first of those examples in every query. '
        'Means below first average seeds within task and then tasks within each '
        'metric family. Metrics with different meanings are never pooled.', '',
        'Classification uses 100 queries per task; entity extraction and reading '
        'comprehension also use 100 queries. Structured output and code probes '
        'use 60 queries per task. F1 is against the single reference in the '
        'repository’s dataset conversion, not the full official multi-reference '
        'benchmark score. Long outputs contain 20–30 words and require the entire '
        'extracted first answer line to match after whitespace normalization; '
        'the parser removes an opening code fence and ignores later lines. Code probes parse '
        'and execute a whitelisted expression and require an operation node, '
        'not just a matching numeric literal. The composed-reference min(sorted(xs)[:3]) '
        'case simplifies to min(xs); execution success does not establish an '
        'internal three-stage procedure.', '',
        'These probes cover semantic-label/F1 evaluation, longer structured '
        'outputs, and controlled code/multi-step execution. They do not measure '
        'unrestricted prose generation, software engineering or real tool use.', '',
        '## Complete plain-prompt comparisons', '',
        '| Metric family | Tasks | '+' | '.join(headers)+' |',
        '|---|---:|'+'---:|'*len(methods)]
    for family in LABELS:
        row = next(r for r in data['families'] if r['metric_family']==family)
        counts = [row['means'][m]['n_tasks'] for m in methods]
        assert len(set(counts))==1, (family, 'incomplete primary comparison')
        lines.append('| '+LABELS[family]+f' | {counts[0]} | '+
                     ' | '.join(f"{row['means'][m]['score']:.4f}" for m in methods)+' |')
    lines += ['',
        'The contextual correction and its ablation use identical signatures, '
        'norm scaling, prompts, layers, schedules and inputs. The independently '
        'tuned dictionary-free baseline uses the winning setting from 552 '
        'development configurations. The broader gains over the submitted '
        'operator principally reflect the changed extraction/intervention; '
        'the extra projected correction does not improve every metric family. '
        'In the code probes, retaining a demonstration provides a usable format '
        'tier, while the zero-demonstration contextual arm has zero execution '
        'accuracy. The one-demo CASS tier is also compared directly to ICL1, '
        'which is stronger here. These are distinct cost/quality points.', '',
        '## Every task', '',
        '| Task | '+' | '.join(headers)+' |',
        '|---|'+'---:|'*len(methods)]
    indexed = {(r['task'], r['method']):r for r in data['per_task']}
    for task in sorted({r['task'] for r in data['per_task']}):
        lines.append('| '+task+' | '+' | '.join(f"{indexed[task,m]['score']:.4f}" for m in methods)+' |')
    lines += ['', '## Routing and uncertainty', '',
        'The submitted norm threshold 5.125 and residual escalation threshold '
        '0.7 are frozen. All 17 broader tasks escalate to ICL4 under that '
        'norm-plus-residual policy. This diagnoses the mismatch to the original '
        'short-task library; it is not evidence of zero-context accuracy or '
        'serving savings on those routed queries. Operator comparisons above '
        'are steering-only; fallback predictions are recorded separately.', '',
        'Per-task paired intervals in `broader_analysis.json` bootstrap queries '
        'after averaging the three fixed demonstration seeds. They are '
        'conditional on those demonstrations. Cross-task intervals, where '
        'available, resample task means within a metric family and are based '
        'on few tasks. Raw predictions are rescored and all compared examples '
        'and queries are checked for identity.', '',
        '## Separately frozen follow-ups', '',
        'Native-template, multi-position and correction-geometry variants use '
        'their own development selections. Each complete condition is reported '
        'as a whole method across metric families; no per-family best '
        'configuration is combined into a synthetic CASS result.', '']
    for prefix, label in [('native_', 'Native-template Llama'), ('position_', 'Position refinement'),
                          ('projector_', 'Weighted/rank correction refinement')]:
        required=[prefix+'best_correction',prefix+'matched_no_correction',prefix+'dictionary_free',prefix+'icl4']
        complete=all((t,m) in indexed for t in sorted({r['task'] for r in data['per_task']}) for m in required)
        if not complete:
            lines.append(label+': pending complete task/seed coverage.');lines.append('');continue
        lines += [f'### {label}', '', '| Metric family | Best correction | Correction off | Dictionary-free | ICL4 |',
                  '|---|---:|---:|---:|---:|']
        for family in LABELS:
            row=next(r for r in data['families'] if r['metric_family']==family)
            lines.append('| '+LABELS[family]+' | '+' | '.join(f"{row['means'][m]['score']:.4f}" for m in required)+' |')
        lines.append('')
    target=ROOT/'rebuttal/2026-09-27-optimized/broader_evidence.md'
    target.write_text('\n'.join(lines))
    print(target)


if __name__=='__main__':main()
