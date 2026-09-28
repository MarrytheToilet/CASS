"""Render every scaling regime and its independently evaluated risk calibration."""
import json
from common import HERE, ROOT

STEMS = [
    ('scale_eval', 'Submitted operator, recomputed shared direction'),
    ('scale_eval_fixed_shared', 'Submitted operator, frozen shared direction'),
    ('scale_context', 'Contextual operator, recomputed shared direction'),
    ('scale_context_fixed_shared', 'Contextual operator, frozen shared direction'),
]


def main():
    routing_path=HERE/'scaling_routing_analysis.json'
    routing=json.loads(routing_path.read_text()) if routing_path.exists() else None
    lines = ['# Dictionary scaling evidence', '',
        'The same 32 natural-language skills are retained while 224 explicitly '
        'synthetic, correlated skills are added in a frozen stratified order. '
        'They cover affine/modular arithmetic and indexed list/character '
        'selection. This is a correlated-expansion stress test, not a sample of '
        '256 independent natural-language domains. Novel15 and Compound10 use '
        'matched queries and three fixed demonstration seeds at every size.', '',
        'Two update policies were specified before observing scaling results: '
        'recompute the shared direction at each size, or retain the initial '
        '32-skill direction and all original bases/anchors exactly. Both full '
        'coding and a projection-ranked 32-block shortlist are evaluated. '
        'Shortlisting time is included in the reported online coding time. '
        'Offline dictionary construction, Gram construction and memory are '
        'reported separately.', '',
        'Accuracy averages seeds within task, then tasks. Intervals resample '
        'the 25 paired task means 50,000 times with seed20260927. Support overlap '
        'is Jaccard similarity; constituent recall is against the manually '
        'specified Compound skills. Support stability and execution accuracy '
        'measure different properties.', '']
    sources = {}
    for stem, label in STEMS:
        path = HERE/(stem+'_analysis.json')
        if not (HERE/(stem+'_done.json')).exists() or not path.exists():
            lines += ['## '+label, '', 'Pending complete analysis.', '']
            continue
        data = json.loads(path.read_text());sources[stem] = data['source_sha256']
        assert data['generation_metrics_verified'] and data['matched_queries_verified']
        lines += ['## '+label, '',
            '| Skills | Solver | Novel accuracy | Compound accuracy | Overall change vs32, pp [95% CI] | Support Jaccard to32 | Across seeds | Constituent recall | Coding, ms |',
            '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['scaling']:
            lo,hi=r['paired_accuracy_change_ci95']
            lines.append(f"| {r['size']} | {r['mode']} | {r['accuracy_by_suite']['novel']:.4f} | "
                f"{r['accuracy_by_suite']['compound']:.4f} | {100*r['paired_accuracy_change_from32']:+.2f} "
                f"[{100*lo:+.2f}, {100*hi:+.2f}] | {r['support_jaccard_to32']:.3f} | "
                f"{r['support_jaccard_across_seeds']:.3f} | {r['compound_constituent_recall']:.3f} | "
                f"{1000*r['coding_median_seconds']:.1f} |")
        lines += ['',
            'The stricter case-aware first-answer-line scores are retained in '
            f'`{stem}_analysis.json`; changing the accuracy metric changes '
            f"{sum(r['failure_label_changes_under_literal_metric'] for r in data['scaling'])} "
            'task/size/solver failure labels in this study.', '',
            '### Held-out-task risk calibration', '',
            'Eight predeclared tasks fit a standardized L2 logistic risk map '
            '(C=1); the remaining 17 tasks evaluate it. Failure is mean '
            'three-seed task accuracy below0.5; this differs from the paper’s '
            'oracle-normalized recovery endpoint. Compare a frozen32 mapping '
            'with a mapping refitted at each size, and with the corresponding '
            'constant training-prior predictor. If training labels have only '
            'one class, the map is that constant prior. Brier score is primary '
            '(lower is better); AUROC measures discrimination and five-bin ECE '
            'is only a small-sample diagnostic. A raw reconstruction residual '
            'is not a failure probability. This is an auxiliary supervised '
            'task-level calibration exercise, separate from the frozen LLM.', '',
            '| Skills | Solver | Features | Policy | Failure rate | Brier [95% CI] | AUROC | ECE5 |',
            '|---:|---|---|---|---:|---:|---:|---:|']
        for r in data['calibration']:
            if r['size'] not in [32,256]:continue
            # At size32 the refitted and frozen policies are identical.
            if r['size']==32 and r['policy']=='refit':continue
            lo,hi=r['brier_ci95'];auc='—' if r['auc'] is None else f"{r['auc']:.3f}"
            lines.append(f"| {r['size']} | {r['mode']} | {r['features']} | {r['policy']} | "
                f"{r['failure_rate']:.3f} | {r['brier']:.4f} [{lo:.4f}, {hi:.4f}] | {auc} | {r['ece5']:.3f} |")
        lines += ['', 'All four sizes, task-level probabilities and paired '
            'Brier differences versus the constant-prior control are retained '
            f'in `{stem}_analysis.json`. Intervals are conditional on the eight '
            'calibration tasks, and the failure prevalence can change with '
            'dictionary size; a smaller Brier value across sizes alone does '
            'not establish improved discrimination.', '',
        ]
        crossfit=HERE/(stem+'_crossfit_calibration.json')
        if crossfit.exists():
            cf=json.loads(crossfit.read_text());assert cf['source_sha256']==data['source_sha256']
            lines += ['### Additional leave-one-task-out sensitivity check', '',
                'This fixed-rule analysis was added after observing that all '
                'eight contextual calibration tasks succeeded at size32. It '
                'retains the original held-out17 result above. Each of25 tasks '
                'is evaluated by a mapping fitted to the other24 tasks; '
                'features, C=1 and the failure threshold remain unchanged. '
                'The frozen32 map also excludes the target at size32. It is '
                'post-hoc cross-fitting, not a new untouched test set. The '
                'intervals condition on cross-fitted predictions and omit '
                'refitting uncertainty from overlapping training sets.', '',
                '| Skills | Solver | Features | Policy | Brier [95% CI] |',
                '|---:|---|---|---|---:|']
            for r in cf['results']:
                if r['size'] not in [32,256] or (r['size']==32 and r['policy']=='refit'):continue
                lo,hi=r['brier_ci95']
                lines.append(f"| {r['size']} | {r['mode']} | {r['features']} | {r['policy']} | {r['brier']:.4f} [{lo:.4f}, {hi:.4f}] |")
            lines += ['',
                'Pooled cross-fit probability AUROC is not used to claim '
                'discrimination: leaving out a positive label lowers the '
                'training prior, while leaving out a negative raises it. '
                'Consequently a leave-one-out constant-prior predictor can '
                'have pooled AUROC zero without a meaningful reversed '
                'risk signal. Common raw-score AUROCs for residual and '
                'negative norm, all paired Brier contrasts and task-level '
                f'predictions are in `{stem}_crossfit_calibration.json`.', '']
        if routing:
            assert routing['sources'][stem]==data['source_sha256']
            lines += ['### Portability of the submitted routing thresholds', '',
                'This additional check applies the existing norm<5.125 and '
                'residual>0.7 rules without retuning. Norm selects the matched '
                'saved replacement output; residual selects matched ICL4. '
                'Escalation takes precedence in the combined rule. Branch '
                'fractions average the75 task/seed cases, while accuracy first '
                'averages seeds within task. The pure ICL4 reference is0.7316 '
                'on these25 tasks. Changes in routed accuracy include changes '
                'in the fraction of cases paying for full ICL.', '',
                '| Skills | Policy | Routed accuracy | ICL4 fraction | Replacement fraction | Routes changed from32 |',
                '|---:|---|---:|---:|---:|---:|']
            for r in routing['results']:
                if r['stem']!=stem or r['mode']!='full' or r['size'] not in [32,256]:continue
                lines.append(f"| {r['size']} | {r['policy']} | {r['accuracy']:.4f} | "
                    f"{r['fractions']['icl4']:.3f} | {r['fractions']['replacement']:.3f} | {r['route_change_from32']:.3f} |")
            lines += ['', 'All sizes, both solvers, task/seed decisions and paired '
                'accuracy intervals are retained in `scaling_routing_analysis.json`. '
                'This measures portability of an operational threshold and does '
                'not turn a reconstruction residual into a failure probability.', '']
        lines += ['### Offline resources and coherence', '',
            '| Skills | Maximum coherence | Median coherence | Basis columns | Dictionary build, s | Gram build, s | Stored bases, MiB | Solver basis copy, MiB | Gram, MiB |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        offline=json.loads((HERE/(stem+'_offline.json')).read_text())
        for r in offline:
            lines.append(f"| {r['size']} | {r['maximum_coherence']:.6f} | {r['median_coherence']:.6f} | "
                f"{r['total_columns']} | {r['dictionary_seconds']:.3f} | {r['gram_seconds']:.3f} | "
                f"{r['joint_basis_bytes']/2**20:.2f} | {r['concatenated_basis_bytes']/2**20:.2f} | {r['gram_bytes']/2**20:.2f} |")
        lines += ['', 'These are array byte counts, not total process memory. '
            'The shared model, anchors, extraction cache, Python metadata and '
            'temporary SVD storage are excluded. The largest coherence, not '
            'the median, enters a global mutual-coherence guarantee.', '']
    lines += ['## Source generation ledgers', '', '```json', json.dumps(sources,indent=2), '```', '']
    target=ROOT/'rebuttal/2026-09-27-optimized/scaling_evidence.md'
    target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
