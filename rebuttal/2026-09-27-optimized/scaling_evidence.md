# Dictionary scaling evidence

The same 32 natural-language skills are retained while 224 explicitly synthetic, correlated skills are added in a frozen stratified order. They cover affine/modular arithmetic and indexed list/character selection. This is a correlated-expansion stress test, not a sample of 256 independent natural-language domains. Novel15 and Compound10 use matched queries and three fixed demonstration seeds at every size.

Two update policies were specified before observing scaling results: recompute the shared direction at each size, or retain the initial 32-skill direction and all original bases/anchors exactly. Both full coding and a projection-ranked 32-block shortlist are evaluated. Shortlisting time is included in the reported online coding time. Offline dictionary construction, Gram construction and memory are reported separately.

Accuracy averages seeds within task, then tasks. Intervals resample the 25 paired task means 50,000 times with seed20260927. Support overlap is Jaccard similarity; constituent recall is against the manually specified Compound skills. Support stability and execution accuracy measure different properties.

## Submitted operator, recomputed shared direction

| Skills | Solver | Novel accuracy | Compound accuracy | Overall change vs32, pp [95% CI] | Support Jaccard to32 | Across seeds | Constituent recall | Coding, ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | full | 0.3770 | 0.1438 | +0.00 [+0.00, +0.00] | 1.000 | 0.600 | 0.667 | 72.6 |
| 64 | full | 0.3475 | 0.1189 | -2.77 [-5.68, -0.33] | 0.715 | 0.562 | 0.650 | 188.2 |
| 128 | full | 0.2980 | 0.1153 | -5.89 [-10.85, -1.78] | 0.603 | 0.598 | 0.650 | 486.3 |
| 256 | full | 0.1935 | 0.0530 | -14.65 [-22.91, -7.08] | 0.558 | 0.631 | 0.750 | 1318.6 |
| 32 | shortlist32 | 0.3770 | 0.1438 | +0.00 [+0.00, +0.00] | 1.000 | 0.600 | 0.667 | 72.3 |
| 64 | shortlist32 | 0.3489 | 0.1173 | -2.75 [-5.62, -0.35] | 0.715 | 0.562 | 0.650 | 221.8 |
| 128 | shortlist32 | 0.2975 | 0.1136 | -5.98 [-10.95, -1.84] | 0.602 | 0.599 | 0.650 | 248.3 |
| 256 | shortlist32 | 0.1940 | 0.0530 | -14.62 [-22.86, -7.08] | 0.558 | 0.631 | 0.750 | 268.1 |

The stricter case-aware first-answer-line scores are retained in `scale_eval_analysis.json`; changing the accuracy metric changes 0 task/size/solver failure labels in this study.

### Held-out-task risk calibration

Eight predeclared tasks fit a standardized L2 logistic risk map (C=1); the remaining 17 tasks evaluate it. Failure is mean three-seed task accuracy below0.5; this differs from the paper’s oracle-normalized recovery endpoint. Compare a frozen32 mapping with a mapping refitted at each size, and with the corresponding constant training-prior predictor. If training labels have only one class, the map is that constant prior. Brier score is primary (lower is better); AUROC measures discrimination and five-bin ECE is only a small-sample diagnostic. A raw reconstruction residual is not a failure probability. This is an auxiliary supervised task-level calibration exercise, separate from the frozen LLM.

| Skills | Solver | Features | Policy | Failure rate | Brier [95% CI] | AUROC | ECE5 |
|---:|---|---|---|---:|---:|---:|---:|
| 32 | full | residual | frozen32 | 0.647 | 0.2858 [0.1215, 0.4572] | 0.333 | 0.282 |
| 32 | full | norm | frozen32 | 0.647 | 0.2916 [0.1103, 0.4742] | 0.121 | 0.228 |
| 32 | full | joint | frozen32 | 0.647 | 0.2982 [0.1285, 0.4750] | 0.227 | 0.337 |
| 32 | full | constant_prior | frozen32 | 0.647 | 0.2803 [0.1039, 0.4568] | 0.500 | 0.228 |
| 256 | full | residual | frozen32 | 0.882 | 0.1131 [0.0171, 0.2525] | 0.367 | 0.019 |
| 256 | full | residual | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 256 | full | norm | frozen32 | 0.882 | 0.1084 [0.0139, 0.2485] | 0.267 | 0.001 |
| 256 | full | norm | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 256 | full | joint | frozen32 | 0.882 | 0.1191 [0.0182, 0.2640] | 0.333 | 0.100 |
| 256 | full | joint | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 256 | full | constant_prior | frozen32 | 0.882 | 0.1039 [0.0156, 0.2362] | 0.500 | 0.007 |
| 256 | full | constant_prior | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 32 | shortlist32 | residual | frozen32 | 0.647 | 0.2858 [0.1215, 0.4572] | 0.333 | 0.282 |
| 32 | shortlist32 | norm | frozen32 | 0.647 | 0.2916 [0.1103, 0.4742] | 0.121 | 0.228 |
| 32 | shortlist32 | joint | frozen32 | 0.647 | 0.2982 [0.1285, 0.4750] | 0.227 | 0.337 |
| 32 | shortlist32 | constant_prior | frozen32 | 0.647 | 0.2803 [0.1039, 0.4568] | 0.500 | 0.228 |
| 256 | shortlist32 | residual | frozen32 | 0.882 | 0.1131 [0.0171, 0.2525] | 0.367 | 0.019 |
| 256 | shortlist32 | residual | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 256 | shortlist32 | norm | frozen32 | 0.882 | 0.1084 [0.0139, 0.2485] | 0.267 | 0.001 |
| 256 | shortlist32 | norm | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 256 | shortlist32 | joint | frozen32 | 0.882 | 0.1191 [0.0182, 0.2640] | 0.333 | 0.100 |
| 256 | shortlist32 | joint | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |
| 256 | shortlist32 | constant_prior | frozen32 | 0.882 | 0.1039 [0.0156, 0.2362] | 0.500 | 0.007 |
| 256 | shortlist32 | constant_prior | refit | 0.882 | 0.1176 [0.0000, 0.2941] | 0.500 | 0.118 |

All four sizes, task-level probabilities and paired Brier differences versus the constant-prior control are retained in `scale_eval_analysis.json`. Intervals are conditional on the eight calibration tasks, and the failure prevalence can change with dictionary size; a smaller Brier value across sizes alone does not establish improved discrimination.

### Additional leave-one-task-out sensitivity check

This fixed-rule analysis was added after observing that all eight contextual calibration tasks succeeded at size32. It retains the original held-out17 result above. Each of25 tasks is evaluated by a mapping fitted to the other24 tasks; features, C=1 and the failure threshold remain unchanged. The frozen32 map also excludes the target at size32. It is post-hoc cross-fitting, not a new untouched test set. The intervals condition on cross-fitted predictions and omit refitting uncertainty from overlapping training sets.

| Skills | Solver | Features | Policy | Brier [95% CI] |
|---:|---|---|---|---:|
| 32 | full | residual | frozen32 | 0.2298 [0.1490, 0.3196] |
| 32 | full | norm | frozen32 | 0.1706 [0.1040, 0.2459] |
| 32 | full | joint | frozen32 | 0.1810 [0.1109, 0.2593] |
| 32 | full | constant_prior | frozen32 | 0.2188 [0.1424, 0.3142] |
| 256 | full | residual | frozen32 | 0.1304 [0.0900, 0.1849] |
| 256 | full | residual | refit | 0.0806 [0.0072, 0.1904] |
| 256 | full | norm | frozen32 | 0.2055 [0.1317, 0.2821] |
| 256 | full | norm | refit | 0.0793 [0.0074, 0.1854] |
| 256 | full | joint | frozen32 | 0.1959 [0.1268, 0.2688] |
| 256 | full | joint | refit | 0.0799 [0.0078, 0.1865] |
| 256 | full | constant_prior | frozen32 | 0.1187 [0.0787, 0.1769] |
| 256 | full | constant_prior | refit | 0.0799 [0.0069, 0.1892] |
| 32 | shortlist32 | residual | frozen32 | 0.2298 [0.1490, 0.3196] |
| 32 | shortlist32 | norm | frozen32 | 0.1706 [0.1040, 0.2459] |
| 32 | shortlist32 | joint | frozen32 | 0.1810 [0.1109, 0.2593] |
| 32 | shortlist32 | constant_prior | frozen32 | 0.2188 [0.1424, 0.3142] |
| 256 | shortlist32 | residual | frozen32 | 0.1304 [0.0900, 0.1849] |
| 256 | shortlist32 | residual | refit | 0.0806 [0.0072, 0.1904] |
| 256 | shortlist32 | norm | frozen32 | 0.2055 [0.1317, 0.2821] |
| 256 | shortlist32 | norm | refit | 0.0793 [0.0074, 0.1854] |
| 256 | shortlist32 | joint | frozen32 | 0.1959 [0.1268, 0.2688] |
| 256 | shortlist32 | joint | refit | 0.0799 [0.0078, 0.1865] |
| 256 | shortlist32 | constant_prior | frozen32 | 0.1187 [0.0787, 0.1769] |
| 256 | shortlist32 | constant_prior | refit | 0.0799 [0.0069, 0.1892] |

Pooled cross-fit probability AUROC is not used to claim discrimination: leaving out a positive label lowers the training prior, while leaving out a negative raises it. Consequently a leave-one-out constant-prior predictor can have pooled AUROC zero without a meaningful reversed risk signal. Common raw-score AUROCs for residual and negative norm, all paired Brier contrasts and task-level predictions are in `scale_eval_crossfit_calibration.json`.

### Portability of the submitted routing thresholds

This additional check applies the existing norm<5.125 and residual>0.7 rules without retuning. Norm selects the matched saved replacement output; residual selects matched ICL4. Escalation takes precedence in the combined rule. Branch fractions average the75 task/seed cases, while accuracy first averages seeds within task. The pure ICL4 reference is0.7316 on these25 tasks. Changes in routed accuracy include changes in the fraction of cases paying for full ICL.

| Skills | Policy | Routed accuracy | ICL4 fraction | Replacement fraction | Routes changed from32 |
|---:|---|---:|---:|---:|---:|
| 32 | steering | 0.2838 | 0.000 | 0.000 | 0.000 |
| 256 | steering | 0.1373 | 0.000 | 0.000 | 0.000 |
| 32 | norm_only | 0.4255 | 0.000 | 0.440 | 0.000 |
| 256 | norm_only | 0.2459 | 0.000 | 0.333 | 0.107 |
| 32 | residual_only | 0.5475 | 0.640 | 0.000 | 0.000 |
| 256 | residual_only | 0.3586 | 0.440 | 0.000 | 0.253 |
| 32 | combined | 0.6147 | 0.640 | 0.120 | 0.000 |
| 256 | combined | 0.3883 | 0.440 | 0.067 | 0.307 |

All sizes, both solvers, task/seed decisions and paired accuracy intervals are retained in `scaling_routing_analysis.json`. This measures portability of an operational threshold and does not turn a reconstruction residual into a failure probability.

### Offline resources and coherence

| Skills | Maximum coherence | Median coherence | Basis columns | Dictionary build, s | Gram build, s | Stored bases, MiB | Solver basis copy, MiB | Gram, MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.994617 | 0.572833 | 1006 | 1.974 | 0.196 | 62.88 | 62.88 | 7.72 |
| 64 | 0.999183 | 0.566633 | 2024 | 4.105 | 0.624 | 126.50 | 126.50 | 31.25 |
| 128 | 0.999328 | 0.599744 | 4062 | 7.714 | 2.279 | 253.88 | 253.88 | 125.88 |
| 256 | 0.997630 | 0.721743 | 7885 | 15.475 | 8.176 | 492.81 | 492.81 | 474.34 |

These are array byte counts, not total process memory. The shared model, anchors, extraction cache, Python metadata and temporary SVD storage are excluded. The largest coherence, not the median, enters a global mutual-coherence guarantee.

## Submitted operator, frozen shared direction

| Skills | Solver | Novel accuracy | Compound accuracy | Overall change vs32, pp [95% CI] | Support Jaccard to32 | Across seeds | Constituent recall | Coding, ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | full | 0.3770 | 0.1438 | +0.00 [+0.00, +0.00] | 1.000 | 0.600 | 0.667 | 69.6 |
| 64 | full | 0.3762 | 0.1445 | -0.03 [-0.29, +0.24] | 0.855 | 0.563 | 0.667 | 174.7 |
| 128 | full | 0.3757 | 0.1465 | +0.03 [-0.45, +0.48] | 0.807 | 0.535 | 0.617 | 392.4 |
| 256 | full | 0.3757 | 0.1472 | +0.05 [-0.48, +0.51] | 0.793 | 0.530 | 0.617 | 1029.8 |
| 32 | shortlist32 | 0.3770 | 0.1438 | +0.00 [+0.00, +0.00] | 1.000 | 0.600 | 0.667 | 68.4 |
| 64 | shortlist32 | 0.3771 | 0.1458 | +0.08 [-0.21, +0.35] | 0.853 | 0.567 | 0.667 | 263.3 |
| 128 | shortlist32 | 0.3779 | 0.1485 | +0.24 [-0.27, +0.72] | 0.781 | 0.542 | 0.633 | 258.6 |
| 256 | shortlist32 | 0.3766 | 0.1436 | -0.04 [-0.72, +0.56] | 0.750 | 0.519 | 0.633 | 270.5 |

The stricter case-aware first-answer-line scores are retained in `scale_eval_fixed_shared_analysis.json`; changing the accuracy metric changes 0 task/size/solver failure labels in this study.

### Held-out-task risk calibration

Eight predeclared tasks fit a standardized L2 logistic risk map (C=1); the remaining 17 tasks evaluate it. Failure is mean three-seed task accuracy below0.5; this differs from the paper’s oracle-normalized recovery endpoint. Compare a frozen32 mapping with a mapping refitted at each size, and with the corresponding constant training-prior predictor. If training labels have only one class, the map is that constant prior. Brier score is primary (lower is better); AUROC measures discrimination and five-bin ECE is only a small-sample diagnostic. A raw reconstruction residual is not a failure probability. This is an auxiliary supervised task-level calibration exercise, separate from the frozen LLM.

| Skills | Solver | Features | Policy | Failure rate | Brier [95% CI] | AUROC | ECE5 |
|---:|---|---|---|---:|---:|---:|---:|
| 32 | full | residual | frozen32 | 0.647 | 0.2858 [0.1215, 0.4572] | 0.333 | 0.282 |
| 32 | full | norm | frozen32 | 0.647 | 0.2916 [0.1103, 0.4742] | 0.121 | 0.228 |
| 32 | full | joint | frozen32 | 0.647 | 0.2982 [0.1285, 0.4750] | 0.227 | 0.337 |
| 32 | full | constant_prior | frozen32 | 0.647 | 0.2803 [0.1039, 0.4568] | 0.500 | 0.228 |
| 256 | full | residual | frozen32 | 0.588 | 0.3243 [0.1600, 0.4939] | 0.329 | 0.337 |
| 256 | full | residual | refit | 0.588 | 0.3252 [0.1592, 0.4962] | 0.329 | 0.338 |
| 256 | full | norm | frozen32 | 0.588 | 0.3373 [0.1554, 0.5199] | 0.114 | 0.287 |
| 256 | full | norm | refit | 0.588 | 0.3373 [0.1554, 0.5199] | 0.114 | 0.287 |
| 256 | full | joint | frozen32 | 0.588 | 0.3392 [0.1699, 0.5138] | 0.257 | 0.395 |
| 256 | full | joint | refit | 0.588 | 0.3379 [0.1674, 0.5133] | 0.257 | 0.392 |
| 256 | full | constant_prior | frozen32 | 0.588 | 0.3244 [0.1480, 0.5009] | 0.500 | 0.287 |
| 256 | full | constant_prior | refit | 0.588 | 0.3244 [0.1480, 0.5009] | 0.500 | 0.287 |
| 32 | shortlist32 | residual | frozen32 | 0.647 | 0.2858 [0.1215, 0.4572] | 0.333 | 0.282 |
| 32 | shortlist32 | norm | frozen32 | 0.647 | 0.2916 [0.1103, 0.4742] | 0.121 | 0.228 |
| 32 | shortlist32 | joint | frozen32 | 0.647 | 0.2982 [0.1285, 0.4750] | 0.227 | 0.337 |
| 32 | shortlist32 | constant_prior | frozen32 | 0.647 | 0.2803 [0.1039, 0.4568] | 0.500 | 0.228 |
| 256 | shortlist32 | residual | frozen32 | 0.647 | 0.2857 [0.1231, 0.4552] | 0.318 | 0.279 |
| 256 | shortlist32 | residual | refit | 0.647 | 0.2861 [0.1223, 0.4565] | 0.318 | 0.280 |
| 256 | shortlist32 | norm | frozen32 | 0.647 | 0.2916 [0.1103, 0.4742] | 0.121 | 0.228 |
| 256 | shortlist32 | norm | refit | 0.647 | 0.2916 [0.1103, 0.4742] | 0.121 | 0.228 |
| 256 | shortlist32 | joint | frozen32 | 0.647 | 0.2987 [0.1309, 0.4735] | 0.242 | 0.336 |
| 256 | shortlist32 | joint | refit | 0.647 | 0.2967 [0.1283, 0.4717] | 0.258 | 0.334 |
| 256 | shortlist32 | constant_prior | frozen32 | 0.647 | 0.2803 [0.1039, 0.4568] | 0.500 | 0.228 |
| 256 | shortlist32 | constant_prior | refit | 0.647 | 0.2803 [0.1039, 0.4568] | 0.500 | 0.228 |

All four sizes, task-level probabilities and paired Brier differences versus the constant-prior control are retained in `scale_eval_fixed_shared_analysis.json`. Intervals are conditional on the eight calibration tasks, and the failure prevalence can change with dictionary size; a smaller Brier value across sizes alone does not establish improved discrimination.

### Additional leave-one-task-out sensitivity check

This fixed-rule analysis was added after observing that all eight contextual calibration tasks succeeded at size32. It retains the original held-out17 result above. Each of25 tasks is evaluated by a mapping fitted to the other24 tasks; features, C=1 and the failure threshold remain unchanged. The frozen32 map also excludes the target at size32. It is post-hoc cross-fitting, not a new untouched test set. The intervals condition on cross-fitted predictions and omit refitting uncertainty from overlapping training sets.

| Skills | Solver | Features | Policy | Brier [95% CI] |
|---:|---|---|---|---:|
| 32 | full | residual | frozen32 | 0.2298 [0.1490, 0.3196] |
| 32 | full | norm | frozen32 | 0.1706 [0.1040, 0.2459] |
| 32 | full | joint | frozen32 | 0.1810 [0.1109, 0.2593] |
| 32 | full | constant_prior | frozen32 | 0.2188 [0.1424, 0.3142] |
| 256 | full | residual | frozen32 | 0.2467 [0.1643, 0.3390] |
| 256 | full | residual | refit | 0.2490 [0.1821, 0.3259] |
| 256 | full | norm | frozen32 | 0.1746 [0.1071, 0.2506] |
| 256 | full | norm | refit | 0.1713 [0.1091, 0.2406] |
| 256 | full | joint | frozen32 | 0.1827 [0.1124, 0.2610] |
| 256 | full | joint | refit | 0.1787 [0.1136, 0.2505] |
| 256 | full | constant_prior | frozen32 | 0.2354 [0.1590, 0.3285] |
| 256 | full | constant_prior | refit | 0.2361 [0.1736, 0.3142] |
| 32 | shortlist32 | residual | frozen32 | 0.2298 [0.1490, 0.3196] |
| 32 | shortlist32 | norm | frozen32 | 0.1706 [0.1040, 0.2459] |
| 32 | shortlist32 | joint | frozen32 | 0.1810 [0.1109, 0.2593] |
| 32 | shortlist32 | constant_prior | frozen32 | 0.2188 [0.1424, 0.3142] |
| 256 | shortlist32 | residual | frozen32 | 0.2295 [0.1483, 0.3200] |
| 256 | shortlist32 | residual | refit | 0.2295 [0.1486, 0.3190] |
| 256 | shortlist32 | norm | frozen32 | 0.1706 [0.1040, 0.2459] |
| 256 | shortlist32 | norm | refit | 0.1706 [0.1040, 0.2459] |
| 256 | shortlist32 | joint | frozen32 | 0.1815 [0.1111, 0.2600] |
| 256 | shortlist32 | joint | refit | 0.1814 [0.1110, 0.2599] |
| 256 | shortlist32 | constant_prior | frozen32 | 0.2188 [0.1424, 0.3142] |
| 256 | shortlist32 | constant_prior | refit | 0.2188 [0.1424, 0.3142] |

Pooled cross-fit probability AUROC is not used to claim discrimination: leaving out a positive label lowers the training prior, while leaving out a negative raises it. Consequently a leave-one-out constant-prior predictor can have pooled AUROC zero without a meaningful reversed risk signal. Common raw-score AUROCs for residual and negative norm, all paired Brier contrasts and task-level predictions are in `scale_eval_fixed_shared_crossfit_calibration.json`.

### Portability of the submitted routing thresholds

This additional check applies the existing norm<5.125 and residual>0.7 rules without retuning. Norm selects the matched saved replacement output; residual selects matched ICL4. Escalation takes precedence in the combined rule. Branch fractions average the75 task/seed cases, while accuracy first averages seeds within task. The pure ICL4 reference is0.7316 on these25 tasks. Changes in routed accuracy include changes in the fraction of cases paying for full ICL.

| Skills | Policy | Routed accuracy | ICL4 fraction | Replacement fraction | Routes changed from32 |
|---:|---|---:|---:|---:|---:|
| 32 | steering | 0.2838 | 0.000 | 0.000 | 0.000 |
| 256 | steering | 0.2843 | 0.000 | 0.000 | 0.000 |
| 32 | norm_only | 0.4255 | 0.000 | 0.440 | 0.000 |
| 256 | norm_only | 0.4279 | 0.000 | 0.440 | 0.000 |
| 32 | residual_only | 0.5475 | 0.640 | 0.000 | 0.000 |
| 256 | residual_only | 0.5604 | 0.667 | 0.000 | 0.027 |
| 32 | combined | 0.6147 | 0.640 | 0.120 | 0.000 |
| 256 | combined | 0.6161 | 0.667 | 0.107 | 0.027 |

All sizes, both solvers, task/seed decisions and paired accuracy intervals are retained in `scaling_routing_analysis.json`. This measures portability of an operational threshold and does not turn a reconstruction residual into a failure probability.

### Offline resources and coherence

| Skills | Maximum coherence | Median coherence | Basis columns | Dictionary build, s | Gram build, s | Stored bases, MiB | Solver basis copy, MiB | Gram, MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.994617 | 0.572833 | 1006 | 0.031 | 0.190 | 62.88 | 62.88 | 7.72 |
| 64 | 0.999197 | 0.562534 | 2028 | 2.035 | 0.629 | 126.75 | 126.75 | 31.38 |
| 128 | 0.999372 | 0.586101 | 4072 | 5.657 | 2.251 | 254.50 | 254.50 | 126.50 |
| 256 | 0.999483 | 0.666173 | 8163 | 13.241 | 8.553 | 510.19 | 510.19 | 508.38 |

These are array byte counts, not total process memory. The shared model, anchors, extraction cache, Python metadata and temporary SVD storage are excluded. The largest coherence, not the median, enters a global mutual-coherence guarantee.

## Contextual operator, recomputed shared direction

| Skills | Solver | Novel accuracy | Compound accuracy | Overall change vs32, pp [95% CI] | Support Jaccard to32 | Across seeds | Constituent recall | Coding, ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | full | 0.6555 | 0.4694 | +0.00 [+0.00, +0.00] | 1.000 | 0.712 | 0.883 | 87.8 |
| 64 | full | 0.6502 | 0.4851 | +0.31 [-2.15, +2.53] | 0.764 | 0.741 | 0.883 | 213.9 |
| 128 | full | 0.6473 | 0.4867 | +0.20 [-2.57, +2.82] | 0.713 | 0.722 | 0.900 | 549.2 |
| 256 | full | 0.6294 | 0.4795 | -1.16 [-3.54, +1.24] | 0.624 | 0.696 | 0.883 | 1299.9 |
| 32 | shortlist32 | 0.6555 | 0.4694 | +0.00 [+0.00, +0.00] | 1.000 | 0.712 | 0.883 | 89.0 |
| 64 | shortlist32 | 0.6506 | 0.4861 | +0.38 [-2.11, +2.64] | 0.760 | 0.736 | 0.883 | 226.2 |
| 128 | shortlist32 | 0.6477 | 0.4858 | +0.19 [-2.60, +2.81] | 0.709 | 0.727 | 0.900 | 237.6 |
| 256 | shortlist32 | 0.6289 | 0.4818 | -1.10 [-3.51, +1.36] | 0.623 | 0.691 | 0.883 | 248.6 |

The stricter case-aware first-answer-line scores are retained in `scale_context_analysis.json`; changing the accuracy metric changes 2 task/size/solver failure labels in this study.

### Held-out-task risk calibration

Eight predeclared tasks fit a standardized L2 logistic risk map (C=1); the remaining 17 tasks evaluate it. Failure is mean three-seed task accuracy below0.5; this differs from the paper’s oracle-normalized recovery endpoint. Compare a frozen32 mapping with a mapping refitted at each size, and with the corresponding constant training-prior predictor. If training labels have only one class, the map is that constant prior. Brier score is primary (lower is better); AUROC measures discrimination and five-bin ECE is only a small-sample diagnostic. A raw reconstruction residual is not a failure probability. This is an auxiliary supervised task-level calibration exercise, separate from the frozen LLM.

| Skills | Solver | Features | Policy | Failure rate | Brier [95% CI] | AUROC | ECE5 |
|---:|---|---|---|---:|---:|---:|---:|
| 32 | full | residual | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | full | norm | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | full | joint | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | full | constant_prior | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | full | residual | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | full | residual | refit | 0.471 | 0.4064 [0.2083, 0.6088] | 0.431 | 0.382 |
| 256 | full | norm | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | full | norm | refit | 0.471 | 0.3181 [0.1573, 0.4919] | 0.458 | 0.367 |
| 256 | full | joint | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | full | joint | refit | 0.471 | 0.3492 [0.1691, 0.5404] | 0.431 | 0.329 |
| 256 | full | constant_prior | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | full | constant_prior | refit | 0.471 | 0.3686 [0.1921, 0.5450] | 0.500 | 0.346 |
| 32 | shortlist32 | residual | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | shortlist32 | norm | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | shortlist32 | joint | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | shortlist32 | constant_prior | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | shortlist32 | residual | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | shortlist32 | residual | refit | 0.471 | 0.4055 [0.2079, 0.6075] | 0.431 | 0.382 |
| 256 | shortlist32 | norm | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | shortlist32 | norm | refit | 0.471 | 0.3181 [0.1573, 0.4919] | 0.458 | 0.367 |
| 256 | shortlist32 | joint | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | shortlist32 | joint | refit | 0.471 | 0.3481 [0.1682, 0.5392] | 0.431 | 0.328 |
| 256 | shortlist32 | constant_prior | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | shortlist32 | constant_prior | refit | 0.471 | 0.3686 [0.1921, 0.5450] | 0.500 | 0.346 |

All four sizes, task-level probabilities and paired Brier differences versus the constant-prior control are retained in `scale_context_analysis.json`. Intervals are conditional on the eight calibration tasks, and the failure prevalence can change with dictionary size; a smaller Brier value across sizes alone does not establish improved discrimination.

### Additional leave-one-task-out sensitivity check

This fixed-rule analysis was added after observing that all eight contextual calibration tasks succeeded at size32. It retains the original held-out17 result above. Each of25 tasks is evaluated by a mapping fitted to the other24 tasks; features, C=1 and the failure threshold remain unchanged. The frozen32 map also excludes the target at size32. It is post-hoc cross-fitting, not a new untouched test set. The intervals condition on cross-fitted predictions and omit refitting uncertainty from overlapping training sets.

| Skills | Solver | Features | Policy | Brier [95% CI] |
|---:|---|---|---|---:|
| 32 | full | residual | frozen32 | 0.2519 [0.1812, 0.3292] |
| 32 | full | norm | frozen32 | 0.1976 [0.1167, 0.2953] |
| 32 | full | joint | frozen32 | 0.2071 [0.1237, 0.3068] |
| 32 | full | constant_prior | frozen32 | 0.2361 [0.1736, 0.3142] |
| 256 | full | residual | frozen32 | 0.2613 [0.1872, 0.3395] |
| 256 | full | residual | refit | 0.2652 [0.2028, 0.3302] |
| 256 | full | norm | frozen32 | 0.3632 [0.2840, 0.4445] |
| 256 | full | norm | refit | 0.2703 [0.2032, 0.3424] |
| 256 | full | joint | frozen32 | 0.3681 [0.2902, 0.4496] |
| 256 | full | joint | refit | 0.2914 [0.2162, 0.3697] |
| 256 | full | constant_prior | frozen32 | 0.2494 [0.1736, 0.3253] |
| 256 | full | constant_prior | refit | 0.2500 [0.1892, 0.3108] |
| 32 | shortlist32 | residual | frozen32 | 0.2519 [0.1812, 0.3292] |
| 32 | shortlist32 | norm | frozen32 | 0.1976 [0.1167, 0.2953] |
| 32 | shortlist32 | joint | frozen32 | 0.2071 [0.1237, 0.3068] |
| 32 | shortlist32 | constant_prior | frozen32 | 0.2361 [0.1736, 0.3142] |
| 256 | shortlist32 | residual | frozen32 | 0.2615 [0.1873, 0.3399] |
| 256 | shortlist32 | residual | refit | 0.2654 [0.2031, 0.3304] |
| 256 | shortlist32 | norm | frozen32 | 0.3632 [0.2840, 0.4445] |
| 256 | shortlist32 | norm | refit | 0.2703 [0.2032, 0.3424] |
| 256 | shortlist32 | joint | frozen32 | 0.3683 [0.2904, 0.4499] |
| 256 | shortlist32 | joint | refit | 0.2916 [0.2163, 0.3700] |
| 256 | shortlist32 | constant_prior | frozen32 | 0.2494 [0.1736, 0.3253] |
| 256 | shortlist32 | constant_prior | refit | 0.2500 [0.1892, 0.3108] |

Pooled cross-fit probability AUROC is not used to claim discrimination: leaving out a positive label lowers the training prior, while leaving out a negative raises it. Consequently a leave-one-out constant-prior predictor can have pooled AUROC zero without a meaningful reversed risk signal. Common raw-score AUROCs for residual and negative norm, all paired Brier contrasts and task-level predictions are in `scale_context_crossfit_calibration.json`.

### Portability of the submitted routing thresholds

This additional check applies the existing norm<5.125 and residual>0.7 rules without retuning. Norm selects the matched saved replacement output; residual selects matched ICL4. Escalation takes precedence in the combined rule. Branch fractions average the75 task/seed cases, while accuracy first averages seeds within task. The pure ICL4 reference is0.7316 on these25 tasks. Changes in routed accuracy include changes in the fraction of cases paying for full ICL.

| Skills | Policy | Routed accuracy | ICL4 fraction | Replacement fraction | Routes changed from32 |
|---:|---|---:|---:|---:|---:|
| 32 | steering | 0.5810 | 0.000 | 0.000 | 0.000 |
| 256 | steering | 0.5694 | 0.000 | 0.000 | 0.000 |
| 32 | norm_only | 0.5810 | 0.000 | 0.000 | 0.000 |
| 256 | norm_only | 0.5694 | 0.000 | 0.000 | 0.000 |
| 32 | residual_only | 0.6638 | 0.440 | 0.000 | 0.000 |
| 256 | residual_only | 0.6798 | 0.360 | 0.000 | 0.160 |
| 32 | combined | 0.6638 | 0.440 | 0.000 | 0.000 |
| 256 | combined | 0.6798 | 0.360 | 0.000 | 0.160 |

All sizes, both solvers, task/seed decisions and paired accuracy intervals are retained in `scaling_routing_analysis.json`. This measures portability of an operational threshold and does not turn a reconstruction residual into a failure probability.

### Offline resources and coherence

| Skills | Maximum coherence | Median coherence | Basis columns | Dictionary build, s | Gram build, s | Stored bases, MiB | Solver basis copy, MiB | Gram, MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.995499 | 0.553645 | 988 | 1.877 | 0.185 | 61.75 | 61.75 | 7.45 |
| 64 | 0.999554 | 0.528179 | 1786 | 3.740 | 0.494 | 111.62 | 111.62 | 24.34 |
| 128 | 0.999635 | 0.584041 | 3523 | 7.570 | 1.722 | 220.19 | 220.19 | 94.69 |
| 256 | 0.999234 | 0.760776 | 6769 | 15.564 | 6.041 | 423.06 | 423.06 | 349.57 |

These are array byte counts, not total process memory. The shared model, anchors, extraction cache, Python metadata and temporary SVD storage are excluded. The largest coherence, not the median, enters a global mutual-coherence guarantee.

## Contextual operator, frozen shared direction

| Skills | Solver | Novel accuracy | Compound accuracy | Overall change vs32, pp [95% CI] | Support Jaccard to32 | Across seeds | Constituent recall | Coding, ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | full | 0.6555 | 0.4694 | +0.00 [+0.00, +0.00] | 1.000 | 0.712 | 0.883 | 86.4 |
| 64 | full | 0.6627 | 0.4753 | +0.67 [-0.24, +2.02] | 0.783 | 0.696 | 0.767 | 177.5 |
| 128 | full | 0.6703 | 0.4760 | +1.15 [-0.27, +3.49] | 0.704 | 0.676 | 0.767 | 366.7 |
| 256 | full | 0.6694 | 0.4760 | +1.10 [-0.26, +3.28] | 0.646 | 0.663 | 0.767 | 653.3 |
| 32 | shortlist32 | 0.6555 | 0.4694 | +0.00 [+0.00, +0.00] | 1.000 | 0.712 | 0.883 | 86.0 |
| 64 | shortlist32 | 0.6627 | 0.4753 | +0.67 [-0.22, +2.00] | 0.764 | 0.706 | 0.733 | 262.0 |
| 128 | shortlist32 | 0.6703 | 0.4724 | +1.01 [-0.28, +3.14] | 0.641 | 0.666 | 0.683 | 208.4 |
| 256 | shortlist32 | 0.6694 | 0.4753 | +1.07 [-0.23, +3.15] | 0.571 | 0.644 | 0.683 | 212.7 |

The stricter case-aware first-answer-line scores are retained in `scale_context_fixed_shared_analysis.json`; changing the accuracy metric changes 0 task/size/solver failure labels in this study.

### Held-out-task risk calibration

Eight predeclared tasks fit a standardized L2 logistic risk map (C=1); the remaining 17 tasks evaluate it. Failure is mean three-seed task accuracy below0.5; this differs from the paper’s oracle-normalized recovery endpoint. Compare a frozen32 mapping with a mapping refitted at each size, and with the corresponding constant training-prior predictor. If training labels have only one class, the map is that constant prior. Brier score is primary (lower is better); AUROC measures discrimination and five-bin ECE is only a small-sample diagnostic. A raw reconstruction residual is not a failure probability. This is an auxiliary supervised task-level calibration exercise, separate from the frozen LLM.

| Skills | Solver | Features | Policy | Failure rate | Brier [95% CI] | AUROC | ECE5 |
|---:|---|---|---|---:|---:|---:|---:|
| 32 | full | residual | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | full | norm | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | full | joint | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | full | constant_prior | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | full | residual | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | residual | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | norm | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | norm | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | joint | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | joint | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | constant_prior | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | full | constant_prior | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 32 | shortlist32 | residual | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | shortlist32 | norm | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | shortlist32 | joint | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 32 | shortlist32 | constant_prior | frozen32 | 0.471 | 0.4706 [0.2353, 0.7059] | 0.500 | 0.471 |
| 256 | shortlist32 | residual | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | residual | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | norm | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | norm | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | joint | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | joint | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | constant_prior | frozen32 | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |
| 256 | shortlist32 | constant_prior | refit | 0.412 | 0.4118 [0.1765, 0.6471] | 0.500 | 0.412 |

All four sizes, task-level probabilities and paired Brier differences versus the constant-prior control are retained in `scale_context_fixed_shared_analysis.json`. Intervals are conditional on the eight calibration tasks, and the failure prevalence can change with dictionary size; a smaller Brier value across sizes alone does not establish improved discrimination.

### Additional leave-one-task-out sensitivity check

This fixed-rule analysis was added after observing that all eight contextual calibration tasks succeeded at size32. It retains the original held-out17 result above. Each of25 tasks is evaluated by a mapping fitted to the other24 tasks; features, C=1 and the failure threshold remain unchanged. The frozen32 map also excludes the target at size32. It is post-hoc cross-fitting, not a new untouched test set. The intervals condition on cross-fitted predictions and omit refitting uncertainty from overlapping training sets.

| Skills | Solver | Features | Policy | Brier [95% CI] |
|---:|---|---|---|---:|
| 32 | full | residual | frozen32 | 0.2519 [0.1812, 0.3292] |
| 32 | full | norm | frozen32 | 0.1976 [0.1167, 0.2953] |
| 32 | full | joint | frozen32 | 0.2071 [0.1237, 0.3068] |
| 32 | full | constant_prior | frozen32 | 0.2361 [0.1736, 0.3142] |
| 256 | full | residual | frozen32 | 0.2346 [0.1665, 0.3092] |
| 256 | full | residual | refit | 0.2307 [0.1464, 0.3225] |
| 256 | full | norm | frozen32 | 0.2109 [0.1275, 0.3105] |
| 256 | full | norm | refit | 0.2277 [0.1389, 0.3270] |
| 256 | full | joint | frozen32 | 0.2202 [0.1346, 0.3209] |
| 256 | full | joint | refit | 0.2363 [0.1437, 0.3401] |
| 256 | full | constant_prior | frozen32 | 0.2194 [0.1559, 0.2965] |
| 256 | full | constant_prior | refit | 0.2188 [0.1424, 0.3142] |
| 32 | shortlist32 | residual | frozen32 | 0.2519 [0.1812, 0.3292] |
| 32 | shortlist32 | norm | frozen32 | 0.1976 [0.1167, 0.2953] |
| 32 | shortlist32 | joint | frozen32 | 0.2071 [0.1237, 0.3068] |
| 32 | shortlist32 | constant_prior | frozen32 | 0.2361 [0.1736, 0.3142] |
| 256 | shortlist32 | residual | frozen32 | 0.2347 [0.1661, 0.3099] |
| 256 | shortlist32 | residual | refit | 0.2317 [0.1472, 0.3235] |
| 256 | shortlist32 | norm | frozen32 | 0.2109 [0.1275, 0.3105] |
| 256 | shortlist32 | norm | refit | 0.2277 [0.1389, 0.3270] |
| 256 | shortlist32 | joint | frozen32 | 0.2199 [0.1344, 0.3209] |
| 256 | shortlist32 | joint | refit | 0.2359 [0.1436, 0.3396] |
| 256 | shortlist32 | constant_prior | frozen32 | 0.2194 [0.1559, 0.2965] |
| 256 | shortlist32 | constant_prior | refit | 0.2188 [0.1424, 0.3142] |

Pooled cross-fit probability AUROC is not used to claim discrimination: leaving out a positive label lowers the training prior, while leaving out a negative raises it. Consequently a leave-one-out constant-prior predictor can have pooled AUROC zero without a meaningful reversed risk signal. Common raw-score AUROCs for residual and negative norm, all paired Brier contrasts and task-level predictions are in `scale_context_fixed_shared_crossfit_calibration.json`.

### Portability of the submitted routing thresholds

This additional check applies the existing norm<5.125 and residual>0.7 rules without retuning. Norm selects the matched saved replacement output; residual selects matched ICL4. Escalation takes precedence in the combined rule. Branch fractions average the75 task/seed cases, while accuracy first averages seeds within task. The pure ICL4 reference is0.7316 on these25 tasks. Changes in routed accuracy include changes in the fraction of cases paying for full ICL.

| Skills | Policy | Routed accuracy | ICL4 fraction | Replacement fraction | Routes changed from32 |
|---:|---|---:|---:|---:|---:|
| 32 | steering | 0.5810 | 0.000 | 0.000 | 0.000 |
| 256 | steering | 0.5920 | 0.000 | 0.000 | 0.000 |
| 32 | norm_only | 0.5810 | 0.000 | 0.000 | 0.000 |
| 256 | norm_only | 0.5920 | 0.000 | 0.000 | 0.000 |
| 32 | residual_only | 0.6638 | 0.440 | 0.000 | 0.000 |
| 256 | residual_only | 0.7007 | 0.520 | 0.000 | 0.080 |
| 32 | combined | 0.6638 | 0.440 | 0.000 | 0.000 |
| 256 | combined | 0.7007 | 0.520 | 0.000 | 0.080 |

All sizes, both solvers, task/seed decisions and paired accuracy intervals are retained in `scaling_routing_analysis.json`. This measures portability of an operational threshold and does not turn a reconstruction residual into a failure probability.

### Offline resources and coherence

| Skills | Maximum coherence | Median coherence | Basis columns | Dictionary build, s | Gram build, s | Stored bases, MiB | Solver basis copy, MiB | Gram, MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.995499 | 0.553645 | 988 | 0.031 | 0.183 | 61.75 | 61.75 | 7.45 |
| 64 | 0.999631 | 0.478376 | 1568 | 1.874 | 0.396 | 98.00 | 98.00 | 18.76 |
| 128 | 0.999793 | 0.461619 | 2735 | 5.570 | 1.071 | 170.94 | 170.94 | 57.07 |
| 256 | 0.999850 | 0.480651 | 4229 | 13.022 | 2.527 | 264.31 | 264.31 | 136.45 |

These are array byte counts, not total process memory. The shared model, anchors, extraction cache, Python metadata and temporary SVD storage are excluded. The largest coherence, not the median, enters a global mutual-coherence guarantee.

## Source generation ledgers

```json
{
  "scale_eval": "c954cc94e3f0b4ce4186ece321f8976598890c2a357b67bba0c7e80ffa46d680",
  "scale_eval_fixed_shared": "2ca0a981074cb4d41433844531dbf90a242fa22e512cbf8f04c8c5e3bee23014",
  "scale_context": "29a75c003282de104387129657e31c5d7db8f30c1588ed93d9ecb310775b9263",
  "scale_context_fixed_shared": "c76f875160372b04c64a66359706330a261c00f09cdcaa81a3b1f8b742a946f6"
}
```
