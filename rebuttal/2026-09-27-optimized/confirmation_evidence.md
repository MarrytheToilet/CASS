# Frozen-configuration confirmation

Both query splits use the same frozen development-selected settings and four demonstrations drawn with seeds 20/21/22. The additional split contains 50 target queries disjoint from that target’s original evaluation and demonstration/development inputs. Target-task mining is excluded before building each LOTO dictionary. Development can use activations of other known skills, and the original benchmark had already been inspected earlier in the research process. These are explicitly non-development targets and disjoint queries, not a claim of a wholly unseen research benchmark.

Every row below is steering-only with no routing. The correction-off arm retains dictionary-based direction scaling. The separate dictionary-free baseline is selected from 552 development settings. The one-demo tier retains one demonstration in its query prompt and has a different serving cost. The LOTO20 split is primary; LOTO32 summaries remain available in the source JSON.

## Original evaluation queries: Submitted prefix metric

| Suite | Submitted operator | Contextual | Correction off | Dictionary-free | One-demo CASS | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3135 | 0.5022 | 0.4872 | 0.4367 | 0.5323 | 0.4787 | 0.6456 |
| Novel15 | 0.3770 | 0.6555 | 0.6290 | 0.5904 | 0.7304 | 0.6157 | 0.8480 |
| Compound10 | 0.1438 | 0.4694 | 0.4544 | 0.2985 | 0.4738 | 0.3049 | 0.5570 |

| Suite | Contextual minus submitted, pp (95% CI) | Correction gain, pp (95% CI) |
|---|---:|---:|
| Non-development LOTO20 | +18.88 [8.03, 30.50] | +1.50 [0.03, 3.03] |
| Novel15 | +27.85 [8.70, 44.41] | +2.65 [0.57, 5.29] |
| Compound10 | +32.55 [15.57, 52.53] | +1.50 [0.40, 2.67] |

## Original evaluation queries: Case-aware literal-first-line metric

| Suite | Submitted operator | Contextual | Correction off | Dictionary-free | One-demo CASS | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.2651 | 0.4727 | 0.4560 | 0.3822 | 0.5085 | 0.4663 | 0.6347 |
| Novel15 | 0.3721 | 0.6448 | 0.6184 | 0.5837 | 0.7304 | 0.6157 | 0.8467 |
| Compound10 | 0.1392 | 0.4227 | 0.3991 | 0.2898 | 0.4738 | 0.3049 | 0.5570 |

| Suite | Contextual minus submitted, pp (95% CI) | Correction gain, pp (95% CI) |
|---|---:|---:|
| Non-development LOTO20 | +20.76 [7.70, 35.43] | +1.67 [0.33, 3.07] |
| Novel15 | +27.27 [8.40, 43.48] | +2.65 [0.57, 5.28] |
| Compound10 | +28.35 [7.84, 50.09] | +2.36 [0.96, 3.90] |

## Additional 50 queries per task: Submitted prefix metric

| Suite | Submitted operator | Contextual | Correction off | Dictionary-free | One-demo CASS | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3213 | 0.5050 | 0.4867 | 0.4703 | 0.5470 | 0.4850 | 0.6357 |
| Novel15 | 0.3769 | 0.6458 | 0.6209 | 0.5844 | 0.7120 | 0.6004 | 0.8391 |
| Compound10 | 0.1400 | 0.4640 | 0.4587 | 0.3047 | 0.4567 | 0.2887 | 0.5653 |

| Suite | Contextual minus submitted, pp (95% CI) | Correction gain, pp (95% CI) |
|---|---:|---:|
| Non-development LOTO20 | +18.37 [6.63, 30.37] | +1.83 [0.57, 3.40] |
| Novel15 | +26.89 [9.29, 42.89] | +2.49 [0.71, 4.67] |
| Compound10 | +32.40 [15.13, 52.40] | +0.53 [-0.27, 1.27] |

## Additional 50 queries per task: Case-aware literal-first-line metric

| Suite | Submitted operator | Contextual | Correction off | Dictionary-free | One-demo CASS | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.2763 | 0.4790 | 0.4617 | 0.4110 | 0.5280 | 0.4750 | 0.6260 |
| Novel15 | 0.3707 | 0.6382 | 0.6124 | 0.5778 | 0.7116 | 0.5996 | 0.8391 |
| Compound10 | 0.1293 | 0.4120 | 0.3980 | 0.2873 | 0.4567 | 0.2887 | 0.5653 |

| Suite | Contextual minus submitted, pp (95% CI) | Correction gain, pp (95% CI) |
|---|---:|---:|
| Non-development LOTO20 | +20.27 [6.63, 35.30] | +1.73 [0.60, 3.17] |
| Novel15 | +26.76 [9.20, 42.76] | +2.58 [0.71, 4.89] |
| Compound10 | +28.27 [7.53, 49.87] | +1.40 [0.40, 2.40] |

## Portability of the submitted routing threshold

This post-hoc check applies residual>0.7 unchanged after both query splits are complete. All contextual signature norms exceed the submitted5.125 threshold, so the replacement branch is unused. The full and correction-off arms use identical ICL4 outputs on every escalated case. Accuracy gains over steering-only therefore include payment for four-example prompting on the stated fraction of task/seed cases. This is an operational policy check, separate from a calibrated failure-probability claim.

### Original queries / Submitted prefix

| Suite | Routed contextual | Same route, correction off | ICL4 | ICL4 fraction | Routed correction gain, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5083 | 0.4919 | 0.6456 | 0.150 | +1.63 [+0.17, +3.10] |
| Novel15 | 0.7881 | 0.7693 | 0.8480 | 0.600 | +1.88 [+0.13, +4.28] |
| Compound10 | 0.4774 | 0.4631 | 0.5570 | 0.200 | +1.43 [+0.33, +2.60] |

### Original queries / Literal first line

| Suite | Routed contextual | Same route, correction off | ICL4 | ICL4 fraction | Routed correction gain, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4814 | 0.4627 | 0.6347 | 0.150 | +1.87 [+0.60, +3.20] |
| Novel15 | 0.7797 | 0.7609 | 0.8467 | 0.600 | +1.88 [+0.13, +4.28] |
| Compound10 | 0.4307 | 0.4077 | 0.5570 | 0.200 | +2.30 [+0.90, +3.87] |

### Additional queries / Submitted prefix

| Suite | Routed contextual | Same route, correction off | ICL4 | ICL4 fraction | Routed correction gain, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5070 | 0.4897 | 0.6357 | 0.150 | +1.73 [+0.47, +3.33] |
| Novel15 | 0.7747 | 0.7591 | 0.8391 | 0.600 | +1.56 [+0.22, +3.33] |
| Compound10 | 0.4700 | 0.4680 | 0.5653 | 0.200 | +0.20 [-0.47, +0.87] |

### Additional queries / Literal first line

| Suite | Routed contextual | Same route, correction off | ICL4 | ICL4 fraction | Routed correction gain, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4850 | 0.4677 | 0.6260 | 0.150 | +1.73 [+0.60, +3.17] |
| Novel15 | 0.7698 | 0.7533 | 0.8391 | 0.600 | +1.64 [+0.22, +3.60] |
| Compound10 | 0.4193 | 0.4087 | 0.5653 | 0.200 | +1.07 [+0.07, +2.13] |

All task/seed decisions and paired comparisons are retained in `extension_routing_analysis.json` with source hashes.

## Interpretation and provenance

The dictionary-free rows above belong to the initial 552-configuration search. A further development-only prompt-position sweep strengthens that control: original-query scores become0.4379/0.6487/0.3055 on LOTO20/Novel15/Compound10, and additional-query scores become 0.4757/0.6444/0.3180. The contextual development winner stays unchanged. Complete matched comparisons and intervals for this stronger control are in `followup_evidence.md`; the figures with the `_position_control` suffix show it explicitly. The Novel difference from a separately tuned dictionary-free system is much smaller than the matched correction-off intervention, which asks a different question.

The overall extraction/intervention improvement over the submitted operator persists on the additional inputs in all three suites. The smaller projected-correction gain has positive intervals on LOTO and Novel under both metrics. For Compound on additional queries, its submitted-prefix interval includes zero, while the stricter literal-first-line interval is positive. Both are retained; significance from one metric is not substituted for the other.

The submitted metric matches the target at the beginning of the first answer line after word-edge punctuation normalization. The literal metric requires the entire whitespace-normalized first line to equal the target, preserving punctuation and signs and preserving case for case-dependent tasks. Neither metric requires all subsequent explanatory lines to be empty. Supplementary metric checks did not select settings.

Intervals use 50,000 paired task-bootstrap draws, averaging the three demonstration seeds before resampling tasks. They describe each stated suite, not simultaneous family-wise coverage across all exploratory comparisons. Source hashes, saved predictions and all paired baseline comparisons are in `extension_comparisons.json`; input-split and generation-to-score checks are in `prediction_ledger_validation.json`.
