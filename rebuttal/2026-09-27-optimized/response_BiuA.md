Response to Reviewer BiuA

**1. Weighted support result.** The corrected Proposition 1 includes the penalty weights in the off-support KKT condition for the implemented unequal-rank objective. For example, let U_1=[e_1,e_2,e_3,e_4], U_2=0.6e_1+0.8e_5, z=1.01e_1 and lambda=0.01. With weights 2 and 1, the restricted solution has residual 0.02e_1. The inactive block correlation is 0.012, exceeding its KKT threshold 0.01, although the original one-block unweighted coherence condition holds.

The corrected statement uses the weighted off-support correlation of the restricted Gram inverse. If its block operator norm theta_S is below one and lambda exceeds eta_S/(1-theta_S), where eta_S is the projected-noise correlation, strict KKT inequalities establish no false positives. The coefficient bound is epsilon/sigma_min(A_S) + lambda sqrt(sum_{t in S}q_t^2)/sigma_min(A_S)^2. An O(epsilon) conclusion therefore also requires lambda=O(epsilon); exact support needs beta-min. We provide the complete derivation.

The conservative coherence condition becomes mu_B < 1/(sR+s-1), with R the penalty-weight ratio. The submitted full-dictionary maxima are 0.9882–0.9946, so this global bound does not certify multi-skill recovery. The sharper support-specific weighted condition does cover some two-block sets: checking all 496 pairs gives 10 qualifying pairs for Llama-3.1, with an upper bound 0.9418 for park-country/word-length. These are geometric conditions, not certificates for the Llama Compound constituents or the observed signatures' noise assumptions. We will state this distinction and remove the median-for-maximum interpretation. We also provide a direct hybrid stability bound separating direction, anchor, projector and gate perturbations.

**2. Identical routing and replacement.** We held the submitted CASS routing assignments and every replacement output fixed across the two arms, using the three shared seeds and paired task-level bootstrap intervals:

| Suite | Full CASS | z-only | Difference, percentage points (95% CI) |
|---|---:|---:|---:|
| LOTO32 | 0.4575 | 0.4210 | +3.65 [0.65, 7.27] |
| Novel15 | 0.4952 | 0.4348 | +6.04 [-0.53, 13.47] |

The Appendix H Novel z-only value near 0.49 uses its separately calibrated policy. That system comparison and the fixed-policy intervention answer different questions; we will label them separately and add the missing intervals. The fixed-policy interval is conditional on that policy, including its original calibration.

To isolate the projected correction even more directly, our new development-selected operator keeps signature, norm scaling, gain, prompt, schedule and route fixed while disabling only the correction. On 20 non-development LOTO targets, Novel15 and Compound10, its paired gains are +1.50 [0.03, 3.03], +2.65 [0.57, 5.29] and +1.50 [0.40, 2.67] points. This separate experiment uses three new demonstration seeds; we will distinguish it from the submitted system. On 50 additional queries per task, the paired gains are +1.83 [0.57, 3.40], +2.49 [0.71, 4.67] and +0.53 [-0.27, 1.27] points. Under case-aware literal-first-line scoring they are +1.73 [0.60, 3.17], +2.58 [0.71, 4.89] and +1.40 [0.40, 2.40].

A separately position-tuned dictionary-free baseline scores 0.4379/0.6487/0.3055, and 0.4757/0.6444/0.3180 on additional queries. This system comparison is distinct from disabling the same operator’s correction.

**3. Compound execution and support.** Matched reruns of the submitted operator give:

| Common policy | Inferred support | Full true support | z-only |
|---|---:|---:|---:|
| Steering on every task | 0.1477 | 0.1607 | 0.1311 |
| Submitted route, identical replacements | 0.2856 | 0.2976 | 0.2696 |

The steering-only inferred-minus-off interval is +1.66 [-0.13, 4.07] points. Wrong-support interventions additionally show +4.61 [0.58, 9.72] points for inferred support over alternatives matched in rank and within-support coherence; the same frozen route yields +4.11 [0.00, 9.39].

The new contextual operator raises Compound execution to 0.4694; correction-off and true-support variants are 0.4544 and 0.4614. These results locate the larger improvement in extraction/intervention design, while the matched ablation identifies the additional corrective contribution. Representative failures distinguish two bottlenecks: the submitted operator emits “temporary” instead of “Temporary” for eternal → antonym+capitalize; the new operator can still emit “Overcome” instead of “Overcame” for overcome → present-past+capitalize, including with true support. On matched new seeds, first-stage-only literal outputs fall from 27.1% to 0.8%, while complete literal answers rise from 13.9% to 42.3%. These descriptive categories do not identify an internal execution sequence. We interpret recovered support as useful corrective geometry and a semantic diagnostic, with sequential functional composition remaining a distinct claim.
