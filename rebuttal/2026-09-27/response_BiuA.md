Response to Reviewer BiuA

**1. Dictionary contribution under the same routing policy.** We recomputed the comparison using the three seeds shared by all arms (0, 1, 2), froze the original CASS routing assignments, and used identical replacement outputs in both arms. We average seeds within each task and bootstrap paired task differences (50,000 resamples).

| Suite | CASS | z-only, same routing | Difference (percentage points), 95% CI |
|---|---:|---:|---:|
| LOTO-32 | 0.4575 | 0.4210 | +3.65 [+0.65, +7.27] |
| Novel-15 | 0.4952 | 0.4348 | +6.04 [-0.53, +13.47] |

These intervals are conditional on the frozen policy. They compare the gated hybrid operator with additive z-only, including the hybrid's adaptive direction weight and projected correction; both retain de-sharing and direction-norm calibration. On Novel, separately calibrated policies produce +0.97 points with CI [-6.23, +7.91]. The larger same-policy difference arises because those separately calibrated policies send different tasks to replacement. The 15-task evidence does not establish a statistically resolved Novel gain. As a policy sensitivity check, reselecting the nested LOTO thresholds using only the three common seeds, then sharing the assignments, yields +1.25 points [-0.85, +3.67]. We will distinguish fixed-policy attribution from policy calibration in the revised comparison.

**2. Proposition 1.** The weighted unequal-rank objective needs a weight-aware condition. Let A_S concatenate the true support blocks, G=A_S^T A_S, q_t=sqrt(r_t), and W_S=diag(q_t I). Assuming G is invertible, define

\[
\theta_S=\max_{j\notin S}\frac{1}{q_j}
\sup_{\max_t\|v_t\|_2\leq1}
\|U_j^T A_S G^{-1}W_Sv\|_2.
\]

With P=A_S G^{-1}A_S^T and eta=max_{j outside S} ||U_j^T(I-P)e||/q_j, the restricted optimum satisfies the full inactive-block KKT conditions when theta_S<1 and lambda>eta/(1-theta_S). Its coefficient error is at most epsilon/sigma_min(A_S) + lambda sqrt(sum_{t in S}q_t^2)/sigma_min(A_S)^2. An O(epsilon) conclusion also requires lambda=O(epsilon). We will give this argument explicitly and account for rank weights, as in the [group-selection literature](https://www.jmlr.org/papers/v9/bach08b.html).

The original maximum-coherence condition certifies no multi-skill setting in our dictionaries. The full Llama joint maximum is 0.9946; its 32 LOTO maxima range from 0.9895 to 0.9950. The other four models also permit only one block under that original inequality. We will remove the median-as-maximum interpretation and the claim that this condition empirically certifies multi-skill recovery. The corrected result is a conditional support-recovery statement; execution and routing remain empirical questions.

**3. Compound execution with a shared policy.** We ran a fresh three-seed comparison on RTX 5090, using the original checkpoint and data, frozen CASS routing assignments, and identical fresh replacement outputs. Inferred support, true support, and z-only score 0.2856, 0.2976, and 0.2696. Inferred minus z-only is +1.60 points [-0.07, +3.87]; true minus inferred is +1.20 [-0.67, +4.13]. These fresh-run comparisons are separate from the checkpoint reanalysis above.

Representative failures clarify the execution limit: for antonym+capitalize, both inferred and true support produce “temporary” for “eternal,” where the target is “Temporary”; for English–French+capitalize, both produce “pouces” for “inches,” where “Pouces” is required. Recovering the constituents does not ensure that both transformations execute. An additional intervention fixes the direction, gate, and adaptive-weight rule and swaps only correction geometry. Inferred support exceeds rank/coherence-matched wrong supports by +4.61 points [+0.58, +9.72] before routing; its gain over removing the projected correction is +1.66 [-0.13, +4.07]. Thus support choice affects execution relative to wrong alternatives, while the overall compound gain over the simpler operator remains uncertain.
