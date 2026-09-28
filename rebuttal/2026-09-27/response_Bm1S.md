Response to Reviewer Bm1S

**1. What CASS composes.** CASS constructs a task-specific corrective operator from a sparse set of previously mined subspaces. The demonstrations provide the new task's direction z; the selected skills provide the projector P_S and anchor mu_S in the correction g alpha(h) P_S(mu_S-h). This is the intended division of labor. Reconstruction-only steering measures how much of the new mapping the dictionary can supply without its task-specific remainder. Its low accuracy motivates retaining z. We will make this operator-level claim explicit in the abstract and introduction and distinguish it from synthesizing an entire unseen task direction from known skills.

We also compared the hybrid operator with additive z-only under fixed routing, retaining de-sharing and direction-norm calibration in both arms. Using the three shared seeds and exactly the original CASS routing assignments and replacement outputs, CASS scores 0.4575 versus 0.4210 on LOTO: +3.65 percentage points, paired task-bootstrap 95% CI [+0.65, +7.27]. This includes adaptive direction weighting and the projected correction. The interval is conditional on the submitted policy. Recalibrating that policy on only the three common seeds yields +1.25 [-0.85, +3.67], so statistical resolution depends on policy calibration. The rank-one interference diagnosis and the measured conditional operator gain support the revised compositional claim.

**2. Actual maximum coherence and the theorem's scope.** We recomputed the joint block coherence of all five full dictionaries and all 160 leave-one-task-out dictionaries. Because the multi-layer bases are block diagonal, a pair's joint coherence is the maximum across layers.

| Model | Full maximum | Range across 32 LOTO dictionaries |
|---|---:|---:|
| Llama-3.1-8B | 0.9946 | 0.9895–0.9950 |
| Llama-3.2-3B | 0.9921 | 0.9834–0.9932 |
| Gemma-2-2B | 0.9882 | 0.9871–0.9897 |
| Qwen3-4B | 0.9937 | 0.9876–0.9943 |
| Qwen2.5-3B | 0.9935 | 0.9920–0.9940 |

Your interpretation is correct: the original condition permits only k_0=1 here. Median pairwise coherence cannot replace its maximum. Moreover, unequal block ranks require incorporating the sqrt(r_t) penalty weights. We will replace Proposition 1 with an explicit weighted KKT statement and proof. A conservative coherence corollary requires mu_B < 1/(sR+s-1), where R is the maximum-to-minimum penalty-weight ratio; equal weights recover 1/(2s-1). This does not certify multi-skill recovery in the measured dictionaries. We will remove the claimed empirical certification and retain the conditional result as a characterization of when recovery is guaranteed. Failure of this sufficient condition does not establish that recovery is impossible.

**3. Intervention on the recovered support.** We ran the requested wrong-support control on Compound-10 with three demonstration seeds. The intervention retains the inferred operator's direction, norm, gate, and adaptive-weight rule, and changes only the correction projector/anchor. Alternatives exclude both inferred and true skills, preserve support size and block ranks, and match mean/maximum within-support coherence to tolerances 0.02/0.03. Of 90 alternatives, 84 pass; every task–seed pair has valid controls, which are averaged before the paired task bootstrap.

Before routing, inferred support scores 0.1477 versus 0.1016 for matched wrong supports: +4.61 points, 95% CI [+0.58, +9.72]. Under the same frozen routing and identical replacement outputs, the difference is +4.11 [0.00, +9.39]. This establishes sensitivity to support choice in the steering path. Against removing only the projected correction, the unrouted gain is smaller, +1.66 [-0.13, +4.07]. True-support steering reaches 0.1607, with true minus inferred +1.30 [-0.53, +4.23]. Correcting the constituent labels therefore does not resolve the execution bottleneck.

For example, both inferred and true support return “temporary” rather than “Temporary” on eternal → antonym+capitalize. The experiment supports a role for selected geometry in the corrective operator; it does not establish sequential execution of the named skills. We will distinguish these two claims explicitly.
