Response to Reviewer DuwT

**1. Difference from prior work.** CASS identifies a shared-direction interference mechanism and uses the resulting subspace geometry to assemble a correction for a new task's demonstration-derived direction. The leading cross-task component is a concrete obstacle to reuse: the submitted random-direction, other-PC and removal-rank controls isolate its role. Sparse coding then selects a small set of skill subspaces from which the correction projector and anchor are assembled.

We will sharpen the comparison to the closest work. ICV extracts demonstration vectors and also studies vector arithmetic; ELICIT stores and retrieves capability vectors and explicitly evaluates unseen-task transfer; ATV generates query-conditioned task vectors with a smaller model. CASS differs in mining low-rank cross-task subspaces, removing the shared interference direction, and performing weighted sparse coding followed by a geometric correction while preserving the new task's own direction. We do not claim the first encoding/decoding framework or the first transfer to unseen tasks. Our nearest-skill and vector-arithmetic controls are matched mechanism comparisons, rather than full reproductions of these systems. [ICV](https://proceedings.mlr.press/v235/liu24bx.html), [ELICIT](https://arxiv.org/html/2410.09343v2), [ATV](https://arxiv.org/abs/2506.03426).

A fixed-setting extension control gives Novel15 0.5328/0.6555/0.6408/0.5303 at removed ranks 0/1/2/4; all three suites and both query sets are reported.

**2. Clear module interfaces.** The replacement overview makes three stage interfaces explicit:

| Stage | Input | Output and reuse |
|---|---|---|
| Offline mining | Known-task clean/corrupt activations | Shared direction, skill bases and anchors; reused across tasks |
| New-task adaptation | At most four demonstrations | Task direction z, sparse support, norm and residual; computed once per task |
| Query serving | Query plus cached task operator | Demonstration-direction update with composed correction, or the selected fallback |

The reconstruction diagnoses what the library explains; it is not the main injected direction. Norm-based replacement and residual-based escalation will be shown as distinct policies, consistent with their separate evaluations. This presentation makes the information flow and the dictionary's role explicit.

**3. Complete mathematical details.** We provide a corrected proof for the actual sqrt(r_t)-weighted group-LASSO objective. Starting from the restricted optimum, we derive the active KKT equation, bound every inactive weighted correlation, establish uniqueness using strict inactive inequalities and full restricted rank, and give the coefficient-error and beta-min conditions. The resulting conservative coherence condition is mu_B < 1/(sR+s-1), with R the penalty-weight ratio. The measured maximum coherences do not certify multi-skill recovery, so the result will be presented as conditional sparse-recovery analysis, without substituting median coherence for the maximum.

A second derivation addresses the actual hybrid update directly: its perturbation is bounded in terms of direction, anchor, projector and gate errors at a fixed hidden state. We state the smooth-score and routing-margin assumptions separately. These arguments clarify the operator's dependence on the recovered geometry; they do not assert a guarantee for discrete multi-step generation.

**4. Dictionary size and model comparisons.** The existing controlled size curve covers 5, 10, 20, 30, 40 and 46 available skills, with accuracies 0.437, 0.491, 0.540, 0.545, 0.526 and 0.519. It establishes improvement followed by saturation over that range, rather than monotonic improvement. We also completed a separately specified 32/64/128/256-entry stress test on Novel15 and Compound10. It retains the original 32 skills and adds 224 explicitly synthetic, correlated transformations. With the new contextual operator and the initial shared direction fixed, accuracy is 0.5810/0.5877/0.5926/0.5920. At 256 entries, a 32-block shortlist gives 0.5918 accuracy while reducing median online coding from 0.653 to 0.213 s. Recomputing the shared direction at each size gives 0.5694 at 256; both update regimes and their support/calibration statistics are reported. This separates library growth from changes to the original representation and does not claim hundreds of independent natural-language domains.

For cross-model experiments, each frozen model mines its own activation dictionary from the same task data and splits. No dictionary augmentation trains or changes the model's weights. The controlled comparisons are between methods within each model. Because the pretrained checkpoints have different histories, cross-model results characterize transfer of the procedure and do not isolate architecture or pretraining as a causal factor. We will make this distinction explicit in the experimental setup.
