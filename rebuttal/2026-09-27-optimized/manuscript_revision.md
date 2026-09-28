# Proposed manuscript text and placement

These are reviewable replacements and additions. The submitted `paper/main.tex`
has not been overwritten. Numerical additions remain tied to their own method,
seed set and evaluation policy.

## Abstract: correct the central claim

Replace the claim that tasks outside a vector library cannot be served with:

> Demonstration-derived steering vectors can serve a new task, but extracting
> each vector independently leaves reusable structure across tasks unused. We
> study how a library of known-task activations can improve the steering operator
> assembled from a new task's few-shot examples.

Replace “an unseen task's steering operator can be assembled” with:

> An unseen task's demonstrations supply its task-specific direction, while
> selected skill subspaces supply a compositional correction. Removing a shared
> rank-one component exposes cross-task geometry that makes this reuse effective.

Keep the submitted recovery and prompt-token results attached to the submitted
method. Any new contextual-operator result belongs in a clearly identified
additional experiment until the method and all corresponding tables are revised
together. “Order of magnitude” must describe the reported input-token accounting
against ten-shot prompting, not measured latency or energy.

## Introduction: contribution paragraph

> CASS makes activation libraries useful beyond storing one vector per task.
> Its first contribution is a controlled diagnosis of shared-direction
> interference: removing the leading cross-task component reveals skill-family
> structure, whereas random-direction and other-PC controls do not reproduce the
> effect. Its second contribution is a reusable subspace representation and a
> sparse coding procedure that assemble a corrective operator for a new task's
> demonstration-derived direction. Matched operator and support interventions
> measure the dictionary's contribution. Conditional recovery analysis and direct
> operator perturbation bounds specify the assumptions behind the theory, while
> extraction signals are evaluated as empirical routing diagnostics.

Remove the universal statement that every newly mined skill lowers the cost of
every future task. Use “a reusable library whose value and calibration can be
measured as it grows,” followed by the actual scaling curve and setup.

## Related work: concrete distinctions

> Demonstration compression and hidden-state injection are established mechanisms.
> ICV extracts latent directions and studies their arithmetic; ELICIT stores and
> retrieves capability vectors, including transfer to unseen tasks; ATV constructs
> query-conditioned task vectors using a smaller model. CASS studies a different
> representation and assembly rule: low-rank skill subspaces after shared-direction
> removal, weighted sparse coding of the new task signature, and a correction that
> retains the new task's own direction. These distinctions motivate the matched
> de-sharing, representation, solver, direction and support controls.

Primary sources: [ICV](https://proceedings.mlr.press/v235/liu24bx.html),
[ELICIT](https://arxiv.org/html/2410.09343v2),
[ATV](https://arxiv.org/abs/2506.03426).

Remove “retrieval by construction cannot leave the library” where it is used to
deny unseen-task behavior. It may describe which stored vector a particular
nearest-skill control selects, but it does not establish that the resulting model
cannot transfer to an unseen task.

## Method: three explicit interfaces

> **Mining, once per model and library.** Known-task clean/corrupt activation pairs
> determine a shared direction U0 at each selected layer. Projecting it out and taking per-task SVDs
> produces skill bases U_t and anchors mu_t. These are cached model-specific
> representations; no language-model weights are updated.
>
> **Adaptation, once per new task.** At most four demonstrations determine
> leave-self-out contrastive signatures. Their de-shared mean z is coded against
> the library using weighted group LASSO with a capped support path. The output
> consists of a support S, coefficient weights, a reconstruction and its residual.
> The reconstruction measures the signature component explained by the library.
>
> **Serving, once per query.** The task direction z supplies the additive update.
> The selected bases and anchors determine a projector P_S and reference mu_S for
> the gated correction. A norm-based policy may instead use prompt-state
> replacement. Residual-based escalation to full ICL is a separately measured
> policy with its own quality and cost.

Replace unqualified statements that the signals establish when the tier is
“safe” with an empirical routing claim. The residual measures reconstruction
mismatch; calibrated task-failure probabilities require a separately evaluated
mapping and a declared failure endpoint. AUROC alone establishes neither
probability calibration nor a safety guarantee.

The overview figure should show the sparse code feeding the correction, and z
feeding the main direction. The ICL fallback should visibly carry its examples;
the caption must not state that all routed queries use zero demonstration context.

A concrete replacement overview is available as
[PDF](../../optimization/2026-09-27-night/figures/submitted_method_overview.pdf),
[editable SVG](../../optimization/2026-09-27-night/figures/submitted_method_overview.svg)
and [PNG](../../optimization/2026-09-27-night/figures/submitted_method_overview.png).
Proposed caption:

> CASS separates reusable mining, new-task adaptation and query serving. Shared
> rank-one removal is applied at each selected layer. The
> new task's de-shared signature supplies the task-specific direction. Weighted
> sparse coding selects the bases and anchors used to construct its corrective
> geometry and gate; direction-norm calibration also uses these anchors. The
> reconstruction measures coverage by the library rather than replacing the
> task direction. Norm-based replacement and residual-based ICL escalation are
> separately evaluated serving policies. The diagram describes the submitted
> method; the additional contextual operator is specified separately.

## Theory: replace the proofs and their empirical interpretation

Use the full weighted KKT statement and proof in
`rebuttal/2026-09-27/theory_correction.md` for Proposition 1. State the actual
maximum block coherence, the penalty weights, the restricted-rank assumption,
the lower and upper noise-scale requirements on lambda, and beta-min when exact
support is claimed. The support-capped numerical path does not automatically
satisfy the theorem's noise condition.

For the implemented hybrid, use the direct perturbation proof in
`optimization/2026-09-27-night/theory_hybrid_stability.md` instead of treating a
reconstruction-only score bound as a deployed-system guarantee. A supplementary
note for the new contextual variant is in `theory_context_correction.md`: it
preserves the orthogonal task component and contracts the selected-span component
at a fixed intervention state. Neither statement certifies free-rollout accuracy.

Replace “verification on real dictionaries” with:

> The maximum coherence in the evaluated dictionaries falls outside the
> conservative global multi-block recovery regime. A sharper support-specific
> weighted condition admits 10/1/1/2/0 of the 496 two-block sets in the five
> dictionaries, respectively. These geometric results do not certify the noise
> assumptions or regularization choice for observed task signatures. Support
> selection and its contribution to execution are evaluated through matched
> interventions.

## Experiments: comparisons that answer the reviews

1. Put the identical-route LOTO/Novel comparison next to the independently
   calibrated system comparison, with paired intervals and calibration scope.
2. Put inferred/true/off/wrong Compound controls in one table under the same
   routing policy. Separate semantic constituent identification from the causal
   contribution of the corrective operator.
3. Identify the new contextual variant, its development split, all selected
   parameters and the independently tuned dictionary-free baseline. Report
   non-development LOTO20, Novel15 and Compound10 separately. Retain all32 as a
   secondary summary and preserve the original submitted results.
4. State the original normalized prefix metric accurately. Add task-case-aware
   whole-line results, semantic token F1 and execution-scored controlled code
   probes with their exact scope.
5. Replace a family-wide claim about Qwen with the observed dependence on prompt
   template, intervention depth, gain and position. Use the factorial result to
   distinguish these factors; do not infer a unique architectural cause.
6. Show size, support stability and calibration together. Label the correlated
   synthetic expansion, the recalibration split and the frozen-shared-direction
   alternative. A raw residual is not a calibrated probability or safety guarantee.
7. Report logical extraction sequences, batched model calls, end-to-end adaptation,
   serving batch size, warmup, prefix caching, CPU coding time and GPU-board
   energy. Keep token-count ratios separate from measured speedups.

## Cross-model setup: replace the fairness ambiguity

> Each pretrained model remains frozen and mines its own activations from the
> same task datasets and evaluation splits. We compare methods within each model
> and use the cross-model results to test transfer of the procedure. The models'
> different pretraining histories are not controlled, so these comparisons do
> not identify an architectural or pretraining cause of steerability differences.
