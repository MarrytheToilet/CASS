# Dictionary scaling stress test

Sizes: 32, 64, 128, 256. The original 32 natural-language skills are kept at every
size. The added 224 skills are deterministic arithmetic, modular arithmetic,
indexed list selection, and indexed character selection tasks (see manifest).
They deliberately include highly related tasks. They test computational scaling,
support stability, and calibration under correlated distractors. They do not
establish scaling across hundreds of unrelated natural-language domains.

For each skill use 100 contrastive pairs, ten demonstrations per prompt, and
the original two Llama layers. Original and generated dictionary data remain
separate from held-out target-task labels. The ordering of added skills is frozen
by a stratified shuffle before model extraction. Build the dictionary anew at
each size using the same rank-one removal and SVD settings.

Evaluate original Compound-10 and Novel-15 with three matched demonstration
seeds and identical configurations at every scale. Record exact constituent
recall for compounds, support size and stability, residual, norm, execution
accuracy, coding time, and dictionary memory. Calibration analyses must declare
the failure threshold and distinguish discrimination (AUROC) from calibrated
probabilities. Keep all scales even if accuracy is non-monotonic.

Also evaluate a predeclared computational alternative: shortlist 32 dictionary
blocks by normalized projection magnitude before running the same group-LASSO
solver. Preserve global shared-component removal and original block ordering.
This is a new scaling variant; report its online shortlist/Gram construction
cost, rather than treating that work as free offline computation.

Before seeing scaling results, reserve eight task types for calibration:
country-continent, animal-baby, verb-gerund, english-italian, word-last-letter,
antonym+capitalize, present-past+capitalize, and present-past+capitalize-first-letter.
Use the remaining 17 for calibration evaluation. Failure means task-average
accuracy below 0.5, averaging three seeds first. Fit L2 logistic mappings from
residual, norm, or both on the eight calibration tasks; compare the frozen
32-entry calibration with a separately fitted calibration at each size. Brier
score is primary and five-bin ECE is diagnostic given the small task count.

A second incremental-growth regime fixes U0 to the initial32-task estimate.
Every original basis and anchor remains exactly unchanged; new task activations
are projected against that same U0 and mined with the original tau=.9/rmax16
settings. The identical32/64/128/256 sizes, full/shortlist32 solvers, targets,
seeds and calibration protocol are repeated. This isolates representation drift
from increased support competition. It was specified before any scaling outputs
were observed. Numerical checks confirm exact retention of old bases/anchors/U0.
Both regimes report maximum and median coherence separately. Calibration includes
constant training-prior controls and paired-task Brier uncertainty; it is a
supplementary fitted risk map, not a no-label serving guarantee.
# Frozen contextual extension

After the submitted-operator scaling checks, repeat the same32/64/128/256 sizes,
full/shortlist solvers and recomputed/frozen shared-direction variants using the
already frozen contextual extension. The extra-skill extraction stores both
contrasts and positive-context means, so this needs no new task labels or mining
choices. Check every32-entry prediction against the completed extension
confirmation before using the larger-size results. Keep original and contextual
scaling files and calibration results separate. All four scaling variants are
reported; none is chosen by its final task accuracy.

## Additional calibration sensitivity analysis

Added after inspecting the completed submitted/recomputed-context analyses on
2026-09-27 at approximately23:22UTC. The eight predeclared calibration tasks all
succeed under the32-entry contextual operator, making its fitted failure map
constant. Preserve that held-out17 analysis and add a separate leave-one-task-out
cross-fit across all25 targets. For each target, fit exactly the same standardized
L2 logistic mapping(C=1) on the other24 task means, with residual/norm/joint
features and a constant-prior control. Compare a mapping frozen from size32 with
one fitted at each size, always excluding the evaluated target from either fit.
No setting, threshold or feature is chosen from these outcomes. Report Brier,
AUROC, ECE and paired task-score differences with the same fixed bootstrap seed.
Bootstrap intervals condition on the cross-fitted predictions and do not include
the variation from refitting overlapping calibration sets. This is a post-hoc
calibration sensitivity check, not a new untouched test set or a training-free
probability guarantee. Apply it to all four dictionary/operator regimes.
Pooled leave-one-out probability AUROC is diagnostic only, because the different
fold priors themselves depend on the held-out class. Report Brier as primary and
raw-score AUROC(residual and negative norm) for common uncalibrated rankings.

Also evaluate the already submitted operational thresholds without retuning:
norm below5.125 selects the saved prompt-state replacement, and residual above0.7
escalates to the saved four-shot ICL output. Record each rule separately and the
combined rule(escalation takes precedence), including branch fractions, routing
changes from32 entries and routed accuracy. Reuse only identical task/query/seed
outputs from the completed confirmation files. This post-hoc portability check
of frozen thresholds is distinct from probability calibration and from selecting
new thresholds for the contextual variant.
