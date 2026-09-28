# Fixed-setting shared-rank ablation of the contextual extension

Specified on 27 September 2026 at approximately 23:59 UTC, before generating
any result for this branch. This is a mechanism ablation, not a new winner
selection protocol.

Use the already frozen plain-prompt contextual configuration (pure-null
contrast, layers12/16, gain1.5, prefill, uncertainty-weighted correction).
Rebuild its dictionary and de-share the task signature with shared rank
0, 1, 2 or 4 at each selected layer. Each condition has a matched correction-off arm. Retain the
same mining activations, four demonstrations, queries, support-cap solver,
rank-energy rule, anchors, prompt and generation engine. The skill ranks and
direction-norm calibration are recomputed as part of each dictionary condition.

Evaluate the20 non-development LOTO targets, Novel15 and Compound10, using
seeds20/21/22, on both the original and additional50-query splits. Target
mining is excluded before every LOTO dictionary construction. No evaluation
label selects a rank or updates the frozen gain/schedule. Rank1 and its off
arm reuse the exact main-confirmation predictions after verifying inputs;
the reconstructed rank1 operator is additionally checked against saved
predictions on the first task/seed. ICL4 is reused under the same input checks.

Report every rank and matched off condition, both metrics, task-bootstrap
rank-vs1 and correction contrasts. This fixed-setting intervention does not
claim that each rank receives its own optimally tuned hyperparameters.
