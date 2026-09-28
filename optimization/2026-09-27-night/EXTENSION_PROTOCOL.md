# Frozen operator extensions

Selections use the same 12-task development split and its two seeds. Combined
contrast/geometry selection uses all 120 predeclared configurations. One-example
serving selection uses the best nonzero-correction one-example configuration
among the 96 previously logged configurations. The independent dictionary-free
baseline selects the best of all 360 configurations. No evaluation labels enter
selection. Exact development file hashes are retained in extension_selected.json.

Report all three selected methods, even if a baseline is stronger. For the two
CASS variants, disable only the correction under the same signature, support
norm scaling, gain, schedule, demonstrations, and query prompts. Compare the
one-example tier with both one- and four-example ICL. This tier pays for one
example on every query; it is not zero-example serving.

Primary transfer uses the 20 known tasks outside the development task set,
with Novel15 and Compound10 as separate suites; all32 known is secondary.
Seeds20/21/22 and original evaluation queries match preceding confirmation.
A further query check uses the last50 examples of each target's dict_pool,
asserted disjoint from its original eval and fewshot pool. For known tasks,
the target is excluded from all dictionary construction and shared-component
estimation. Query-level disjointness is per target, not a claim that no input
string ever occurs in another task. The new query check uses frozen settings
and also includes original, mixed-signature, and pure-null/no-correction controls.
It does not furnish a previously unobserved benchmark-level test.

The independent dictionary-free baseline additionally receives a denser gain
check at0.75/1.5/3/4, covering the intermediate1.5 used by CASS and stronger
raw-direction scales. This adds192 mean/PC1 configurations to the original360,
using only the same development queries. Freeze the best among all552 and
confirm it on the same original and additional-query splits. The initial360-grid
baseline and its already-started confirmation are preserved, rather than silently
replaced. This is a baseline-strength check, with no evaluation-label selection.
# Additional frozen-threshold analysis

After completing both query-split confirmations, apply the submitted residual
threshold0.7 without retuning. Use identical ICL4 outputs for full and
correction-off arms on every escalated case. Retain the original norm threshold
5.125; all contextual signatures exceed it, so no fresh-query replacement output
is needed. Report branch fractions, both scoring metrics and paired task
intervals on all suites. This is a post-hoc portability analysis of an existing
threshold, separate from steering-only quality and probability calibration.
