# Original method and tested extension

## Submitted CASS

Offline, mine contrastive activations on known tasks, estimate and remove the
shared rank-one direction, and store a low-rank basis and differential anchor for
each task and selected layer. For a new task, extract four leave-self-out
signatures using24 clean and24 label-shuffled sequences, de-share their mean,
and solve the weighted group-LASSO path with support cap5. The demonstration
direction is calibrated by the selected anchor norms. A support projector and
weighted differential anchor define the gated, state-adaptive correction.
Weak-signal routing uses prompt-state replacement; residual-based ICL escalation
is a separate operating point. Sparse coefficients reconstruct a representation
but do not replace the main task-specific direction in the deployed operator.

## Development-selected extension, original-query confirmation complete

The no-demonstration contrast retains the same query and compares its clean
few-shot hidden state with its zero-shot hidden state. Both dictionary mining
and new-task extraction use this contrast. Four examples with six orderings each
need24 clean plus4 zero-shot sequences. Dictionary mining still uses the same
known-task labels and no target evaluation examples.

Store each known task's mean clean contextual hidden state in addition to its
subspace. For the inferred support, let mu_clean be their coefficient-weighted
mean and let d be the calibrated new-task direction. Define c as one minus the
mean pairwise cosine of the four de-shared signatures, clipped to[0,1]. The
selected update is

    h' = h + 1.5 d + c g P_S(mu_clean - h - 1.5 d),

applied only at the final prompt position during prefill. Its correction blends
the projected, already-shifted state toward a contextual reference. The out-of-
span component keeps its demonstration-derived update. This is a new tested
variant, not an unchanged implementation of the submitted equation.

Three comparisons answer different questions: disabling only the correction
isolates that term; a separately tuned raw-mean/PC1/state-interpolation baseline
removes the entire library; swapping only correction support under matched
geometry/norm tests the support choice. These results must be reported distinctly.

The full original-query confirmation contains 57 tasks, three demonstration
seeds, and eight matched arms. Primary non-development LOTO20 / Novel15 /
Compound10 accuracies are .502218 / .655495 / .469373, versus .487231 /
.629021 / .454392 with only the correction disabled. The original submitted
operator on the same new seeds obtains .313456 / .377034 / .143843.
Further query validation and the denser independent baseline remain pending.

The new Compound true-support control is .461373, versus inferred support
.469373 and correction-off .454392. Wrong real-skill supports meet the exact
rank/coherence requirements for all three seeds on only four tasks, so their
paired comparison must be explicitly scoped to that subset. In that subset,
inferred minus wrong is +.028611 [+.013889,+.043333]; matching correction norm
as well yields +.021944 [+.007222,+.036667]. Preserve the original submitted
operator's all-ten-task support controls separately; do not merge these cohorts.

## Timing

A Gram matrix is cached with each reusable dictionary, retaining the original
weighted coordinate updates and lambda path. Selected-layer decoder extraction
avoids unused vocabulary logits and all-layer outputs. The contextual correction
can be collected algebraically as h+beta-cg(hQ)Q^T with constant beta, enabling
compilation. Actual adaptation and serving timings, including prefix-cached ICL,
remain separate from any input-token accounting.
