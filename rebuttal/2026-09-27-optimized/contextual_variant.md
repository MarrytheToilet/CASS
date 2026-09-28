# Additional experiment: contextual correction

This is a separately developed variant of CASS. The submitted method and its
tables remain unchanged. The change concerns the activation contrast and the
form of the corrective intervention; language-model weights remain frozen.

## Extraction and dictionary

For each of four demonstrations, form six leave-self-out prompts by resampling
the order of the other three demonstrations. Subtract the hidden state for the
same query without demonstrations from the mean positive-context state. Stack
the two selected layers and remove the dictionary's shared component to obtain
four signatures, $z_1,\ldots,z_4$, and their mean $z$.

The offline dictionary uses the corresponding clean-context versus zero-context
contrast on the known-task mining examples. Shared-direction removal, per-skill
low-rank bases and weighted sparse coding follow the existing procedure. Cache
both the de-shared contrast anchor $\mu_t^\Delta$ and the positive-context hidden
mean $\mu_t^+$ for each skill. The latter is a reference state, not an additive
task vector. For LOTO, exclude the target task before estimating the shared
direction or constructing any dictionary block.

The final pure-null contrast requires 24 positive-context sequences plus four
zero-context sequences. The measured implementation executes them as two batches
of 24 and four; this preserves the confirmation run's batch shape. Intermediate
mixture experiments also require corrupted prompts and have different costs.

## Frozen intervention

Let $S$ be the recovered support and
$w_t=\|c_t\|_2/\sum_{j\in S}\|c_j\|_2$. Define

$$
d=\frac{\sum_{t\in S}w_t\|\mu_t^\Delta\|_2}{\|z\|_2}\,z,
\qquad
\mu^\Delta=\sum_{t\in S}w_t\mu_t^\Delta,
\qquad
\mu_l^+=\sum_{t\in S}w_t\mu_{t,l}^+.
$$

Here $P_{S,l}$ is the orthogonal projector onto the union of the selected
layer-$l$ bases. The two scalar gates are

$$
g=\max(0,\cos(z,\mu^\Delta)),\qquad
a=\operatorname{clip}_{[0,1]}
\left(1-\frac{1}{6}\sum_{i<j}\cos(z_i,z_j)\right).
$$

Small denominator stabilizers are retained in the implementation. With no
selected support, use $d=z$ and zero correction. For nonempty support, apply

$$
F_l(h)=h+\gamma d_l+agP_{S,l}(\mu_l^+-h-\gamma d_l).
$$

The development-selected configuration is layers 12 and 16, $\gamma=1.5$, and
intervention only at the final prompt position during prefill. The same four
examples determine the signature and gates for every query of the task.

At a fixed intervention state, this update leaves the component of
$h+\gamma d_l$ orthogonal to the selected span unchanged and moves its selected
component toward the contextual anchor. The direct contraction and perturbation
proof is in `optimization/2026-09-27-night/theory_context_correction.md`. This
geometric statement does not assert that the model internally executes the named
skills in sequence.

## Comparisons and selection

The correction-off arm sets $a=0$ while retaining the signature, dictionary-based
direction-norm calibration, layers, gain, schedule, examples and queries. It
therefore isolates the extra projected correction; it is not a dictionary-free
baseline. A separate baseline independently selects raw-mean, principal-direction
or state-interpolation updates across 552 development configurations, including
its own extraction contrast, layers, gain and schedule.

Development uses 12 specified original-task targets, 12 held-out examples per
target, and demonstration seeds 10/11. The contextual search has 120 settings.
Settings are frozen before confirmation with seeds 20/21/22. The primary known
target comparison excludes those 12 development targets; all32 is secondary.
The additional query validation uses 50 target examples disjoint from the
target's development/demonstration and original evaluation inputs. Development
dictionaries can contain activations of other known skills, so this protocol is
not described as complete research-process isolation of all known targets.

`extension_selected.json` records the selected settings, every development mean
and the source hashes. `extension_comparisons.json` links the original-query and
additional-query predictions to the paired task-bootstrap comparisons. The
original prefix metric and stricter case-aware literal-first-line metric are
reported separately. No query outcome chooses a per-task configuration.

## Other evaluated tiers

The one-demonstration tier is separately selected: it uses the original shuffled
contrast, a contextual-residual correction with strength one, gain 1.5, and
intervention during prefill and every decoding step. It retains one demonstration in every
query prompt and is compared with one-shot ICL as well as four-shot ICL. Its
costs and results are not merged with the zero-demonstration contextual variant.

Per-task demonstration cross-validation, native templates, broader token-position
interventions and exact permutation averaging are separate follow-ups. Their
configuration searches, additional adaptation costs and confirmation results
remain separately labeled in the evidence records.
