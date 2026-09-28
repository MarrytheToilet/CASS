# End-to-end latency and sampled GPU energy

The original-method benchmark compares original versus selective-layer extraction,
the algebraically equivalent offline-Gram solver, and compiled gated operators.
It uses three predeclared tasks (antonym, country-capital, English-French), target-
held-out dictionaries, greedy generation to eight tokens and batches1/8/25.
Model/dictionary/Gram/compilation setup are reported separately from warm online
adaptation and per-batch serving. Extraction uses 48 sequences, batch8 or48.
Predictions and activation differences are checked and recorded; BF16 batching
or compilation differences must not be silently described as bitwise identity.

A separate benchmark measures the frozen combined variant, including only the
extraction sequences its selected contrast needs (pure null:28; mixed:52;
shuffled:48), sparse coding and ready-to-use operator construction. Contextual
residual correction is algebraically collected into a constant affine term and
a low-rank matrix product; compilation setup and generation agreement are logged.
Online costs are not inferred from token ratios or from solver-only speedups.

The primary matched original-method adaptation timing keeps extraction batch8:
selected-layer activations are exactly equal to the original at that partition.
The one-batch48 measurement remains a separate throughput condition, since BF16
batch-shape differences changed some greedy outputs. Before the contextual
benchmark starts, its minimal extractor is fixed to the confirmation partitions:
one24-sequence clean batch plus one4-query zero-shot batch (28 sequences, two
model calls) for pure null. Mixed contrast additionally uses a24-sequence corrupt
batch. This preserves the confirmation activations while omitting unused work.

A further baseline reuses one four-demonstration prefix KV cache per task.
Prefix construction is charged once; copying and batch expansion are charged to
serving. Prefix token boundaries are verified against full prompt tokenization,
left-padding attention/position masks are explicit, and predictions are compared
with uncached generation at every batch size. This uses a direct greedy loop,
so the report must identify that implementation and any numerical output changes.
It is a local Transformers cache comparator, not a benchmark of a specialized
production inference engine. Cache bytes are recorded. The original and new
CASS timing results must state whether their ICL comparator uses prefix reuse.

Seven measurement repeats group calls to approximately0.8s or longer, allowing
100ms nvidia-smi power samples to bracket intervals. Energy integrates GPU board
power, including idle board draw, and excludes CPU/system energy. Report this
scope, not an unqualified whole-system energy saving. Per-task break-even uses
adaptation overhead divided by per-query serving savings, separately for each
batch size; report no break-even if the serving difference is non-positive.
# Matched cached decoding engine

In addition to the HF-generate and prefix-cached ICL measurements, run CASS and
ICL4 through the same cached greedy loop. CASS caches only the common Q: prefix;
ICL4 caches its demonstration prefix. Include cache copying and batch expansion
in both serving measurements, and record both prefix-build costs and memory.
Apply the frozen contextual operator only after the full query suffix is present.
For prefill-only steering, remove hooks before the first decode continuation.
Compare every resulting prediction with the unchanged HF-generate reference at
batch sizes1/8/25. Keep any numerical disagreement explicit. This comparison
separates the serving method from differences between generation implementations.

A separate service-contract measurement stops at the first irrevocable answer
line, using the already verified AnswerStop parser in both CASS and ICL4. The
eight-token cap, examples, tasks, batches and cached engine remain fixed. This
avoids charging continuation after the scored answer is complete. Include parser
overhead in timing and check both methods against HF-generate with identical
stopping. Report this as answer-delimited serving, alongside the original bounded
eight-token workload; do not select the contract by its measured speedup.

Task-state accounting separately compares the actual float32 Q/direction/anchor
shapes and short CASS prefix with an unquantized BF16 four-demonstration KV prefix.
Prefix boundaries use the same token-identity check as PrefixICL. Include all
Novel/Compound targets and every broader task once complete. Exclude shared
model weights, the offline dictionary and transient query KV explicitly; this is
a representation-footprint comparison, not a measured total-VRAM or concurrent-
throughput benchmark. Longer demonstrations change this ratio, so report task
groups separately rather than extrapolating the three short timing tasks.
