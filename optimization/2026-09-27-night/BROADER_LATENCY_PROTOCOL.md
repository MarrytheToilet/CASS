# Cached serving on the seven natural-language probes

Specified before this timing study, after the broader quality evaluations.
Include all seven predeclared natural-language tasks: news-topic, sentiment,
commonsense-choice, reading-comprehension, entity-person, entity-organization,
and entity-location. No task is selected by observed speed. Transfer the frozen
plain-prompt contextual operator, with the same seed20 demonstrations.

Use a single uncompiled greedy prefix-cache engine for both CASS and ICL4.
CASS caches the shared Q: prefix; ICL4 caches its four demonstrations. Include
cache copying and batch expansion. Use the already validated first-answer stop
rule and the32-token ceiling used in the broader evaluation. Check cached
outputs against saved outputs for all100 queries at batch4, and against fresh
uncached generation on the first32 queries at batch1/8. Record every raw output
and score change. These changes cannot be hidden behind a timing comparison.

At each batch size1/4/8, time processing the first32 queries as one fixed workload,
with seven repeated measurements after warmup. Report elapsed workload time,
queries per second, actual output lengths and GPU-board joules. This covers
variable query lengths; it is not a distribution of individual request latency.
Measure prefix construction separately. Measure warm adaptation using24 positive
sequences in six batch4 calls plus four zero-context sequences in one call, then
the same coding/operator construction; model and library construction are
separate. Compare reextracted signatures with the saved cache and report drift.

GPU-board energy uses the same100ms sampling; it excludes CPU/system energy.
No compilation, quantization, model training, evaluation-label tuning or new
steering setting is introduced. These results supplement the short-task cost
study; they do not overwrite either cached/uncached or decoding-contract result.
