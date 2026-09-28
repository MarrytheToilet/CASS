# Eight-hour optimization session

Start: 2026-09-27 19:18:42 UTC. Deadline: 2026-09-28 03:18:42 UTC.
Authorized physical GPUs: 2 and 3 only. Other jobs must remain untouched.

## Selection and evaluation

All original results and manuscript files are preserved. New experiments are
written here. Every attempted configuration is retained, including configurations
that do not improve performance. Rebuttal claims will distinguish exploratory
selection from evaluation with frozen settings.

The first tuning stage uses the first 12 entries of each task's existing
fewshot_pool as validation queries. Four demonstrations are sampled from the
remaining entries (or, for small pools, the split leaves at least four examples).
These queries are separate from both the dictionary extraction pool and the
original eval_queries. Final evaluations use frozen configurations on the original
evaluation split, with multiple new demonstration seeds. Prior observations of
original benchmark results are acknowledged; these are held-out-query checks,
not a claim that the benchmark itself has never been inspected.

The initial development tasks are country-capital, person-sport, antonym,
present-past, singular-plural, english-french, english-spanish, next-item,
word-length, choose-first-of-list, animal-from-list, and color-from-list.
Other known tasks and all novel/compound tasks are reserved for transfer checks
of globally selected settings. Per-task selection, if used, must use the four
provided demonstrations only and must account for its additional adaptation cost.

Primary metrics and case sensitivity match the original implementation; they
will be called normalized prefix exact match where appropriate. Any new execution
or semantic metrics must be specified before observing their results. Report
equal-task averages and paired task bootstrap intervals, with seeds averaged
within task. Fixed routing and identical replacement outputs remain necessary
for attributing improvements to dictionary correction.

## Experimental priorities

1. Exact numerical acceleration of sparse coding and end-to-end timing.
2. Llama correction strength, injection schedule, and modest subspace shrinkage.
3. Qwen layer, gain, schedule, and prompt/extraction compatibility.
4. Compound execution with controlled support interventions.
5. Larger, explicitly characterized dictionaries and broader execution probes.
6. Reviewer-by-reviewer evidence ledger and revised replies.

The original weighted-theorem correction and conditional scope remain necessary;
better empirical accuracy cannot repair an invalid mathematical condition.
