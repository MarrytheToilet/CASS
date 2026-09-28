# Measured adaptation and serving costs

Device: NVIDIA GeForce RTX 5090. Llama-3.1-8B-Instruct, BF16, SDPA; physical GPU3. Models and dictionary caches are warm. Timings include host-side adaptation and synchronized GPU execution. Energy is sampled GPU-board energy, excluding CPU and system energy.

All three predeclared timing tasks are reported. Seven grouped measurement repeats support within-run bootstrap intervals; they do not measure between-run or device variability. Fixed eight-token-budget decoding and answer-delimited decoding are separate, predeclared contracts. The latter may finish after different numbers of tokens for the two methods.

## Original operator: exact implementation optimization

| Task | Original extraction, s | Original adaptation, s | Optimized adaptation, s | Adaptation speedup | Original GPU J | Optimized GPU J |
|---|---:|---:|---:|---:|---:|---:|
| antonym | 0.1830 | 1.5074 | 0.2758 | 5.47× | 215.82 | 83.92 |
| country-capital | 0.2163 | 0.9776 | 0.2846 | 3.44× | 179.93 | 98.85 |
| english-french | 0.2124 | 3.1049 | 0.3689 | 8.42× | 374.65 | 105.68 |

The original 48 sequences are six batch-8 model calls. Selective-layer collection and a cached Gram solver preserve the extraction batch shape, weighted objective, regularization path and operator. All 447 recorded query/batch comparisons retain identical generations (149 queries at serving batch sizes 1, 8 and 25). This is a tested numerical-equivalence result, not a proof for all inputs. Batch-48 extraction is faster but changes some BF16 generations and is excluded from the exact comparison.

## Contextual variant: reusable-prefix serving

The frozen contextual variant needs 28 extraction sequences, executed as 24 clean-context plus four zero-context sequences in two model calls. This partition preserves all 149 confirmation generations on the timing tasks. Its adaptation takes 0.196/0.165/0.411 s and approximately 50.4/50.6/76.5 GPU-board joules, respectively. Dictionary/model loading is separate. The primary results below use the uncompiled operator.

Both CASS and ICL4 use the same greedy engine and an unquantized reusable prefix cache. Cache copying and expansion are timed. CASS caches its query prefix; ICL4 caches the demonstrations. Each serving time and energy is for one batch, not one query.

### Natural decoding, at most eight tokens

| Task | Batch | CASS, ms | ICL4, ms | ICL4/CASS | CASS, J | ICL4, J | Latency saving 95% CI, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| antonym | 1 | 137.28 | 137.63 | 1.003 | 39.86 | 40.17 | [-4.95, 1.12] |
| antonym | 8 | 141.41 | 143.01 | 1.011 | 39.97 | 40.64 | [0.53, 5.21] |
| antonym | 25 | 183.44 | 183.88 | 1.002 | 50.97 | 54.61 | [-0.32, 1.58] |
| country-capital | 1 | 138.23 | 136.14 | 0.985 | 40.14 | 40.14 | [-5.04, -1.64] |
| country-capital | 8 | 181.25 | 176.51 | 0.974 | 44.93 | 45.29 | [-5.46, -2.03] |
| country-capital | 25 | 185.41 | 190.18 | 1.026 | 52.70 | 58.10 | [4.11, 7.21] |
| english-french | 1 | 139.22 | 139.80 | 1.004 | 40.56 | 40.80 | [-0.31, 2.09] |
| english-french | 8 | 176.20 | 176.98 | 1.004 | 44.54 | 45.42 | [0.20, 3.57] |
| english-french | 25 | 185.90 | 190.69 | 1.026 | 52.58 | 57.78 | [3.81, 5.87] |

### First-answer-delimited decoding, at most eight tokens

| Task | Batch | CASS, ms | ICL4, ms | ICL4/CASS | CASS, J | ICL4, J | Latency saving 95% CI, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| antonym | 1 | 43.03 | 42.05 | 0.977 | 10.47 | 10.50 | [-1.08, -0.49] |
| antonym | 8 | 62.48 | 44.12 | 0.706 | 16.13 | 11.43 | [-18.72, -17.02] |
| antonym | 25 | 123.77 | 102.19 | 0.826 | 32.67 | 30.09 | [-36.75, -20.11] |
| country-capital | 1 | 76.12 | 75.94 | 0.998 | 20.60 | 20.51 | [-0.68, -0.00] |
| country-capital | 8 | 166.55 | 161.45 | 0.969 | 40.45 | 39.55 | [-15.75, -0.00] |
| country-capital | 25 | 172.42 | 177.84 | 1.031 | 46.92 | 52.21 | [-2.82, 6.77] |
| english-french | 1 | 75.91 | 75.38 | 0.993 | 20.43 | 20.51 | [-2.20, 0.27] |
| english-french | 8 | 139.68 | 117.12 | 0.838 | 34.08 | 28.47 | [-26.21, -22.12] |
| english-french | 25 | 147.17 | 130.53 | 0.887 | 40.80 | 39.36 | [-17.27, -15.46] |

Prefix caching largely removes the repeated demonstration-prefill cost on these short prompts. At batch 25, eight-token-budget serving is near parity (1.002–1.026×), and answer-delimited serving is task dependent (0.826–1.031×). These measurements do not support an order-of-magnitude wall-clock speedup. With uncached ICL4, the contextual variant takes 0.196–0.198 s per batch versus 0.251–0.260 s, giving nominal warm-start amortization at 74–165 queries. That calculation does not apply to cached ICL.

A break-even estimate divides extra adaptation cost by per-query serving savings. When the saving is indistinguishable from zero or negative, a reliable finite break-even is not established. The machine-readable summary retains the raw point estimates and their uncertainty rather than treating large ratios as measured thresholds.

## Numerical and setup checks

Cached ICL preserves all 447 per-query scores under both submitted prefix and literal-first-line metrics in the dedicated cache check. In the matched engine checks, CASS changes one antonym answer at batch 1 and one at batch 25 (both compiled and uncompiled); the other scores are unchanged. Consequently cached-engine timing is not labeled bitwise identical to HF generation. Raw outputs and score differences are retained.

Compilation is an optional implementation variant, not required for the primary timing rows. Compilation/setup observations, prefix-cache construction, model loading, and dictionary Gram construction are recorded separately. Warm measurements are not cold-process measurements. All five timing sources and the exact-generation check are hash-linked in `latency_night_summary.json` and `timing_numerical_audit.json`.

The paper’s 10.2× figure describes amortized input tokens relative to ten-shot prompting. It is not a latency or energy measurement.

## Retained task-state tensors

Float32 operator tensor-shape accounting (Q, direction, anchor) plus unquantized BF16 reusable prefix KV. Excludes shared model weights, the global offline dictionary, Python metadata, allocator overhead and transient query KV. No claim of measured multi-task throughput or total-system memory reduction; task states are assumed individually retained.

The ratio is the median of paired task/seed ratios; it is not the ratio of the two displayed group medians. Rows are a snapshot and their task counts must be checked before citing a complete suite.

| Suite | Tasks | Task/seeds | CASS median, MiB | ICL4 median, MiB | Median paired ICL4/CASS |
|---|---:|---:|---:|---:|---:|
| classification_accuracy | 3 | 9 | 1.938 | 23.250 | 8.61× |
| code_expression_execution | 4 | 12 | 2.859 | 20.250 | 7.90× |
| code_multistep_execution | 3 | 9 | 2.438 | 23.375 | 9.59× |
| compound | 10 | 30 | 2.438 | 4.562 | 1.95× |
| entity_token_f1 | 3 | 9 | 1.938 | 13.875 | 6.52× |
| long_structured_whole_answer | 3 | 9 | 2.797 | 30.375 | 10.86× |
| novel | 15 | 45 | 2.438 | 4.625 | 1.91× |
| reading_token_f1 | 1 | 3 | 1.391 | 95.250 | 86.65× |

These sizes describe per-task retained tensors, not total VRAM, measured concurrency, or total-system throughput.
