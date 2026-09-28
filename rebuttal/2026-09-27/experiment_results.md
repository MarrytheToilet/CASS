# Completed rebuttal evidence

All fresh GPU experiments used only physical GPUs 2 and 3 on the author's
server (RTX 5090, BF16/SDPA, PyTorch 2.13.0+cu130, Transformers 4.57.1).
Original model files and remote copies were matched by SHA-256. The submitted
paper and original result CSVs were not overwritten.

## Matched reanalysis of original checkpoints

Original CASS routes and replacement outputs are frozen. Seeds 0/1/2 are averaged
within task; intervals use 50,000 paired task-bootstrap samples.

| Suite | CASS | Additive z-only | Difference in points, 95% CI |
|---|---:|---:|---:|
| LOTO-32 | 0.4575 | 0.4210 | +3.65 [+0.65, +7.27] |
| Novel-15 | 0.4952 | 0.4348 | +6.04 [-0.53, +13.47] |

The original separately calibrated Novel comparison is +0.97 [-6.23,+7.91].
Recalibrating nested LOTO routing using only the three common seeds, then sharing
that policy, gives +1.25 [-0.85,+3.67]. The positive fixed-published-policy LOTO
interval is conditional on that policy. Additive z-only retains de-sharing and
norm calibration; this is not removal of every use of the dictionary.

Sources: `matched_results.json`, `*_matched_seeds.csv`, and
`*_fixed_route_per_task.csv`.

## Fresh Compound-10 intervention

Ten tasks × three seeds × nine conditions = 270 cells. The original common route
uses steering on four tasks and replacement on six; fresh replacement outputs
are identical across all compared arms. Raw generations reproduce every cell.

| Condition | Unrouted | Same frozen route |
|---|---:|---:|
| Inferred support | 0.1477 | 0.2856 |
| True support, original operator construction | 0.1607 | 0.2976 |
| Additive z-only | 0.1311 | 0.2696 |
| Projected correction off, keep hybrid direction-weight rule | 0.1311 | 0.2696 |
| True correction geometry, keep inferred direction/weight rule | 0.1653 | 0.3023 |
| Matched wrong correction geometry | 0.1016 | 0.2445 |

Wrong supports exclude inferred and true skills, preserve support size and block
ranks, and match mean/max internal block coherence within 0.02/0.03. Of 90
alternatives, 84 pass; every task–seed pair has valid alternatives. Average valid
alternatives within seed, then seeds within task. The swaps retain z, its norm,
the original gate, and the original adaptive-weight rule evaluated at the arm's
current hidden state. They do not impose identical hidden-state trajectories.

| Paired difference, points | Unrouted 95% CI | Fixed-route 95% CI |
|---|---:|---:|
| Inferred − matched wrong | +4.61 [+0.58, +9.72] | +4.11 [0.00, +9.39] |
| Inferred − correction off | +1.66 [-0.13, +4.07] | +1.60 [-0.07, +3.87] |
| True support − inferred | +1.30 [-0.53, +4.23] | +1.20 [-0.67, +4.13] |

Support choice affects the steering path relative to deliberately wrong
alternatives. The average benefit over correction-off remains uncertain; the
experiment does not establish sequential execution of named skills. For example,
both inferred and true support produce `temporary` instead of `Temporary` for
`eternal` under antonym+capitalize, and `pouces` instead of `Pouces` for `inches`
under English–French+capitalize. The first transformation can succeed while the
second fails.

Sources: `fresh_compound_summary.json`, `fresh_compound_per_task.csv`,
`compound_controls.csv`, `compound_control_matching.json`, and
`compound_control_generations.jsonl`. These fresh results are kept separate from
the original-checkpoint Compound reanalysis in `matched_results.json`.

## Fresh latency and sampled GPU energy

Three fixed tasks; target excluded from each 31-skill dictionary; five measured
repetitions after warmup. Prompt extraction uses the original seed-0 examples.
Natural greedy generation is capped at eight tokens; no shared-prefix cache is
used. Dictionary/model loading is outside warm adaptation and reported separately.

48-sequence extraction takes 0.196–0.220 s with batch size 8 (six model calls),
0.154–0.179 s at batch size 24 (two calls), or 0.140–0.160 s in one combined batch
of 48. Coding and operator construction take 0.749–3.201 s. Complete adaptation
is measured directly, rather than summing medians of separately timed stages.

| Task | Complete adaptation (s) | CASS / ICL4, batch 1 (ms) | CASS / ICL4, batch 25 (ms) | Break-even at batch 25, queries |
|---|---:|---:|---:|---:|
| Antonym | 1.507 | 133 / 129 | 217 / 246 | 1,316 |
| Country–capital | 0.995 | 135 / 129 | 221 / 256 | 711 |
| English–French | 3.256 | 136 / 129 | 219 / 257 | 2,162 |

Break-even is adaptation time divided by measured per-query savings, rounded up.
There is no latency break-even at batch size 1. Batch size 8 is near parity;
its small measured savings imply approximately 2,498–13,051 queries and should
not be interpreted as a robust speed advantage without further measurement.

Sampled GPU-board energy is approximately 183–366 J for full adaptation and
49–54 J per 25-query CASS batch versus 75–80 J for ICL4. Sampling is every 100 ms
and includes board idle/background draw, but excludes CPU and system energy.
These readings do not establish whole-system energy savings.

Model loading took 4.36 s; reconstructing each held-out dictionary from cached
activations took 1.80–1.85 s. The shared initial activation-load/full-dictionary
setup took 2.20 s. Offline mining is separate. The original 10.2× figure is an
input-token ratio versus 10-shot prompting, not a measured latency multiplier.

Sources: `latency_results.json`, `latency_summary.json`, `benchmark_latency.py`.

## Fresh Qwen diagnosis

Eight original tasks, 399 evaluation queries per condition, one fixed diagnostic
grid, known-task oracle subspaces. All steering continues during decoding.

| Method | Task-mean accuracy |
|---|---:|
| Zero-shot | 0.0201 |
| Four-shot ICL | 0.8098 |
| Layers 14+20, last prefill token | 0.1002 |
| Layers 14+20, last four prefill tokens | 0.1279 |
| Layers 14+20, all valid prefill tokens | 0.0726 |
| Layer 14 only, last token | 0.0126 |
| Layer 20 only, last token | 0.1627 |

Placement matters, but the tested alternatives retain a large gap to ICL. This
does not identify a unique architectural cause or establish a family-wide
impossibility. Output inspection finds both format and transformation failures.
`qwen_output_diagnostics.json` reports target-word occurrence as a descriptive
check only; it is not semantic accuracy and does not replace the primary metric.

Sources: `qwen_probe.csv`, `qwen_probe_generations.jsonl`,
`qwen_probe_summary.json`, and the runtime/extraction metadata files.

## Theory and remaining scope

`theory_correction.md` gives a corrected weighted KKT proof, an independently
verified unequal-rank counterexample, and the proper smooth-score scope for
Proposition 2. `coherence_results.json` covers five full dictionaries and all
160 LOTO dictionaries: the original maximum-coherence bound certifies no
multi-skill case. `verify_theory.py` passed 50 numerical sanity checks.

No new long-form/code/tool-use benchmark or hundred-skill semantic dictionary
was evaluated. Existing scaling reaches 46 skills drawn from a 47-task pool.
The original evaluator matches the full target word sequence at the start of
the first answer line, with punctuation normalization and task-specific case
handling; trailing continuation is allowed. It is not strict whole-output match.
