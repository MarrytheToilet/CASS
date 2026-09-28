# Reviewer coverage ledger

Final evidence status, 2026-09-28 01:35 UTC. “Complete” describes the evidence, not a claim
that the experiment must be favorable or that acceptance is assured.

| Reviewer request | Evidence / action | Status |
|---|---|---|
| cajh: batched 48-forward latency | Original batch8/batch48 measurements; Gram solver and selective-layer extraction benchmarks; matched prefix-cache decoding | Complete, including cached and answer-delimited contracts |
| cajh: adaptation energy | Board-power integration, adaptation and serving measured separately; CPU energy excluded explicitly | Complete; CPU/system energy explicitly outside measurement |
| cajh: accuracy versus matched ICL4 | New operator, correction-off, initial 552-setting and further position-tuned dictionary-free controls, ICL1/4, three new seeds | Original and additional-query confirmation complete; both metrics and all intervals recorded |
| cajh: long-form/multi-step/code/semantic metrics | All 17 predeclared broader tasks: reading/entity token F1, 20–30-word structured outputs, execution-scored code expressions | All plain, native, position and projector comparisons complete; scope is controlled |
| cajh: Qwen failure mechanism | Native template, layer/gain/position factorial, model-specific contrast development; both Qwen versions | Both Qwen models complete on original and additional queries; both Qwen3 factorials complete; all three scoring views retained |
| cajh: hundreds of skills, support stability, calibration | 32/64/128/256-entry controlled correlated expansion; support overlap, constituent recall, Brier/AUC; recomputed and fixed shared direction | Complete: all four expansion regimes, held-out calibration, post-hoc cross-fitting, known-task risk fitting and frozen-threshold portability |
| BiuA: weighted unequal-rank Proposition 1 | Independently constructed counterexample; corrected weighted KKT proof; all496 two-block supports per model checked,14 total satisfy geometric bound;42 planted-signal checks | Complete |
| BiuA: identical routing and replacement on LOTO/Novel | Original shared-seed paired comparisons; distinguish from independently retuned z-only | Complete |
| BiuA: Compound inferred/true/off with common route | Original matched controls; new contextual operator controls with wrong-support and correction-norm matching | Complete; new valid wrong-support matching covers four tasks and is labeled accordingly |
| BiuA: representative Compound failures | Deterministic examples from the same intervention logs, matched input/target/output | Both complete, including case-aware stage diagnostics |
| Bm1S: actual maximum coherence and non-vacuity | Five full dictionaries and all160 LOTO dictionaries; support-specific weighted conditions non-vacuous for10/1/1/2/0 pairs; observed-task assumptions remain separate | Complete |
| Bm1S: what is composed | Separate task-specific direction from composed projector/anchor; revise abstract, overview and operator explanation | Draft and proposed manuscript replacements ready |
| Bm1S: causal relevance versus semantic association | Wrong supports preserve block ranks and match coherence; norm-matched variants; correction-off isolates operator gain | Both original and new controls complete |
| DuwT: novelty relative to prior work | Primary-source comparison to ICV, ELICIT and ATV; no first encoder/decoder or first unseen-task claim | Sources checked; response and manuscript text drafted |
| DuwT: module clarity | Inputs/outputs and reuse frequency for mining, task adaptation, query serving; revised algorithm explanation | Replacement PDF/SVG overview, module table and caption ready; fixed-setting rank controls complete on both query sets |
| DuwT: mathematical rigor | Full weighted support proof plus direct hybrid perturbation bound; separate exact geometry result for the new variant | Complete |
| DuwT: task-count variation | Existing 5–46 curve plus 32–256 controlled expansion; report accuracy and calibration without monotonicity claim | Complete: original curve and all four 32–256 regimes |
| DuwT: cross-model fairness | Same task data/splits; each frozen model mines its own activations; within-model paired contrasts; no inference that pretraining is controlled | Clarification ready |

Additional validation: generation-to-score recomputation, stricter first-answer
whole-line scoring, configuration/source hashes, no tuning on evaluation labels,
and a reusable-prefix ICL latency baseline. The four replies will use compact
results with links to the detailed evidence rather than recounting the search.
