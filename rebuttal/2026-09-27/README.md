# CASS rebuttal package

Prepared from the four reviews supplied on 2026-09-27. Responses will be separate
for cajh, BiuA, Bm1S, and DuwT. The response deadline and length limit have not yet
been supplied. No review response has been posted and paper/main.tex is unchanged.

## Response priorities

| Reviewer | Main decision-relevant issue | Evidence or action |
|---|---|---|
| BiuA | Dictionary gain under identical routing; weighted support theorem; compound controls | Matched seeds and routing reanalysis; corrected KKT proof; intervention experiment |
| Bm1S | What is composed; maximum-coherence condition; causal role of support | State the compositional object precisely; report all dictionary maxima; swap only the correction geometry |
| cajh | Batched end-to-end cost against 4-shot ICL; scope; Qwen; hundreds of skills | Measure adaptation and serving separately; qualify token versus latency/energy claims; use existing scope evidence accurately |
| DuwT | Novelty and module clarity; proof; cross-model fairness; scale | Distinguish diagnostic/method contributions from standard sparse coding; same task data and within-model comparisons; correct theory |

The central response should explain how a reusable skill dictionary contributes
to the **corrective operator** applied alongside a new task's demonstration
direction. Pure dictionary reconstruction and constituent identification do not
establish that an unseen mapping has been synthesized, or that generation executes
the named skills sequentially. Preserve the rank-one interference finding and
the measured correction gains as the strongest evidence.

## Completed CPU evidence

`analyze_evidence.py` reads the existing checkpoints without overwriting them.
`matched_results.json` includes source CSV SHA-256 hashes. Per-task and per-seed
CSV files permit inspection. Common seeds are 0, 1, 2; seeds are averaged within
task, followed by 50,000 paired task-bootstrap samples (seed 20260927).

Primary comparison freezes the **original CASS routing assignments**, including
its nested LOTO thresholds. Both arms then use the exact same replacement score
on each replacement-routed task. Intervals are conditional on this fixed policy.

| Suite | CASS | z-only, same route | Difference, percentage points | 95% interval |
|---|---:|---:|---:|---:|
| LOTO-32 | 0.4575 | 0.4210 | +3.65 | [+0.65, +7.27] |
| Novel-15 | 0.4952 | 0.4348 | +6.04 | [-0.53, +13.47] |
| Compound-10 | 0.2846 | 0.2753 | +0.93 | [-0.80, +3.40] |

On Compound, the existing true-support arm under the same policy is 0.2986;
true minus inferred is +1.40 points, interval [-0.40, +4.60]. The common-seed
unrouted values are inferred 0.1451, true 0.1607, and z-only 0.1351. Their
differences should not be replaced with a claim of exactly identical behavior.

Important interpretation details:

- Original headline CASS results average five seeds; several baselines average
  three. These matched comparisons use only the three shared seeds, so values
  differ slightly from the submitted tables.
- The original z-only implementation retains U0 removal and support-dependent
  norm scaling. Switching to the hybrid changes both the additive direction's
  adaptive weight and its projected correction. This does not isolate only the
  latter term or remove every use of the dictionary. The fresh `no_correction`
  control retains the hybrid's direction-weight rule to separate these effects.
- For the original **separately calibrated** policies, the matched Novel
  difference is +0.97 points, interval [-6.23, +7.91]. This answers the missing
  interval question, but is a different estimand from fixing CASS's assignments.
- Sensitivity: reselecting CASS's nested LOTO thresholds using only the three
  common seeds, then sharing those assignments, gives 0.5009 vs 0.4883, a
  +1.25-point difference with interval [-0.85, +3.67]. Thus the positive
  fixed-published-policy LOTO interval is not robust to every policy calibration.
  Keep this sensitivity available and do not claim policy-independent significance.
- Existing true-support weights are per-block projections of z, not a joint
  least-squares refit over correlated blocks (the old code comment is imprecise).

`coherence_results.json` recomputes the actual block-diagonal joint coherence for
five full dictionaries and all 160 LOTO dictionaries. The joint norm is the
maximum across layers, not their average.

| Model | Full joint maximum | Range over its 32 LOTO dictionaries |
|---|---:|---:|
| Llama-3.1-8B | 0.994555 | 0.989519–0.995023 |
| Llama-3.2-3B | 0.992069 | 0.983429–0.993179 |
| Gemma-2-2B | 0.988240 | 0.987125–0.989653 |
| Qwen3-4B | 0.993710 | 0.987560–0.994275 |
| Qwen2.5-3B | 0.993471 | 0.991989–0.994011 |

The original equal-weight coherence inequality allows only s=1 in all these
dictionaries. Unequal-rank weighting must also be accounted for; the conservative
support-independent corrected bound does not even certify s=1 for some
unequal-rank dictionaries. This is a limitation of that sufficient certificate,
not a claim of empirical impossibility. `theory_correction.md` gives the full
corrected deterministic statement and proof, plus a verified independent
counterexample and the narrower scope appropriate for Proposition 2.

## GPU work

All three remote experiment suites are complete and their outputs have been
retrieved and validated. GPUs 2 and 3 have been released. See
`experiment_results.md` for the compact results and `response_*.md` for the four
separate English replies.

The author authorized **only remote physical GPUs 2 and 3** for follow-on
experiments. They are RTX 5090 cards. All subsequent GPU scripts must follow this
constraint. Remote workspace:

    /root/autodl-tmp/CASS-rebuttal-20260927

`run_remote_pair.sh` uses GPU 2 for Llama re-extraction then latency, and GPU 3
for Qwen re-extraction/probing followed by Llama correction controls.
`prepare_remote_bundle.py --code-only` was transferred; dense activations are
re-extracted on the server using the original protocol. Existing local checkpoint
reanalysis and fresh remote experiments remain separate. No credentials are
written to this directory. The bundle excludes .env, git history, and unrelated
projects.

The remote environment is project-local `.venv` (Transformers 4.57.1, inheriting
PyTorch 2.13.0+cu130); project model files reside in `models/`. The shared original
Llama directory had only two of four shards and Qwen was absent. Qwen uses the
fixed original revision through the HF mirror recommended by
[AutoDL's official network documentation](https://api.autodl.com/docs/network_turbo/).
The official `/etc/network_turbo` proxy was also verified, but the mirror gave
better throughput on a 16 MiB probe. For Llama, the author supplied an updated
credential after the existing login failed. The official HF API authorized the
fixed-revision objects, whose missing shards were fetched with parallel ranged
requests to HF's signed CDN URLs. No token or signed URL was saved by the final
downloader. Both complete checkpoints were SHA-256 verified against the original
local model files before execution. Receipts record source, revision, and hashes.

- `benchmark_latency.py`: natural greedy generation with 4-shot ICL versus CASS,
  batch sizes 1/8/25, 48-sequence extraction with batching, actual end-to-end
  adaptation, five repetitions, and sampled GPU board energy. Report hardware,
  software, warm-cache scope, and lack of shared-prefix caching explicitly.
- `compound_controls.py`: same additive direction, norm, gate and adaptive-scale
  rule in the correction-swap arms. The scale uses each arm's current state and
  the original inferred geometry; its realized value may differ after trajectories
  diverge. Swap only corrective projector/anchor: inferred, true, three
  wrong supports, or correction off. Wrong controls exclude inferred and true
  skills and preserve per-block rank. Mean/max within-support coherence must be
  within 0.02/0.03 to pass matching. Retain the match metadata and all generations.
- `qwen_probe.py`: eight original tasks, one fixed oracle-intervention grid;
  one/two layers and last-one/last-four/all-valid prefill positions, always with
  continued decode-step injection. This probes the intervention, not an isolated
  architecture-level cause.
- `summarize_remote.py`: checks experiment completeness and computes fresh-run
  paired task intervals, matched-wrong-support summaries, and measured latency
  or GPU-board-energy amortization only when per-query savings are positive.

Validation covers all 270 Compound task/seed/condition cells and 56 Qwen cells
against their saved generations, plus all 165 timing/power observations. The
earlier incomplete local latency attempt is archived separately and is not used
for any result or claim in the responses.

Completed fresh Qwen diagnostic: `qwen_probe_summary.json` aggregates all 56
task/condition rows (eight tasks, seven conditions); predictions and metadata
are retained. Mean accuracies: zero-shot 0.0201, ICL4 0.8098, two layers/last-one
0.1002, two layers/last-four 0.1279, two layers/all-valid 0.0726, layer 14 alone
0.0126, layer 20 alone 0.1627. These are a fixed diagnostic grid with no subsequent
hyperparameter selection. Pairwise intervals in the machine-readable summary
are descriptive, unadjusted for multiple comparisons. The weights were fully
SHA-256 checked against the original local checkpoint before GPU execution.

## Scope and claims to keep separate

- Existing 5–7-word capitalization is a multi-token formatting probe; it is not
  evidence for open-ended long-form generation, code, or tool use.
- Qwen already uses two injection layers and every decode step. Multi-layer
  injection therefore was not absent from the original negative result.
  Its exact failure mechanism is not established by healthy dictionary geometry.
- Existing scaling is limited to a 47-skill pool, six held-out tasks. Do not
  extrapolate it to hundreds of distinct skills or relabel duplicated skills as
  evidence of semantic scaling.
- Cross-model results use the same registered task data/splits and compare
  methods within each model. The base models' pretraining corpora are not
  controlled, so this does not isolate architecture as a causal explanation.
- The 10.2x number is an amortized input-token ratio versus **10-shot** ICL at
  1,000 queries, with dictionary mining charged separately. It is not measured
  speedup, energy saving, or the comparison with 4-shot ICL.
- Using the published rounded token counts, per-task extraction amortizes at
  about 53 queries against 4-shot ICL and 19 against 10-shot ICL. Full deployment
  accounting additionally needs dictionary mining and fallback costs.

## Reproduce the completed checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python rebuttal/2026-09-27/analyze_evidence.py all --all-loto
```

The theory note cites the primary group-Lasso source used to check the weighting
condition. It distinguishes a standard KKT argument from a new theoretical
contribution. English reviewer-specific drafts will be maintained alongside this
evidence rather than overwriting the submitted paper.
