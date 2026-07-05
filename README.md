<p align="center">
  <img src="assets/pr.png" alt="CASS — compose skills for unseen tasks" width="62%">
</p>

# CASS: Compositional Adaptive Subspace Steering

**Training-free adaptation to unseen tasks by mining and composing skill
subspaces from LLM activations.**

Few-shot prompting pays for the same demonstrations on every query. CASS
pays once: it distills what a model's activations already know about
previously seen tasks into a reusable *skill dictionary*, and serves a
brand-new task — arriving with as few as 4 examples — with **zero
in-context tokens**, at a fraction of ICL's serving cost.

---

## The story

A production LLM service is rarely killed by one big task; it is nibbled
to death by small ones — field normalization, label translation,
entity-to-attribute mapping — each arriving with a handful of examples
and recurring thousands of times. Prepending demonstrations works
uniformly well and costs uniformly much: a 10-shot prompt multiplies
per-query tokens by an order of magnitude, paid again on every call.

Activation steering offers a way out: a *task vector* distilled from
demonstrations, injected into the residual stream, triggers the task with
no demonstrations in context (Hendel et al. 2023; Todd et al. 2024). But
existing vector libraries only serve tasks they have already stored. The
case that dominates the long tail — an **unseen** task with a k≤4 example
budget — has no answer.

The obvious training-free move is to compose: approximate the new task's
vector from stored ones. **It does not work.** Averaging dictionary
vectors recovers *zero percent* of oracle steering performance on our
benchmark. Our diagnosis: every contrastively extracted task vector
shares a large **task-generic component** — the signature of "an
in-context task is being executed", independent of which one. Under
addition the shared parts compound while the task-specific parts dilute.

Projecting out this single shared direction (`U₀`, rank 1):

- drops the median cosine between task means from 0.34 to 0.18,
- exposes a family-block geometry that was invisible before,
- lifts downstream composition accuracy from **0.32 to 0.54** —
  while removing a *random* direction of the same rank does nothing (0.33).

<p align="center">
  <img src="figures/cosine_matrix_llama31-8b.png" width="72%"
       alt="Pairwise cosine between task means, before/after removing U0">
</p>

What remains is only *partially* compositional: a held-out task is well
approximated inside the span of a few related skills (currency lands on
{capital, landmark, park}, French on {German, Spanish}), but an
irreducible task-specific remainder lies outside every other task's
subspace, and steering fails without it. CASS is built around exactly
this division of labor.

## Method

<p align="center">
  <img src="assets/pipeline.png" alt="CASS pipeline" width="97%">
</p>

**1 — Mine skills (offline, once).** For each known task, contrast clean
10-shot prompts against label-deranged twins and record differential
activations at every layer in one forward pass. Remove the shared
task-generic component `U₀` (rank-1 SVD of stacked task means) and store
each skill as a low-rank subspace `Uₜ` with anchor `μₜ` at two
residual-stream depths (layers {12,16} of 32).

**2 — Code the task (few-shot).** Distill the k≤4 demonstrations into a
denoised direction `z` (6 resampled prompt pairs per example,
leave-self-out shots), then solve a **group LASSO** whose support is
shared across layers:

```
z ≈ Σₜ Uₜ cₜ + ε        (support S, code c, residual ε)
```

Block coordinate descent with exact group soft-thresholding; a
support-capped λ-path (s_max = 5) replaces cross-validation. The sparse
code doubles as an interpretable statement of *which known skills the new
task is made of*, and ε doubles as a reliability signal.

**3 — Serve queries (adaptive routing).** The extraction signals pick the
path per task:

| signal | route |
|---|---|
| strong, reliable `‖z‖` | **hybrid steering**: `h ← h + α̃γ·z + g·α·P_S(μ_S − h)` — the demonstration direction plus a trust-gated correction toward the recovered subspace |
| weak `‖z‖` (< 0.9 × median anchor norm) | **prompt-state replacement** (compression; Hendel-style) |
| high residual ε | **escalate to full ICL** |

Queries then run with zero context. Two propositions (support recovery
under block-coherence, error transfer) delineate where composition is
reliable, and their preconditions are *measured*, not assumed.

## Main results

Llama-3.1-8B-Instruct, strict exact-match, 3 seeds. Full tables in the paper.

| method (k=4) | LOTO-32 acc | median ρ | Novel-15 acc | Compound-10 acc |
|---|---|---|---|---|
| naive anchor averaging | 0.06 | 0.00 | 0.05 | 0.05 |
| retrieval (nearest skill) | 0.20 | 0.18 | 0.17 | 0.03 |
| ICV-style top-PC | 0.33 | 0.61 | 0.25 | — |
| z-only additive | 0.34 | 0.64 | 0.30 | — |
| task-vector replacement (Hendel) | 0.39 | 0.60 | 0.47 | 0.23 |
| CASS (composition only) | 0.39 | 0.85 | 0.37 | 0.16 |
| **CASS (full, routed)** | **0.46** | **0.87** | **0.50** | **0.29** |
| learned-Δ (40 grad steps, gradient reference) | 0.50 | 0.90 | 0.52 | 0.30 |
| 4-shot ICL (same examples, 4× tokens) | 0.72 | — | 0.85 | — |

ρ = recovery of the (ICL − zero-shot) gap. Highlights:

- **85%** of the 27 steerable held-out tasks exceed ρ = 0.6; the
  dictionary beats the demonstration vector alone by **+5.5 pts** pooled
  across 47 tasks (p = 2×10⁻⁴).
- **Interpretable composition**: on 10 compound tasks the top-weighted
  skill is a true constituent **9/10** times; supports are 61%
  same-family (random: 17%); inverse tasks select exactly their forward
  counterparts' subspaces.
- **Reliability signals for free**: ‖z‖ flags unsteerable tasks at AUC
  0.93; oracle recovery falls 0.87 / 0.74 / 0.51 across ε terciles, so ε
  prices the ICL fallback.
- **Cross-model**: transfers with only a layer-pair search to Llama-3.2-3B
  (+6.0 pts, p<10⁻⁴) and Gemma-2-2B (+4.2, p<10⁻⁴); the Qwen family is
  not steerable by this primitive (reported as a limitation, localized to
  the injection stage).
- **Serving cost**: hooks add 0% latency; 10-shot ICL is 2.6× slower
  wall-clock. Token break-even at ~17 queries, **10.2× cheaper** at 1k
  queries per task.

<p align="center">
  <img src="figures/e1_main_llama31-8b.png" width="97%"
       alt="E1 leave-one-task-out results">
</p>

<p align="center">
  <img src="figures/e2_heatmap_llama31-8b.png" width="97%"
       alt="E2 compound tasks: skill identification and execution">
</p>

<p align="center">
  <img src="figures/e5_scale_llama31-8b.png" width="48%"
       alt="Accuracy vs dictionary size">
  <img src="figures/e1_eps_vs_rho_llama31-8b.png" width="44%"
       alt="Oracle recovery vs coding residual">
</p>

*Left: held-out accuracy grows with dictionary size — each mined skill
makes every future unseen task easier. Right: the coding residual ε
predicts steering quality.*

## Reproducing

**Setup.** Python 3.10, PyTorch 2.8, transformers 4.57; one RTX 3090
(24 GB) suffices. Model weights are loaded locally
(paths in `src/cass/config.py`, `HF_HUB_OFFLINE=1`): Llama-3.1-8B-Instruct
(main), Llama-3.2-3B, Gemma-2-2B, Qwen3-4B, Qwen2.5-3B (cross-model).
Task data comes from the Todd et al. `function_vectors` repo in
`third_party/`. A `.env` with an OpenAI-compatible endpoint is needed
**only** for dataset synthesis (step 07) and failure judging (step 17) —
the method itself never calls an API.

**Run the pipeline.** Scripts run in numeric order; every experiment
checkpoints to `results/<model>/` and resumes safely (re-running a
finished step is a no-op). Multi-stage scripts take a stage argument.

```bash
python scripts/01_extract_and_baselines.py llama31-8b   # activations + ZS/ICL baselines
python scripts/02_injection_setup.py llama31-8b all     # steerability scan + freeze hparams
python scripts/03_build_dictionary.py llama31-8b        # dictionary + H1 diagnostics
CASS_KS=1,2,4 python scripts/04_e1_loto.py llama31-8b   # E1: leave-one-task-out
python scripts/05_analyze_e1.py llama31-8b              # per-task recovery table
python scripts/06_e2_compound.py llama31-8b             # E2: compound tasks
python scripts/07_synthesize_tasks.py                   # novel tasks via API (datasets only)
python scripts/08_e7_novel.py llama31-8b                # novel suite, frozen dictionary
CASS_REP=all python scripts/09_e4_ablations.py llama31-8b all   # ablation matrix
python scripts/10_e5_scale.py llama31-8b                # dictionary-size scaling
python scripts/11_baselines.py all                      # Hendel/ICV/retrieval/blend/ICL4
python scripts/12_fill_gaps.py                          # remaining Table-1 cells
python scripts/13_review_response.py                    # controls: random-dir, denoise, hparams
python scripts/14_improvements.py                       # learned-Δ, signed gate, prefill-only
python scripts/15_router.py                             # signal-aware routing (headline numbers)
python scripts/16_soft_hybrid.py                        # soft-vs-hard routing probe
python scripts/17_failure_analysis.py llama31-8b all    # LLM-judge taxonomy + contains metric
python scripts/18_cost_latency.py llama31-8b all        # token cost + latency benchmark
python scripts/19_paper_figures.py                      # all figures (PNG)
python scripts/20_table2.py                             # Table 2 LaTeX from CSVs
```

**Regenerating paper numbers.** Every number regenerates from the
checkpointed CSVs and stored generations (`results/*/**.jsonl`, enabling
offline re-scoring under any matcher) without re-running GPU experiments —
steps 19–20 are CPU-only. The full study used ~55 GPU-hours; the largest
single run (E1 over 32 tasks, all modes) takes under two hours.

## Layout

```
src/cass/    library: tasks, models (hooked LM), extract, dictionary,
             solver (group LASSO), steer, pipeline, compound, zcache,
             evaluate, api
scripts/     numbered pipeline 01–20 (above)
results/     per-model CSVs, JSON, stored generations (jsonl)
figures/     generated figures (PNG)
assets/      README images
third_party/ Todd et al. function_vectors task data
```

## Limitations

Short-form transformations with exact-match scoring; steerability of the
injection primitive is a model-family property (Qwen sits outside it —
run the one-time layer scan of step 02 before deploying); the layer pair
and three scalars are calibrated once per family; absolute accuracy
trails prompting — CASS is a cost–quality tier beneath ICL, with the
ε-fallback managing the gap.
