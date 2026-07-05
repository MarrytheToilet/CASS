# CASS: Compositional Adaptive Subspace Steering

Training-free adaptation to unseen tasks by mining and composing skill
subspaces from LLM activations. Offline, contrastive activations of known
tasks are distilled into a dictionary of low-rank skill subspaces (with a
shared task-generic component removed). A novel task arriving with k≤4
examples is coded over the dictionary by group LASSO, and its queries are
served with zero context under a signal-routed steering policy.

## Setup

- Python 3.10, PyTorch 2.8, transformers 4.57 (conda env `llm_env`);
  single RTX 3090 (24 GB) suffices.
- Local model weights under `/home/hanyu/models/` (paths in
  `src/cass/config.py`; `HF_HUB_OFFLINE=1`). Main model:
  Llama-3.1-8B-Instruct; cross-model: Llama-3.2-3B, Gemma-2-2B,
  Qwen3-4B, Qwen2.5-3B.
- `.env` provides an OpenAI-compatible API endpoint, used **only** for
  dataset synthesis (07) and failure judging (17) — the method itself
  never calls an API.
- Task data from the Todd et al. function_vectors repo in `third_party/`.

## Layout

```
src/cass/    library: tasks, models (hooked LM), extract, dictionary,
             solver (group LASSO), steer, pipeline, compound, zcache,
             evaluate, api
scripts/     numbered pipeline (below)
results/     per-model outputs: CSVs, JSON, stored generations (jsonl)
figures/     generated figures (PNG)
paper/       LaTeX (not tracked in git)
```

## Pipeline

Scripts run in numeric order; every experiment checkpoints to
`results/<model>/` and resumes safely, so re-running a finished step is a
no-op. Multi-stage scripts take a stage argument shown in brackets.

| # | script | produces |
|---|--------|----------|
| 01 | `01_extract_and_baselines.py` | contrastive activations, ZS/10-shot ICL baselines |
| 02 | `02_injection_setup.py` [scan\|pair] | layer×γ steerability scan; frozen `injection_hparams.json` |
| 03 | `03_build_dictionary.py` | dictionary + H1 diagnostics (cosine matrices, μ_B) |
| 04 | `04_e1_loto.py` | E1 leave-one-task-out (`CASS_KS`, `CASS_SEEDS` env) |
| 05 | `05_analyze_e1.py` | `e1_summary.csv` (recovery ρ per task) |
| 06 | `06_e2_compound.py` | E2 compound tasks (case-sensitive) |
| 07 | `07_synthesize_tasks.py` | novel tasks via API (datasets only) |
| 08 | `08_e7_novel.py` | novel-suite evaluation (frozen dictionary) |
| 09 | `09_e4_ablations.py` | ablation axes (`CASS_REP=all` → all-32 replication) |
| 10 | `10_e5_scale.py` | dictionary-size scaling |
| 11 | `11_baselines.py` [lit\|blend32\|icl4\|hendel_c] | Hendel replace, ICV, retrieval, blend, 4-shot ICL, compound honesty check |
| 12 | `12_fill_gaps.py` | remaining Table-1 cells (incl. learned-Δ, oracle) |
| 13 | `13_review_response.py` | denoise/dictionary disentangle, random-dir control, hparam sensitivity |
| 14 | `14_improvements.py` | learned-Δ, signed gate, prefill-only |
| 15 | `15_router.py` | signal-aware routing (CASS full numbers) |
| 16 | `16_soft_hybrid.py` | soft-interpolation probe (hard routing wins) |
| 17 | `17_failure_analysis.py` [judge\|contains] | LLM-judge taxonomy, contains-vs-exact metric |
| 18 | `18_cost_latency.py` [tokens\|latency] | token cost table, latency micro-benchmark |
| 19 | `19_paper_figures.py` | all paper figures (PNG) |
| 20 | `20_table2.py` | Table 2 LaTeX from ablation CSVs |

Every number in the paper regenerates from the checkpointed CSVs and
stored generations without re-running GPU experiments (19/20 are
CPU-only). The full study used ~55 GPU-hours; the largest single run
(E1 LOTO over 32 tasks, all modes) takes under two hours.
