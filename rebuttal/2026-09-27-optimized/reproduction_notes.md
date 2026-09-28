# Reproduction and evidence map

All experiment code and saved outputs for the optimization window are under
`optimization/2026-09-27-night/`. The submitted implementation in `src/cass/` and
the submitted manuscript are unchanged. Earlier submitted-method controls remain
under `rebuttal/2026-09-27/`. The new experiments use physical GPUs2 and3 only.

## Inputs and comparison units

The twelve development targets are country-capital, person-sport, antonym,
present-past, singular-plural, english-french, english-spanish, next-item,
word-length, choose-first-of-list, animal-from-list and color-from-list.
Development uses the first twelve few-shot-pool queries and four examples drawn
from the remaining pool with seeds10/11. The query and example pools are disjoint.
Settings are selected by average development accuracy, with deterministic first
grid-entry tie breaking, then saved with a SHA256 of the development ledger.

Confirmation uses seeds20/21/22. The primary known-task comparison excludes the
twelve development targets, leaving twenty targets; Novel15 and Compound10 are
separate suites. Known-target mining is excluded when that target is evaluated.
Development libraries can include other known skills, so the twenty-target split
is not a claim that their activations were absent from the whole research process.
Additional-query confirmation uses the last fifty entries of each target's
dictionary pool, checked to be disjoint from that target's original evaluation
and demonstration pools; the target itself is absent from the evaluation library.
These datasets had been seen in the project before this window, and the additional
checks are not described as a pristine blind benchmark.

The original score is normalized first-answer prefix accuracy, with case retained
on Compound tasks. A stricter first-answer-line score preserves punctuation,
signs and task-relevant case. It is reported separately, without switching the
primary metric according to significance. Task-bootstrap intervals average the
three demonstration seeds first and then resample tasks50,000 times with
seed20260927. Broader-task query intervals average the same three seeds per query
before resampling queries and are conditional on those demonstration draws.

## Experimental branches

| Branch | Selection / runner | Main output ledger | Scope |
|---|---|---|---|
| Submitted formula tuning | `llama_dev.py`, `confirm_llama.py` | `llama_confirm.jsonl` | Gain/correction/schedule changes to the original operator |
| Contrast changes | `signature_dev.py`, `signature_confirm.py` | `signature_confirm.jsonl` | Shuffled versus zero-context contrasts |
| Contextual correction | `combined_dev.py`, `freeze_extensions.py`, `extension_confirm.py` | `extension_confirm.jsonl`, `extension_fresh.jsonl` | New operator, frozen120-setting search; one-demo tier separate |
| Independent dictionary-free baseline | `dictionary_free_dev.py`, `dictionary_free_refine.py`, `dictionary_free_confirm.py` | `dictionary_free_refined_confirm.jsonl`, `dictionary_free_refined_fresh.jsonl` |552 development settings; no dictionary at inference |
| Support intervention | `support_controls.py` | `support_controls.jsonl` | Inferred/true/off/wrong controls, coherence/rank/norm matching |
| Demonstration-based adaptation | `adaptive_calibration.py`, `adaptive_combined.py` | `adaptive_eval.jsonl`, `adaptive_combined.jsonl` | Additional extraction and teacher-forced development losses; separate cost |
| Native Llama | `llama_native.py` | `llama_native_confirm.jsonl`, `llama_native_fresh.jsonl` |288 development configurations, native prompt with generic instruction |
| Prompt positions | `position_refine.py` | `position_confirm.jsonl`, `position_fresh.jsonl` |80 development configurations, separate dictionary-free refinement |
| Correction geometry | `projector_refine.py` | `projector_confirm.jsonl`, `projector_fresh.jsonl` |120 development settings, weighted/rank-truncated corrections |
| All demonstration orders | `exhaustive_orders.py` | `exhaustive_orders_confirm.jsonl`, `exhaustive_orders_fresh.jsonl` | All six permutations of three context examples; no label-selected setting |
| Demonstration-only routing | `teacher_routing.py` | `teacher_route_confirm.jsonl`, `teacher_route_fresh.jsonl` | Fixed fourfold CASS-vs-ICL3 NLL rule; additional adaptation cost |
| Qwen diagnosis | `qwen_factorial.py`, `qwen_prompt_factorial.py` | `qwen_factorial.jsonl`, `qwen_prompt_factorial.jsonl` | Oracle diagnostics; separates prompt/instruction/intervention factors |
| Qwen transfer | `qwen_contrast.py` | `qwen3-4b_contrast_confirm.jsonl`, `qwen25-3b_contrast_confirm.jsonl` |200 model-specific development configurations, three-seed confirmation |
| Broader metrics | `broader_run.py`, `broader_extensions.py`, `broader_native.py`, `broader_position.py` | `broader*.jsonl` | All seventeen tasks for each transferred setting; no per-task winning mixture |
| Dictionary scaling | `extract_scale_bank.py`, `scale_eval.py` | `scale_eval*.jsonl`, `scale_context*.jsonl` | Four sizes, two update policies and two solvers; submitted/new operators separate |

The protocol files beside these runners state the parameter grids and comparison
rules. All candidates are retained, including weaker settings. Every new branch
is a separate method comparison; the best result from each task or metric family
must not be assembled into an apparent single-method result.

## Source, generation and cache records

Each JSONL row stores its target, demonstration seed, method/configuration,
queries, raw decoded predictions and metric. Available demonstrations and
support/residual diagnostics are also stored. Some oracle diagnostics omit
demonstrations because their operator comes from the known target library.
Reused baseline outputs are accepted only after exact query/demo identity checks.

`record_run.py` saves code/configuration hashes, library versions and an archive
of relevant source files for jobs launched after it was installed during the
window. It deliberately records only a small allowlist of environment variables.
Earlier jobs retain their scripts, raw outputs and run logs; do not imply that
they acquired an earlier source archive retroactively. `job_*.log`, queue JSON
files and worker state files preserve completions and failed/retried jobs.

`validate_prediction_ledgers.py` recomputes metrics from saved generations,
checks unique row keys, input lengths, demonstration draws and declared query
splits. Its source-hashed result is `prediction_ledger_validation.json`.
This audit verifies recorded outputs and inputs, not model pretraining independence
or an internal mechanism of execution. Repeated predictions across ablations
must not be counted as independent test cases.

Activation tensors and run archives have been copied back to the workspace.
`cache_inventory.py create` on the server wrote hashes of those artifacts;
`cache_inventory.py verify` on the local copy verified all 2,217 files.
The inventory does not include model weights, which remain separately downloaded
artifacts. Qwen2.5's downloaded files have a dedicated pinned-revision checksum
record in `qwen25_verified.json`.

## Regenerate analysis

Run the following from the repository root using the configured Python environment
with NumPy, pandas, SciPy, scikit-learn, Matplotlib, Torch and Transformers:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python optimization/2026-09-27-night/validate_prediction_ledgers.py
python optimization/2026-09-27-night/summarize_night.py
python optimization/2026-09-27-night/robust_metrics.py
python optimization/2026-09-27-night/compare_extensions.py
python optimization/2026-09-27-night/analyze_extension_routing.py
python optimization/2026-09-27-night/analyze_support_controls.py
python optimization/2026-09-27-night/analyze_compound_stages.py
python optimization/2026-09-27-night/analyze_broader.py
python optimization/2026-09-27-night/analyze_qwen_factorial.py
python optimization/2026-09-27-night/analyze_qwen_prompt_factorial.py
python optimization/2026-09-27-night/analyze_scaling.py --stem scale_eval
python optimization/2026-09-27-night/analyze_scaling.py --stem scale_eval_fixed_shared
python optimization/2026-09-27-night/analyze_scaling.py --stem scale_context
python optimization/2026-09-27-night/analyze_scaling.py --stem scale_context_fixed_shared
python optimization/2026-09-27-night/analyze_scale_routing.py
python optimization/2026-09-27-night/write_confirmation_report.py
python optimization/2026-09-27-night/write_followup_report.py
python optimization/2026-09-27-night/write_broader_report.py
python optimization/2026-09-27-night/write_qwen_report.py
python optimization/2026-09-27-night/write_scaling_report.py
python optimization/2026-09-27-night/write_cost_report.py
```

GPU reruns use the archived version of each runner and its protocol, with the
same model revision, precision, batching, generation budget and prompt template.
Changing batching or compilation can change BF16 greedy generations; the timing
studies record those changes explicitly. The 48 logical extraction sequences are
not48 sequential model calls. Cached-prefix and answer-delimited timing contracts
are separate from uncached prompting and fixed-token-budget timing.


## Module diagram and numerical claim cross-check

`make_method_overview.py` exports the submitted three-stage information flow to
PDF, SVG and PNG under `figures/submitted_method_overview.*`. The proposed caption
is in `manuscript_revision.md`; the additional contextual operator is identified
separately.

`audit_response_numbers.py` checks the principal response tables and selected
numerical statements against the named analysis files. It records the source and
response hashes in `response_number_audit.json`. This is a numerical cross-check,
not an automatic validation of scientific interpretation or every numeral in the
prose. It complements the generation-ledger and metric recomputation checks.


## Shared-rank mechanism control for the extension

`SHARED_RANK_PROTOCOL.md` specifies a fixed-setting comparison of removed shared
rank0/1/2/4, each with a matched correction-off arm. Run
`shared_rank_ablation.py` and `shared_rank_ablation.py --fresh` for the original
and additional-query splits. The primary rank1 outputs are reused after input
checks and a reconstructed-operator generation check; every other rank is
retained. The study changes shared rank without retuning gain or schedule and
does not select a deployment rank from evaluation labels. Results appear in
`followup_evidence.md` after complete coverage and metric verification.

`make_shared_rank_figure.py` exports the complete original/additional-query
curves and paired rank-one versus rank-zero intervals to
`figures/shared_rank_control.*`, with CSV data and source hashes. It presents
all four specified ranks and does not select a new rank from evaluation results.

## Known-task risk fitting

`KNOWN_RISK_PROTOCOL.md` and `known_task_calibration.py` specify the additional
post-hoc supervised calibration check using32 known LOTO tasks only. Each
record uses one four-example extraction and one demonstration seed's accuracy;
this differs from the earlier three-seed task-mean endpoint. All three feature
mappings and the constant prior transfer unchanged to every scale condition.
Task-cluster intervals keep the three evaluation seeds together. The output
`known_calibration_evidence.md` reports all regimes and does not replace earlier
calibration protocols.

## Broader cached-serving comparison

`BROADER_LATENCY_PROTOCOL.md` fixes all seven NLP tasks, seed 20, batches 1/4/8,
32 queries per timed workload, seven timing repeats, a 32-token output limit
and first-answer stopping. `benchmark_broader_cached.py` uses the same
uncompiled reusable-prefix engine for contextual CASS and ICL4. Adaptation,
prefix construction and serving are measured separately. Run
`analyze_broader_latency.py` after the complete measurement file is present to
regenerate `broader_cost_evidence.md`, including GPU-board energy and all
cached/uncached generation and score checks. Timing intervals describe
within-run workload variation, not independent runs or per-request tail latency.
