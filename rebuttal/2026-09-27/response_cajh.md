Response to Reviewer cajh

**1. Measured adaptation and serving cost.** We benchmarked Llama-3.1-8B-Instruct on RTX 5090 (BF16, SDPA, PyTorch 2.13.0+cu130, Transformers 4.57.1), with five measured repetitions after warmup. Each target is excluded from its 31-skill dictionary. The 48 extraction sequences take 0.196–0.220 s at the original batch size of 8 (six model calls), or 0.140–0.160 s when concatenated into one batch of 48. Coding and operator construction add 0.749–3.201 s. Directly measured full adaptation and serving are:

| Task | Full adaptation (s) | CASS / ICL4, 25-query batch (ms) | Latency break-even (queries) |
|---|---:|---:|---:|
| Antonym | 1.507 | 217 / 246 | 1,316 |
| Country–capital | 0.995 | 221 / 256 | 711 |
| English–French | 3.256 | 219 / 257 | 2,162 |

At batch size 1, CASS takes 133–136 ms versus approximately 129 ms for ICL4, so it has no latency break-even there. At batch size 8, the measured advantage is small. These are warm hybrid-path measurements with natural greedy generation (maximum eight tokens), excluding model loading and offline dictionary construction, and without shared-prefix caching. Sampled GPU-board energy is approximately 183–366 J for full adaptation and 49–54 J per 25-query CASS batch versus 75–80 J for ICL4. The 100 ms power sampling excludes CPU/system energy; it does not establish whole-system energy savings.

The reported 10.2x ratio concerns input tokens versus **10-shot** ICL at 1,000 queries, with dictionary mining amortized separately. Against four-shot ICL, the published mean prompt counts are approximately 52.4 versus 10.5 tokens/query, plus 2,211 extraction tokens per task. These imply an input-token break-even at approximately 53 queries once the dictionary is available. Dictionary construction and ICL fallbacks must be added when accounting for a complete deployment. Four-shot ICL and CASS occupy different measured accuracy–cost points; the token calculation does not establish a wall-clock or energy advantage.

**2. Generalization beyond short transformations.** The existing additional probe uses 5–7-word targets: CASS obtains 0.793, four-shot ICL 1.00, and prompt-state replacement 0.00. The evaluator requires the complete target word sequence at the start of the first answer line, with the original normalization rules; it tolerates trailing continuation. This tests multi-token formatting under the frozen dictionary. It does not establish performance on open-ended long-form generation, code, tool use, or multi-step reasoning. We will state the evaluated task and scoring scope explicitly.

**3. Qwen and broader interventions.** We ran a new eight-task Qwen3-4B diagnostic using known-task oracle subspaces, with the original data and fixed gamma/beta/alpha_max=1/2/1. All interventions continue at every decoding step; the position variants modify prefill as well.

| Intervention | Mean accuracy |
|---|---:|
| Layers 14+20, last prefill token | 0.100 |
| Layers 14+20, last four prefill tokens | 0.128 |
| Layers 14+20, all valid prefill tokens | 0.073 |
| Layer 14 only, last token | 0.013 |
| Layer 20 only, last token | 0.163 |
| Four-shot ICL / zero-shot | 0.810 / 0.020 |

The multi-token changes do not close the gap in this fixed diagnostic. Layer 20 alone performs better than the pair, showing that intervention placement matters. Because the experiment uses known-task oracle subspaces, the failure does not depend on recovering an unseen task's support. It does not identify a unique architectural cause or prove a family-wide inability to steer. The original multi-layer negative finding should therefore be scoped to the tested model and intervention. These fresh RTX 5090 measurements use a checkpoint verified against the original model files; they are reported separately from the submitted experiments.

**4. Larger dictionaries and reliability.** We have measured a 47-task pool, using up to 46 skills when the target is held out. In the six-task scaling experiment, accuracy improves from 0.437 with 5 skills to 0.545 with 30, and is 0.519 with 46. This is evidence of benefit followed by saturation within that range. It does not establish stability at hundreds of skills. Residual calibration is also an empirical diagnostic: its reported fixed-k AUC is 0.62, while the signal norm's is 0.93. We will keep those roles distinct and avoid interpreting either score as a certified safety guarantee.
