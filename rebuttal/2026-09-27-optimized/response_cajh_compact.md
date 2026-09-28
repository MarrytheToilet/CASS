Response to Reviewer cajh — compact version

**1. End-to-end cost and quality.** The original 48 sequences are six batch-8 calls. On RTX 5090/Llama-3.1-8B-Instruct (BF16), extraction takes 0.183–0.216 s across three predeclared tasks. Selective-layer collection and cached Gram computation reduce warm task adaptation from 0.978–3.105 s to 0.276–0.369 s, preserving every generation in 447 query/batch comparisons. Optimized 48-sequence adaptation uses 83.9–105.7 GPU-board joules (original: 179.9–374.7 J).

Our separate contextual variant uses 28 sequences in two calls: 0.165–0.411 s and 50.4–76.5 GPU-board joules per adaptation. With the same uncompiled engine and reusable-prefix caching for both methods, batch-25 CASS/ICL4 serving takes 183–186/184–191 ms under an eight-token budget. Answer-delimited decoding gives task-dependent ICL4/CASS ratios of 0.826–1.031x. The paper's 10.2x ratio therefore describes amortized input tokens against ten-shot prompting, not latency. Tables cover batches 1/8/25 and all seven broader NLP tasks, with energy and output checks.

Settings selected on 12 development tasks were frozen before three-seed transfer confirmation, using the same four examples in all arms:

| Suite | Submitted steering | Contextual | Correction off | ICL4 |
|---|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3135 | 0.5022 | 0.4872 | 0.6456 |
| Novel15 | 0.3770 | 0.6555 | 0.6290 | 0.8480 |
| Compound10 | 0.1438 | 0.4694 | 0.4544 | 0.5570 |

Paired correction gains are +1.50 [0.03, 3.03], +2.65 [0.57, 5.29] and +1.50 [0.40, 2.67] points (95% task-bootstrap intervals). Fifty additional disjoint queries per task give 0.5050/0.6458/0.4640. These are additional steering-only experiments, distinct from the submitted routed system.

**2. Broader evaluation.** All 17 predeclared probes use frozen settings and three seeds. Contextual steering reaches 0.7211 classification accuracy, 0.5735 entity F1 and 0.8870 reading F1, versus 0.2578/0.1956/0.8102 for the submitted operator and 0.7956/0.8613/0.9295 for ICL4. Whole first-answer-line accuracy on 20–30-word structured outputs rises from 0.1278 to 0.3037 (ICL4: 0.6704). Correction-off controls separate these extraction/intervention gains from dictionary correction.

For execution-scored code/composed-expression probes, a tier retaining one demonstration reaches 0.9319/0.5241, versus one-shot ICL's 0.9500/0.8759. Zero-demo steering scores zero; the frozen residual policy escalates these probes to ICL4. We report every task and distinguish these controlled outputs from unrestricted prose, software generation and real tool use.

**3. Qwen.** A contextual Qwen3 configuration selected on the same 12 development tasks reaches prefix accuracy 0.3864/0.4461/0.4357 on LOTO20/Novel15/Compound10, versus the original configuration's 0.0740/0.0495/0.0050 and native ICL4's 0.5451/0.7039/0.5047. Paired gains over the original are +31.24 [21.19, 42.23], +39.66 [28.01, 52.04] and +43.07 [23.00, 63.87] points. Only disabling the correction gives 0.3841/0.4360/0.3994. Normalized whole-line scores are 0.2610/0.4181/0.3791; punctuation-preserving literal scores are 0.2137/0.4114/0.1656, with final punctuation explaining most of the Compound difference.

Two oracle factorials diagnose the intervention separately from support estimation. At native prompting/gain 2, layer 24 gives 0.263 versus 0.042 for layers 14+20. Four-position injection helps the plain two-layer setting but adds little to native layer 24. A matched-remine control attributes +20.9 [12.4, 30.7] points to adding the generic instruction at native layer 24; with instruction fixed, the native/plain difference is +1.6 [-2.6, 5.9]. This identifies configuration dependence rather than a family-wide inability to steer. Qwen2.5 independently reaches 0.3464/0.3519/0.2296 versus 0.0223/0.0161/0.0027 originally; its development-selected zero-correction configuration reaches 0.3630/0.3814/0.2900. Full metric/configuration tables include both.

**4. Hundreds of skills.** A separately specified 32/64/128/256-entry stress test adds 224 correlated synthetic transformations. Keeping the initial shared direction fixed gives contextual accuracy 0.5810/0.5877/0.5926/0.5920 on Novel15+Compound10. At 256, support Jaccard to 32/across seeds is 0.646/0.663 and constituent recall is 0.767. A 32-block shortlist reduces median coding from 0.653 to 0.213 s, with accuracy 0.5918.

Unchanged routing thresholds give 0.7007 versus ICL4's 0.7316, with 48% steering/52% ICL4 and 92% of routing decisions unchanged from 32. This is an operational routing result. For the submitted operator, a supervised norm-risk map fitted on 32 known tasks gives Brier 0.1510 versus the training prior's 0.2188 at 256; residual-only is 0.2235. Failure is defined per demonstration seed as accuracy < 0.5. Contextual risk maps do not beat their prior. All calibration protocols and shared-direction regimes are retained.
