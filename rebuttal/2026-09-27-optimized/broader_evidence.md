# Broader-task evidence

All 17 predeclared tasks are included. Settings were selected on the original 12 development targets and frozen before these evaluations. Each task uses four adaptation examples and three demonstration seeds. The one-demo tier retains the first of those examples in every query. Means below first average seeds within task and then tasks within each metric family. Metrics with different meanings are never pooled.

Classification uses 100 queries per task; entity extraction and reading comprehension also use 100 queries. Structured output and code probes use 60 queries per task. F1 is against the single reference in the repository’s dataset conversion, not the full official multi-reference benchmark score. Long outputs contain 20–30 words and require the entire extracted first answer line to match after whitespace normalization; the parser removes an opening code fence and ignores later lines. Code probes parse and execute a whitelisted expression and require an operation node, not just a matching numeric literal. The composed-reference min(sorted(xs)[:3]) case simplifies to min(xs); execution success does not establish an internal three-stage procedure.

These probes cover semantic-label/F1 evaluation, longer structured outputs, and controlled code/multi-step execution. They do not measure unrestricted prose generation, software engineering or real tool use.

## Complete plain-prompt comparisons

| Metric family | Tasks | Submitted | Contextual | Correction off | Dictionary-free | One-demo CASS | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification accuracy | 3 | 0.2578 | 0.7211 | 0.7211 | 0.4244 | 0.7533 | 0.7111 | 0.7956 |
| Entity extraction token F1 | 3 | 0.1956 | 0.5735 | 0.5829 | 0.4603 | 0.6021 | 0.4722 | 0.8613 |
| Reading comprehension token F1 | 1 | 0.8102 | 0.8870 | 0.8883 | 0.8886 | 0.8168 | 0.8817 | 0.9295 |
| Long structured first-answer-line accuracy | 3 | 0.1278 | 0.3037 | 0.3093 | 0.1796 | 0.5389 | 0.6630 | 0.6704 |
| Code-expression execution accuracy | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9319 | 0.9500 | 1.0000 |
| Composed code-expression execution accuracy | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5241 | 0.8759 | 1.0000 |

The contextual correction and its ablation use identical signatures, norm scaling, prompts, layers, schedules and inputs. The independently tuned dictionary-free baseline uses the winning setting from 552 development configurations. The broader gains over the submitted operator principally reflect the changed extraction/intervention; the extra projected correction does not improve every metric family. In the code probes, retaining a demonstration provides a usable format tier, while the zero-demonstration contextual arm has zero execution accuracy. The one-demo CASS tier is also compared directly to ICL1, which is stronger here. These are distinct cost/quality points.

## Every task

| Task | Submitted | Contextual | Correction off | Dictionary-free | One-demo CASS | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-max | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9944 | 1.0000 | 1.0000 |
| code-min | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9722 | 0.8000 | 1.0000 |
| code-sort3-max | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1278 | 0.8500 | 1.0000 |
| code-sort3-min | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9611 | 0.9944 | 1.0000 |
| code-sort3-sum | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4833 | 0.7833 | 1.0000 |
| code-sorted | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8667 | 1.0000 | 1.0000 |
| code-sum | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8944 | 1.0000 | 1.0000 |
| commonsense-choice | 0.7167 | 0.7133 | 0.7133 | 0.7233 | 0.7000 | 0.7200 | 0.7433 |
| entity-location | 0.1363 | 0.5284 | 0.5506 | 0.4308 | 0.5128 | 0.3460 | 0.8833 |
| entity-organization | 0.1778 | 0.3433 | 0.3312 | 0.2921 | 0.4452 | 0.3612 | 0.7406 |
| entity-person | 0.2728 | 0.8489 | 0.8670 | 0.6581 | 0.8483 | 0.7093 | 0.9600 |
| long-reverse-words | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0111 |
| long-title | 0.3833 | 0.3722 | 0.3722 | 0.5389 | 0.7111 | 0.9889 | 1.0000 |
| long-upper | 0.0000 | 0.5389 | 0.5556 | 0.0000 | 0.9056 | 1.0000 | 1.0000 |
| news-topic | 0.0567 | 0.5367 | 0.5267 | 0.3767 | 0.6233 | 0.4800 | 0.6733 |
| reading-comprehension | 0.8102 | 0.8870 | 0.8883 | 0.8886 | 0.8168 | 0.8817 | 0.9295 |
| sentiment | 0.0000 | 0.9133 | 0.9233 | 0.1733 | 0.9367 | 0.9333 | 0.9700 |

## Routing and uncertainty

The submitted norm threshold 5.125 and residual escalation threshold 0.7 are frozen. All 17 broader tasks escalate to ICL4 under that norm-plus-residual policy. This diagnoses the mismatch to the original short-task library; it is not evidence of zero-context accuracy or serving savings on those routed queries. Operator comparisons above are steering-only; fallback predictions are recorded separately.

Per-task paired intervals in `broader_analysis.json` bootstrap queries after averaging the three fixed demonstration seeds. They are conditional on those demonstrations. Cross-task intervals, where available, resample task means within a metric family and are based on few tasks. Raw predictions are rescored and all compared examples and queries are checked for identity.

## Separately frozen follow-ups

Native-template, multi-position and correction-geometry variants use their own development selections. Each complete condition is reported as a whole method across metric families; no per-family best configuration is combined into a synthetic CASS result.

### Native-template Llama

| Metric family | Best correction | Correction off | Dictionary-free | ICL4 |
|---|---:|---:|---:|---:|
| Classification accuracy | 0.3189 | 0.3189 | 0.3944 | 0.7089 |
| Entity extraction token F1 | 0.4869 | 0.4934 | 0.4172 | 0.8380 |
| Reading comprehension token F1 | 0.8467 | 0.8474 | 0.8539 | 0.9116 |
| Long structured first-answer-line accuracy | 0.3593 | 0.3537 | 0.0426 | 0.6148 |
| Code-expression execution accuracy | 0.0000 | 0.0000 | 0.0000 | 0.3764 |
| Composed code-expression execution accuracy | 0.0000 | 0.0000 | 0.0000 | 0.9259 |

### Position refinement

| Metric family | Best correction | Correction off | Dictionary-free | ICL4 |
|---|---:|---:|---:|---:|
| Classification accuracy | 0.7211 | 0.7211 | 0.5478 | 0.7956 |
| Entity extraction token F1 | 0.5735 | 0.5829 | 0.4903 | 0.8613 |
| Reading comprehension token F1 | 0.8870 | 0.8883 | 0.8186 | 0.9295 |
| Long structured first-answer-line accuracy | 0.3037 | 0.3093 | 0.1111 | 0.6704 |
| Code-expression execution accuracy | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Composed code-expression execution accuracy | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Weighted/rank correction refinement

| Metric family | Best correction | Correction off | Dictionary-free | ICL4 |
|---|---:|---:|---:|---:|
| Classification accuracy | 0.7211 | 0.7211 | 0.5478 | 0.7956 |
| Entity extraction token F1 | 0.5735 | 0.5829 | 0.4903 | 0.8613 |
| Reading comprehension token F1 | 0.8870 | 0.8883 | 0.8186 | 0.9295 |
| Long structured first-answer-line accuracy | 0.3037 | 0.3093 | 0.1111 | 0.6704 |
| Code-expression execution accuracy | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Composed code-expression execution accuracy | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
