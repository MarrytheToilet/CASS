# Qwen: intervention and prompt diagnostics

Two controlled oracle studies distinguish the injection primitive from few-shot support estimation. Each evaluates all specified conditions on 20 known targets excluded from development-target selection. Oracle task activations remain available in these diagnostic studies. They are not unseen-task steering scores.

## Prompt/layer/gain/position factorial

| Prompt | Layers | Gain | Prompt positions | Oracle accuracy |
|---|---|---:|---:|---:|
| plain | (14, 20) | 1 | 1 | 0.0810 |
| plain | (14, 20) | 1 | 4 | 0.1290 |
| plain | (14, 20) | 2 | 1 | 0.1030 |
| plain | (14, 20) | 2 | 4 | 0.1760 |
| plain | (24,) | 1 | 1 | 0.1540 |
| plain | (24,) | 1 | 4 | 0.1760 |
| plain | (24,) | 2 | 1 | 0.2090 |
| plain | (24,) | 2 | 4 | 0.2160 |
| chat | (14, 20) | 1 | 1 | 0.0430 |
| chat | (14, 20) | 1 | 4 | 0.0420 |
| chat | (14, 20) | 2 | 1 | 0.0420 |
| chat | (14, 20) | 2 | 4 | 0.0430 |
| chat | (24,) | 1 | 1 | 0.2380 |
| chat | (24,) | 1 | 4 | 0.2370 |
| chat | (24,) | 2 | 1 | 0.2630 |
| chat | (24,) | 2 | 4 | 0.2660 |

The native/chat condition includes a task-agnostic system instruction. The layer-setting contrast changes both depth and the number of intervened layers; it does not isolate depth alone. Prompt positions affect prefill; generation continues with the prescribed final-token intervention. Gain and the other factors remain fixed in each paired contrast.

## Separate format and instruction

All four prompt protocols are remined on the same device with identical 100-pair/10-example mining data and batch-8 extraction boundaries. Gain is two and one prompt position is intervened. The instruction is identical in content in plain and native formats:

> Infer the input-output rule from the examples when provided. Return only the output, without explanation.

| Native format | Generic instruction | Layers 14+20 | Layer 24 |
|---|---|---:|---:|
| False | False | 0.1030 | 0.2090 |
| False | True | 0.1510 | 0.2470 |
| True | False | 0.0100 | 0.0540 |
| True | True | 0.0420 | 0.2630 |

At native layer24, the instruction effect is +20.9 percentage points [12.4, 30.7]. Holding the instruction fixed, native versus plain format at layer24 differs by +1.6 [-2.6, 5.9]. The evidence identifies an instruction/intervention interaction; it does not establish native format alone as the remedy or identify a unique architectural cause.

Every conditional paired effect, including unfavorable conditions, is retained in `qwen_factorial_summary.json` and `qwen_prompt_factorial_summary.json`. Intervals resample 20 task means with 50,000 draws; these are individual contrast intervals, not simultaneous coverage over all factorial contrasts.

## Separately frozen few-shot confirmation

The initial Qwen3 intervention is selected on the original 12 development targets. It uses native format plus the generic instruction, a later-layer setting and the original shuffled contrast. The confirmation uses the same four examples in every arm and three new demonstration seeds. These results include support estimation.

| Suite | Original plain CASS | Selected plain CASS | Selected native CASS | Native z-only | Native ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.0740 | 0.1203 | 0.2420 | 0.2213 | 0.5451 |
| Novel15 | 0.0495 | 0.1582 | 0.2129 | 0.1857 | 0.7039 |
| Compound10 | 0.0050 | 0.0556 | 0.0533 | 0.0400 | 0.5047 |

The oracle is a reference intervention, not an intrinsic accuracy ceiling. The instruction adds prompt content and is included in both native CASS and native ICL controls. These Qwen results do not inherit the Llama-specific timing measurements.

### qwen3-4b: original queries

One setting is selected using the same 12 development tasks and then applied to all 45 non-development confirmation targets. “Best overall” is the development winner including zero correction; it is not selected on confirmation accuracy. Native prompting includes the same generic instruction in every native arm. The plain-original arm uses the submitted layers 14+20, gain 1 configuration. Qwen2.5 plain activations are freshly mined, so that arm is a matched rerun of the method rather than a bitwise replication of the original checkpointed activations.

The additional-query check keeps the same demonstrations and settings, using 50 target inputs disjoint from the target’s original evaluation and few-shot inputs. It was specified after the Qwen3 original-query result, without retuning, and before either model’s additional-query evaluation.

```json
{
  "best_overall": {
    "correction": "uncertainty",
    "gamma": 1.0,
    "kind": "context_residual",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "all"
  },
  "best_correction": {
    "correction": "uncertainty",
    "gamma": 1.0,
    "kind": "context_residual",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "all"
  },
  "development_overall": 0.5659722222222222,
  "development_correction": 0.5659722222222222,
  "source_sha256": "6640f472584540c64f6b409f5feb1aab219993f4bc552331d9b7fa1b9bcad31f"
}
```

Submitted prefix metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3864 | 0.3841 | 0.3864 | 0.2390 | 0.0740 | 0.5451 |
| Novel15 | 0.4461 | 0.4360 | 0.4461 | 0.2164 | 0.0495 | 0.7039 |
| Compound10 | 0.4357 | 0.3994 | 0.4357 | 0.0560 | 0.0050 | 0.5047 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +31.24 [+21.19, +42.23] |
| Non-development LOTO20 | best_correction − native_shuffle24 | +14.74 [+6.23, +24.10] |
| Non-development LOTO20 | best_correction − matched_no_correction | +0.23 [-1.07, +1.53] |
| Non-development LOTO20 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − plain_original | +39.66 [+28.01, +52.04] |
| Novel15 | best_correction − native_shuffle24 | +22.97 [+11.20, +36.10] |
| Novel15 | best_correction − matched_no_correction | +1.01 [-0.34, +2.42] |
| Novel15 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − plain_original | +43.07 [+23.00, +63.87] |
| Compound10 | best_correction − native_shuffle24 | +37.97 [+20.07, +57.37] |
| Compound10 | best_correction − matched_no_correction | +3.63 [+1.00, +7.29] |
| Compound10 | best_correction − best_overall | +0.00 [+0.00, +0.00] |

Case-aware literal-first-line metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.2137 | 0.1767 | 0.2137 | 0.1107 | 0.0097 | 0.5323 |
| Novel15 | 0.4114 | 0.3886 | 0.4114 | 0.2023 | 0.0347 | 0.6997 |
| Compound10 | 0.1656 | 0.1322 | 0.1656 | 0.0180 | 0.0036 | 0.4924 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +20.40 [+14.31, +26.77] |
| Non-development LOTO20 | best_correction − matched_no_correction | +3.70 [+1.63, +6.47] |
| Novel15 | best_correction − plain_original | +37.67 [+26.19, +49.66] |
| Novel15 | best_correction − matched_no_correction | +2.27 [+0.90, +3.78] |
| Compound10 | best_correction − plain_original | +16.20 [+7.46, +25.20] |
| Compound10 | best_correction − matched_no_correction | +3.34 [-0.07, +7.07] |

Generation source SHA256: `272b98f955ca5cae8ed7c43379fb94dd68aa85f39e0d5a0a519d1b1a9e7dfaa0`.

### qwen3-4b: 50 additional queries per task

One setting is selected using the same 12 development tasks and then applied to all 45 non-development confirmation targets. “Best overall” is the development winner including zero correction; it is not selected on confirmation accuracy. Native prompting includes the same generic instruction in every native arm. The plain-original arm uses the submitted layers 14+20, gain 1 configuration. Qwen2.5 plain activations are freshly mined, so that arm is a matched rerun of the method rather than a bitwise replication of the original checkpointed activations.

The additional-query check keeps the same demonstrations and settings, using 50 target inputs disjoint from the target’s original evaluation and few-shot inputs. It was specified after the Qwen3 original-query result, without retuning, and before either model’s additional-query evaluation.

```json
{
  "best_overall": {
    "correction": "uncertainty",
    "gamma": 1.0,
    "kind": "context_residual",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "all"
  },
  "best_correction": {
    "correction": "uncertainty",
    "gamma": 1.0,
    "kind": "context_residual",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "all"
  },
  "development_overall": 0.5659722222222222,
  "development_correction": 0.5659722222222222,
  "source_sha256": "6640f472584540c64f6b409f5feb1aab219993f4bc552331d9b7fa1b9bcad31f"
}
```

Submitted prefix metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3830 | 0.3653 | 0.3830 | 0.2463 | 0.0697 | 0.5333 |
| Novel15 | 0.4258 | 0.4080 | 0.4258 | 0.2049 | 0.0449 | 0.6929 |
| Compound10 | 0.4240 | 0.4060 | 0.4240 | 0.0467 | 0.0120 | 0.5093 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +31.33 [+21.90, +41.90] |
| Non-development LOTO20 | best_correction − native_shuffle24 | +13.67 [+6.73, +21.20] |
| Non-development LOTO20 | best_correction − matched_no_correction | +1.77 [+0.57, +3.20] |
| Non-development LOTO20 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − plain_original | +38.09 [+24.49, +52.44] |
| Novel15 | best_correction − native_shuffle24 | +22.09 [+9.29, +36.18] |
| Novel15 | best_correction − matched_no_correction | +1.78 [+0.18, +3.51] |
| Novel15 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − plain_original | +41.20 [+22.07, +61.27] |
| Compound10 | best_correction − native_shuffle24 | +37.73 [+19.33, +57.80] |
| Compound10 | best_correction − matched_no_correction | +1.80 [-0.47, +5.13] |
| Compound10 | best_correction − best_overall | +0.00 [+0.00, +0.00] |

Case-aware literal-first-line metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.2233 | 0.1807 | 0.2233 | 0.1197 | 0.0130 | 0.5157 |
| Novel15 | 0.4000 | 0.3778 | 0.4000 | 0.1956 | 0.0387 | 0.6902 |
| Compound10 | 0.1793 | 0.1633 | 0.1793 | 0.0173 | 0.0020 | 0.5027 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +21.03 [+14.13, +28.17] |
| Non-development LOTO20 | best_correction − matched_no_correction | +4.27 [+2.40, +6.27] |
| Novel15 | best_correction − plain_original | +36.13 [+22.40, +50.84] |
| Novel15 | best_correction − matched_no_correction | +2.22 [+0.80, +3.87] |
| Compound10 | best_correction − plain_original | +17.73 [+7.53, +29.60] |
| Compound10 | best_correction − matched_no_correction | +1.60 [-1.13, +4.60] |

Generation source SHA256: `45077f7acc4913a2c093440cb41163a22df7601c4818f1f8f766cf1d5a81b51c`.

### qwen25-3b: original queries

One setting is selected using the same 12 development tasks and then applied to all 45 non-development confirmation targets. “Best overall” is the development winner including zero correction; it is not selected on confirmation accuracy. Native prompting includes the same generic instruction in every native arm. The plain-original arm uses the submitted layers 14+20, gain 1 configuration. Qwen2.5 plain activations are freshly mined, so that arm is a matched rerun of the method rather than a bitwise replication of the original checkpointed activations.

The additional-query check keeps the same demonstrations and settings, using 50 target inputs disjoint from the target’s original evaluation and few-shot inputs. It was specified after the Qwen3 original-query result, without retuning, and before either model’s additional-query evaluation.

```json
{
  "best_overall": {
    "correction": 0.0,
    "gamma": 2.0,
    "kind": "original",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "prefill"
  },
  "best_correction": {
    "correction": "uncertainty",
    "gamma": 1.0,
    "kind": "context_residual",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "all"
  },
  "development_overall": 0.4826388888888889,
  "development_correction": 0.48263888888888884,
  "source_sha256": "4f14f6f52b28ea8a48f203bef374640650fc7daaed49d55058ded950ed300b1f"
}
```

Submitted prefix metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3464 | 0.3414 | 0.3630 | 0.1110 | 0.0223 | 0.5407 |
| Novel15 | 0.3519 | 0.3531 | 0.3814 | 0.0551 | 0.0161 | 0.7121 |
| Compound10 | 0.2296 | 0.2256 | 0.2900 | 0.0020 | 0.0027 | 0.4511 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +32.40 [+23.14, +41.97] |
| Non-development LOTO20 | best_correction − native_shuffle24 | +23.54 [+14.00, +33.53] |
| Non-development LOTO20 | best_correction − matched_no_correction | +0.50 [-0.67, +1.67] |
| Non-development LOTO20 | best_correction − best_overall | -1.66 [-8.30, +3.87] |
| Novel15 | best_correction − plain_original | +33.58 [+21.01, +47.63] |
| Novel15 | best_correction − native_shuffle24 | +29.68 [+18.16, +42.65] |
| Novel15 | best_correction − matched_no_correction | -0.13 [-1.90, +1.92] |
| Novel15 | best_correction − best_overall | -2.96 [-8.70, +3.88] |
| Compound10 | best_correction − plain_original | +22.70 [+6.30, +41.40] |
| Compound10 | best_correction − native_shuffle24 | +22.77 [+6.17, +41.80] |
| Compound10 | best_correction − matched_no_correction | +0.40 [-1.33, +2.33] |
| Compound10 | best_correction − best_overall | -6.04 [-11.13, -1.50] |

Case-aware literal-first-line metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.2970 | 0.2900 | 0.3310 | 0.0933 | 0.0067 | 0.5270 |
| Novel15 | 0.3376 | 0.3387 | 0.3733 | 0.0506 | 0.0067 | 0.7113 |
| Compound10 | 0.2230 | 0.2210 | 0.2867 | 0.0020 | 0.0000 | 0.4498 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +29.04 [+19.93, +38.77] |
| Non-development LOTO20 | best_correction − matched_no_correction | +0.70 [-0.83, +2.27] |
| Novel15 | best_correction − plain_original | +33.09 [+20.36, +47.22] |
| Novel15 | best_correction − matched_no_correction | -0.11 [-1.79, +1.87] |
| Compound10 | best_correction − plain_original | +22.30 [+6.23, +40.73] |
| Compound10 | best_correction − matched_no_correction | +0.20 [-1.40, +1.87] |

Generation source SHA256: `a56c51ede57cccd9f4bb191e7a0eb64eeac9abe28461ed83ea6a7f1fd852b346`.

### qwen25-3b: 50 additional queries per task

One setting is selected using the same 12 development tasks and then applied to all 45 non-development confirmation targets. “Best overall” is the development winner including zero correction; it is not selected on confirmation accuracy. Native prompting includes the same generic instruction in every native arm. The plain-original arm uses the submitted layers 14+20, gain 1 configuration. Qwen2.5 plain activations are freshly mined, so that arm is a matched rerun of the method rather than a bitwise replication of the original checkpointed activations.

The additional-query check keeps the same demonstrations and settings, using 50 target inputs disjoint from the target’s original evaluation and few-shot inputs. It was specified after the Qwen3 original-query result, without retuning, and before either model’s additional-query evaluation.

```json
{
  "best_overall": {
    "correction": 0.0,
    "gamma": 2.0,
    "kind": "original",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "prefill"
  },
  "best_correction": {
    "correction": "uncertainty",
    "gamma": 1.0,
    "kind": "context_residual",
    "layers": [
      24,
      28
    ],
    "null_fraction": 1.0,
    "schedule": "all"
  },
  "development_overall": 0.4826388888888889,
  "development_correction": 0.48263888888888884,
  "source_sha256": "4f14f6f52b28ea8a48f203bef374640650fc7daaed49d55058ded950ed300b1f"
}
```

Submitted prefix metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3323 | 0.3317 | 0.3657 | 0.1290 | 0.0227 | 0.5433 |
| Novel15 | 0.3644 | 0.3547 | 0.3689 | 0.0524 | 0.0138 | 0.7156 |
| Compound10 | 0.2313 | 0.2460 | 0.2747 | 0.0027 | 0.0040 | 0.4613 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +30.97 [+21.43, +40.80] |
| Non-development LOTO20 | best_correction − native_shuffle24 | +20.33 [+11.17, +29.90] |
| Non-development LOTO20 | best_correction − matched_no_correction | +0.07 [-1.17, +1.37] |
| Non-development LOTO20 | best_correction − best_overall | -3.33 [-9.33, +1.90] |
| Novel15 | best_correction − plain_original | +35.07 [+22.18, +49.51] |
| Novel15 | best_correction − native_shuffle24 | +31.20 [+19.24, +44.89] |
| Novel15 | best_correction − matched_no_correction | +0.98 [-0.31, +2.36] |
| Novel15 | best_correction − best_overall | -0.44 [-5.60, +5.11] |
| Compound10 | best_correction − plain_original | +22.73 [+6.80, +40.87] |
| Compound10 | best_correction − native_shuffle24 | +22.87 [+6.33, +41.40] |
| Compound10 | best_correction − matched_no_correction | -1.47 [-4.20, +0.80] |
| Compound10 | best_correction − best_overall | -4.33 [-8.87, -0.40] |

Case-aware literal-first-line metric:

| Suite | Best correction | Correction off | Best overall | Native shuffled layer24 | Plain original | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3030 | 0.3020 | 0.3460 | 0.1110 | 0.0077 | 0.5270 |
| Novel15 | 0.3569 | 0.3471 | 0.3578 | 0.0484 | 0.0071 | 0.7156 |
| Compound10 | 0.2180 | 0.2353 | 0.2713 | 0.0027 | 0.0000 | 0.4613 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − plain_original | +29.53 [+19.70, +39.90] |
| Non-development LOTO20 | best_correction − matched_no_correction | +0.10 [-0.93, +1.07] |
| Novel15 | best_correction − plain_original | +34.98 [+22.00, +49.47] |
| Novel15 | best_correction − matched_no_correction | +0.98 [-0.36, +2.40] |
| Compound10 | best_correction − plain_original | +21.80 [+6.60, +39.73] |
| Compound10 | best_correction − matched_no_correction | -1.73 [-4.47, +0.60] |

Generation source SHA256: `7f44588c2fe2ecc2ef3a0b5f25fdda01620e6322532616c58d2fb55f068d7a38`.
