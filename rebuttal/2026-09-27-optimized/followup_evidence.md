# Separately frozen follow-up variants

Each development-selected variant transfers one fixed setting to every target. All-order enumeration is parameter-free given the original contextual setting. Results remain separate from the initial contextual confirmation and from each other. The same four examples and three demonstration seeds are used in compared arms. No table combines the best method from each suite or task.

The native-template condition adds the same generic system instruction to both steering and prompting baselines; its native submitted-operator arm also remakes the dictionary under that prompt format. Correction-off keeps dictionary-based direction calibration. The independently tuned dictionary-free arm omits the library. “Overall context setting” denotes the best setting within the contextual search, including zero correction; it does not select across unrelated methods on evaluation labels.

Prefix and strict case-aware literal-first-line metrics are reported separately. Paired intervals use task means over three seeds and 50,000 bootstrap draws. The additional-query split has50 disjoint target queries per task. Timing and memory measurements for the earlier operator are not automatically assigned to a different prompt or implementation.

## Native-template Llama

Development-frozen settings:

```json
{
  "development_means": {
    "best_correction": 0.5625,
    "best_overall": 0.5625,
    "dictionary_free": 0.4930555555555555
  },
  "best_correction": {
    "correction": "uncertainty",
    "gamma": 2.0,
    "kind": "context_residual",
    "layers": [
      16,
      20
    ],
    "mode": "context",
    "null_fraction": 1.0,
    "schedule": "prefill"
  },
  "best_overall": {
    "correction": "uncertainty",
    "gamma": 2.0,
    "kind": "context_residual",
    "layers": [
      16,
      20
    ],
    "mode": "context",
    "null_fraction": 1.0,
    "schedule": "prefill"
  },
  "dictionary_free": {
    "gamma": 1.0,
    "layers": [
      16,
      20
    ],
    "mode": "mean",
    "null_fraction": 1.0,
    "schedule": "prefill"
  },
  "source_sha256": "1d738c6acd4eb000216e53264f5a3d964aeebbd629220e4b2e8ea43db20935dc"
}
```

### Original queries / Submitted prefix metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Native submitted operator | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4057 | 0.4217 | 0.4057 | 0.3600 | 0.1410 | 0.5166 | 0.6209 |
| Novel15 | 0.4556 | 0.4536 | 0.4556 | 0.3924 | 0.1341 | 0.5078 | 0.7240 |
| Compound10 | 0.2890 | 0.2804 | 0.2890 | 0.2194 | 0.0187 | 0.4140 | 0.5204 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − native_original | +26.47 [+18.20, +34.97] |
| Non-development LOTO20 | best_correction − matched_no_correction | -1.60 [-3.33, -0.33] |
| Non-development LOTO20 | best_correction − dictionary_free | +4.57 [+0.13, +9.30] |
| Non-development LOTO20 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − icl4 | -21.53 [-29.95, -13.57] |
| Novel15 | best_correction − native_original | +32.16 [+19.37, +46.00] |
| Novel15 | best_correction − matched_no_correction | +0.20 [-1.07, +1.24] |
| Novel15 | best_correction − dictionary_free | +6.32 [+2.44, +10.59] |
| Novel15 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − icl4 | -26.83 [-40.32, -11.62] |
| Compound10 | best_correction − native_original | +27.02 [+13.49, +43.27] |
| Compound10 | best_correction − matched_no_correction | +0.86 [-0.00, +2.06] |
| Compound10 | best_correction − dictionary_free | +6.96 [+2.09, +12.16] |
| Compound10 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − icl4 | -23.14 [-35.60, -11.73] |

### Original queries / Case-aware literal-first-line metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Native submitted operator | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3597 | 0.3763 | 0.3597 | 0.2940 | 0.0887 | 0.4902 | 0.6021 |
| Novel15 | 0.4472 | 0.4456 | 0.4472 | 0.3824 | 0.1283 | 0.5011 | 0.7205 |
| Compound10 | 0.2816 | 0.2737 | 0.2816 | 0.2127 | 0.0187 | 0.4037 | 0.5118 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − matched_no_correction | -1.67 [-3.53, -0.37] |
| Non-development LOTO20 | best_correction − native_original | +27.10 [+18.70, +35.77] |
| Non-development LOTO20 | best_correction − dictionary_free | +6.57 [+1.20, +12.60] |
| Novel15 | best_correction − matched_no_correction | +0.16 [-1.16, +1.26] |
| Novel15 | best_correction − native_original | +31.89 [+19.24, +45.61] |
| Novel15 | best_correction − dictionary_free | +6.47 [+2.78, +10.59] |
| Compound10 | best_correction − matched_no_correction | +0.80 [-0.00, +1.86] |
| Compound10 | best_correction − native_original | +26.29 [+13.00, +42.60] |
| Compound10 | best_correction − dictionary_free | +6.89 [+2.09, +11.99] |

### Additional queries / Submitted prefix metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Native submitted operator | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3810 | 0.3893 | 0.3810 | 0.3367 | 0.1303 | 0.5307 | 0.6207 |
| Novel15 | 0.4751 | 0.4738 | 0.4751 | 0.4018 | 0.1493 | 0.4973 | 0.7187 |
| Compound10 | 0.2940 | 0.2847 | 0.2940 | 0.2333 | 0.0373 | 0.4000 | 0.4987 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − native_original | +25.07 [+16.03, +34.33] |
| Non-development LOTO20 | best_correction − matched_no_correction | -0.83 [-2.60, +0.73] |
| Non-development LOTO20 | best_correction − dictionary_free | +4.43 [+0.80, +8.23] |
| Non-development LOTO20 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − icl4 | -23.97 [-32.43, -15.77] |
| Novel15 | best_correction − native_original | +32.58 [+20.27, +45.96] |
| Novel15 | best_correction − matched_no_correction | +0.13 [-0.76, +0.89] |
| Novel15 | best_correction − dictionary_free | +7.33 [+3.82, +11.24] |
| Novel15 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − icl4 | -24.36 [-38.76, -9.33] |
| Compound10 | best_correction − native_original | +25.67 [+9.93, +43.80] |
| Compound10 | best_correction − matched_no_correction | +0.93 [-0.27, +2.20] |
| Compound10 | best_correction − dictionary_free | +6.07 [-0.27, +12.54] |
| Compound10 | best_correction − best_overall | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − icl4 | -20.47 [-31.40, -10.00] |

### Additional queries / Case-aware literal-first-line metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Native submitted operator | ICL1 | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3393 | 0.3477 | 0.3393 | 0.2807 | 0.0880 | 0.5060 | 0.6053 |
| Novel15 | 0.4609 | 0.4573 | 0.4609 | 0.3871 | 0.1413 | 0.4902 | 0.7156 |
| Compound10 | 0.2773 | 0.2693 | 0.2773 | 0.2307 | 0.0320 | 0.3880 | 0.4940 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − matched_no_correction | -0.83 [-2.57, +0.73] |
| Non-development LOTO20 | best_correction − native_original | +25.13 [+16.37, +34.20] |
| Non-development LOTO20 | best_correction − dictionary_free | +5.87 [+1.13, +11.63] |
| Novel15 | best_correction − matched_no_correction | +0.36 [-0.62, +1.29] |
| Novel15 | best_correction − native_original | +31.96 [+19.78, +45.24] |
| Novel15 | best_correction − dictionary_free | +7.38 [+3.91, +11.33] |
| Compound10 | best_correction − matched_no_correction | +0.80 [-0.40, +2.07] |
| Compound10 | best_correction − native_original | +24.53 [+10.07, +41.07] |
| Compound10 | best_correction − dictionary_free | +4.67 [-0.80, +9.93] |

## Prompt-position refinement

Development-frozen settings:

```json
{
  "best_correction": {
    "correction": "uncertainty",
    "gamma": 1.5,
    "kind": "context_residual",
    "null_fraction": 1.0,
    "pipeline": "context",
    "positions": 1,
    "schedule": "prefill"
  },
  "best_overall": {
    "correction": "uncertainty",
    "gamma": 1.5,
    "kind": "context_residual",
    "null_fraction": 1.0,
    "pipeline": "context",
    "positions": 1,
    "schedule": "prefill"
  },
  "dictionary_free": {
    "gamma": 1.0,
    "layers": [
      12,
      16
    ],
    "mode": "mean",
    "null_fraction": 0.5,
    "pipeline": "dictionary_free",
    "positions": "all",
    "schedule": "all"
  },
  "development_means": {
    "best_correction": 0.7638888888888888,
    "best_overall": 0.7638888888888888,
    "dictionary_free": 0.704861111111111
  },
  "source_sha256": "e38fcae5f4f0c6af6d79eeeb9657f497ab5d049aa3af9c57bb567b8186ef7744"
}
```

### Original queries / Submitted prefix metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5022 | 0.4872 | 0.5022 | 0.4379 | 0.5022 | 0.6456 |
| Novel15 | 0.6555 | 0.6290 | 0.6555 | 0.6487 | 0.6555 | 0.8480 |
| Compound10 | 0.4694 | 0.4544 | 0.4694 | 0.3055 | 0.4694 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − matched_no_correction | +1.50 [+0.03, +3.03] |
| Non-development LOTO20 | best_correction − dictionary_free | +6.43 [-0.70, +13.62] |
| Non-development LOTO20 | best_correction − icl4 | -14.34 [-21.73, -7.94] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.65 [+0.57, +5.29] |
| Novel15 | best_correction − dictionary_free | +0.67 [-7.50, +9.25] |
| Novel15 | best_correction − icl4 | -19.25 [-33.89, -7.71] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +1.50 [+0.40, +2.67] |
| Compound10 | best_correction − dictionary_free | +16.39 [+4.78, +29.87] |
| Compound10 | best_correction − icl4 | -8.76 [-15.95, -3.33] |

### Original queries / Case-aware literal-first-line metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4727 | 0.4560 | 0.4727 | 0.3621 | 0.4727 | 0.6347 |
| Novel15 | 0.6448 | 0.6184 | 0.6448 | 0.6456 | 0.6448 | 0.8467 |
| Compound10 | 0.4227 | 0.3991 | 0.4227 | 0.3022 | 0.4227 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − matched_no_correction | +1.67 [+0.33, +3.07] |
| Non-development LOTO20 | best_correction − dictionary_free | +11.06 [+1.13, +22.79] |
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.65 [+0.57, +5.28] |
| Novel15 | best_correction − dictionary_free | -0.08 [-8.03, +8.09] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +2.36 [+0.96, +3.90] |
| Compound10 | best_correction − dictionary_free | +12.05 [-3.22, +28.13] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |

### Additional queries / Submitted prefix metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5050 | 0.4867 | 0.5050 | 0.4757 | 0.5050 | 0.6357 |
| Novel15 | 0.6458 | 0.6209 | 0.6458 | 0.6444 | 0.6458 | 0.8391 |
| Compound10 | 0.4640 | 0.4587 | 0.4640 | 0.3180 | 0.4640 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − matched_no_correction | +1.83 [+0.57, +3.40] |
| Non-development LOTO20 | best_correction − dictionary_free | +2.93 [-5.53, +10.60] |
| Non-development LOTO20 | best_correction − icl4 | -13.07 [-20.90, -6.43] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.49 [+0.71, +4.67] |
| Novel15 | best_correction − dictionary_free | +0.13 [-6.76, +7.33] |
| Novel15 | best_correction − icl4 | -19.33 [-32.98, -9.29] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +0.53 [-0.27, +1.27] |
| Compound10 | best_correction − dictionary_free | +14.60 [+4.47, +26.80] |
| Compound10 | best_correction − icl4 | -10.13 [-19.00, -3.20] |

### Additional queries / Case-aware literal-first-line metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4790 | 0.4617 | 0.4790 | 0.4003 | 0.4790 | 0.6260 |
| Novel15 | 0.6382 | 0.6124 | 0.6382 | 0.6369 | 0.6382 | 0.8391 |
| Compound10 | 0.4120 | 0.3980 | 0.4120 | 0.3147 | 0.4120 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − matched_no_correction | +1.73 [+0.60, +3.17] |
| Non-development LOTO20 | best_correction − dictionary_free | +7.87 [-3.30, +20.53] |
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.58 [+0.71, +4.89] |
| Novel15 | best_correction − dictionary_free | +0.13 [-6.89, +7.47] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +1.40 [+0.40, +2.40] |
| Compound10 | best_correction − dictionary_free | +9.73 [-4.73, +24.40] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |

## Weighted/rank correction refinement

Development-frozen settings:

```json
{
  "best_correction": {
    "correction": "uncertainty",
    "correction_multiplier": 1.0,
    "gamma": 1.5,
    "kind": "context_residual",
    "null_fraction": 1.0,
    "pipeline": "projector",
    "positions": 1,
    "projection": "union",
    "schedule": "prefill"
  },
  "best_overall": {
    "correction": "uncertainty",
    "correction_multiplier": 1.0,
    "gamma": 1.5,
    "kind": "context_residual",
    "null_fraction": 1.0,
    "pipeline": "projector",
    "positions": 1,
    "projection": "union",
    "schedule": "prefill"
  },
  "dictionary_free": {
    "gamma": 1.0,
    "layers": [
      12,
      16
    ],
    "mode": "mean",
    "null_fraction": 0.5,
    "pipeline": "dictionary_free",
    "positions": "all",
    "schedule": "all"
  },
  "position_cass": {
    "correction": "uncertainty",
    "gamma": 1.5,
    "kind": "context_residual",
    "null_fraction": 1.0,
    "pipeline": "context",
    "positions": 1,
    "schedule": "prefill"
  },
  "source_sha256": "68f4f8d0a91933d8735a095fede773208bd24716b1d96884fab1e01f9e1b296c",
  "position_development_source_sha256": "e38fcae5f4f0c6af6d79eeeb9657f497ab5d049aa3af9c57bb567b8186ef7744",
  "development_means": {
    "best_correction": 0.7638888888888888,
    "best_overall": 0.7638888888888888
  }
}
```

### Original queries / Submitted prefix metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Position setting | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5022 | 0.4872 | 0.5022 | 0.4379 | 0.5022 | 0.5022 | 0.6456 |
| Novel15 | 0.6555 | 0.6290 | 0.6555 | 0.6487 | 0.6555 | 0.6555 | 0.8480 |
| Compound10 | 0.4694 | 0.4544 | 0.4694 | 0.3055 | 0.4694 | 0.4694 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − matched_no_correction | +1.50 [+0.03, +3.03] |
| Non-development LOTO20 | best_correction − dictionary_free | +6.43 [-0.70, +13.62] |
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − icl4 | -14.34 [-21.73, -7.94] |
| Novel15 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.65 [+0.57, +5.29] |
| Novel15 | best_correction − dictionary_free | +0.67 [-7.50, +9.25] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − icl4 | -19.25 [-33.89, -7.71] |
| Compound10 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +1.50 [+0.40, +2.67] |
| Compound10 | best_correction − dictionary_free | +16.39 [+4.78, +29.87] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − icl4 | -8.76 [-15.95, -3.33] |

### Original queries / Case-aware literal-first-line metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Position setting | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4727 | 0.4560 | 0.4727 | 0.3621 | 0.4727 | 0.4727 | 0.6347 |
| Novel15 | 0.6448 | 0.6184 | 0.6448 | 0.6456 | 0.6448 | 0.6448 | 0.8467 |
| Compound10 | 0.4227 | 0.3991 | 0.4227 | 0.3022 | 0.4227 | 0.4227 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − matched_no_correction | +1.67 [+0.33, +3.07] |
| Non-development LOTO20 | best_correction − dictionary_free | +11.06 [+1.13, +22.79] |
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.65 [+0.57, +5.28] |
| Novel15 | best_correction − dictionary_free | -0.08 [-8.03, +8.09] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +2.36 [+0.96, +3.90] |
| Compound10 | best_correction − dictionary_free | +12.05 [-3.22, +28.13] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − position_cass | +0.00 [+0.00, +0.00] |

### Additional queries / Submitted prefix metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Position setting | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5050 | 0.4867 | 0.5050 | 0.4757 | 0.5050 | 0.5050 | 0.6357 |
| Novel15 | 0.6458 | 0.6209 | 0.6458 | 0.6444 | 0.6458 | 0.6458 | 0.8391 |
| Compound10 | 0.4640 | 0.4587 | 0.4640 | 0.3180 | 0.4640 | 0.4640 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − matched_no_correction | +1.83 [+0.57, +3.40] |
| Non-development LOTO20 | best_correction − dictionary_free | +2.93 [-5.53, +10.60] |
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − icl4 | -13.07 [-20.90, -6.43] |
| Novel15 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.49 [+0.71, +4.67] |
| Novel15 | best_correction − dictionary_free | +0.13 [-6.76, +7.33] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − icl4 | -19.33 [-32.98, -9.29] |
| Compound10 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +0.53 [-0.27, +1.27] |
| Compound10 | best_correction − dictionary_free | +14.60 [+4.47, +26.80] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − icl4 | -10.13 [-19.00, -3.20] |

### Additional queries / Case-aware literal-first-line metric

| Suite | Correction setting | Correction off | Overall context setting | Dictionary-free | Position setting | Original contextual setting | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4790 | 0.4617 | 0.4790 | 0.4003 | 0.4790 | 0.4790 | 0.6260 |
| Novel15 | 0.6382 | 0.6124 | 0.6382 | 0.6369 | 0.6382 | 0.6382 | 0.8391 |
| Compound10 | 0.4120 | 0.3980 | 0.4120 | 0.3147 | 0.4120 | 0.4120 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | best_correction − matched_no_correction | +1.73 [+0.60, +3.17] |
| Non-development LOTO20 | best_correction − dictionary_free | +7.87 [-3.30, +20.53] |
| Non-development LOTO20 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Non-development LOTO20 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − matched_no_correction | +2.58 [+0.71, +4.89] |
| Novel15 | best_correction − dictionary_free | +0.13 [-6.89, +7.47] |
| Novel15 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Novel15 | best_correction − position_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − matched_no_correction | +1.40 [+0.40, +2.40] |
| Compound10 | best_correction − dictionary_free | +9.73 [-4.73, +24.40] |
| Compound10 | best_correction − frozen_cass | +0.00 [+0.00, +0.00] |
| Compound10 | best_correction − position_cass | +0.00 [+0.00, +0.00] |

## All six demonstration orders

### Original queries / Submitted prefix metric

| Suite | All orders | Correction off | Original contextual setting | Original correction off | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5035 | 0.4909 | 0.5022 | 0.4872 | 0.6456 |
| Novel15 | 0.6602 | 0.6358 | 0.6555 | 0.6290 | 0.8480 |
| Compound10 | 0.4690 | 0.4540 | 0.4694 | 0.4544 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | exhaustive_cass − frozen_cass | +0.13 [-0.27, +0.53] |
| Non-development LOTO20 | exhaustive_cass − exhaustive_off | +1.26 [-0.17, +2.67] |
| Non-development LOTO20 | exhaustive_cass − icl4 | -14.21 [-21.80, -7.77] |
| Novel15 | exhaustive_cass − frozen_cass | +0.47 [-0.22, +1.36] |
| Novel15 | exhaustive_cass − exhaustive_off | +2.44 [+0.57, +4.81] |
| Novel15 | exhaustive_cass − icl4 | -18.78 [-33.51, -6.98] |
| Compound10 | exhaustive_cass − frozen_cass | -0.04 [-0.84, +0.70] |
| Compound10 | exhaustive_cass − exhaustive_off | +1.50 [+0.27, +2.80] |
| Compound10 | exhaustive_cass − icl4 | -8.80 [-15.72, -3.67] |

### Original queries / Case-aware literal-first-line metric

| Suite | All orders | Correction off | Original contextual setting | Original correction off | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4734 | 0.4577 | 0.4727 | 0.4560 | 0.6347 |
| Novel15 | 0.6482 | 0.6233 | 0.6448 | 0.6184 | 0.8467 |
| Compound10 | 0.4217 | 0.3980 | 0.4227 | 0.3991 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | exhaustive_cass − frozen_cass | +0.07 [-0.33, +0.50] |
| Non-development LOTO20 | exhaustive_cass − exhaustive_off | +1.57 [+0.43, +2.80] |
| Novel15 | exhaustive_cass − frozen_cass | +0.33 [-0.20, +0.91] |
| Novel15 | exhaustive_cass − exhaustive_off | +2.48 [+0.75, +4.66] |
| Compound10 | exhaustive_cass − frozen_cass | -0.10 [-1.04, +0.73] |
| Compound10 | exhaustive_cass − exhaustive_off | +2.36 [+0.90, +3.76] |

### Additional queries / Submitted prefix metric

| Suite | All orders | Correction off | Original contextual setting | Original correction off | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.5057 | 0.4863 | 0.5050 | 0.4867 | 0.6357 |
| Novel15 | 0.6480 | 0.6209 | 0.6458 | 0.6209 | 0.8391 |
| Compound10 | 0.4693 | 0.4593 | 0.4640 | 0.4587 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | exhaustive_cass − frozen_cass | +0.07 [-0.27, +0.43] |
| Non-development LOTO20 | exhaustive_cass − exhaustive_off | +1.93 [+0.73, +3.33] |
| Non-development LOTO20 | exhaustive_cass − icl4 | -13.00 [-20.83, -6.37] |
| Novel15 | exhaustive_cass − frozen_cass | +0.22 [-0.49, +1.02] |
| Novel15 | exhaustive_cass − exhaustive_off | +2.71 [+1.07, +4.71] |
| Novel15 | exhaustive_cass − icl4 | -19.11 [-32.93, -8.84] |
| Compound10 | exhaustive_cass − frozen_cass | +0.53 [+0.13, +1.00] |
| Compound10 | exhaustive_cass − exhaustive_off | +1.00 [+0.20, +1.80] |
| Compound10 | exhaustive_cass − icl4 | -9.60 [-18.13, -2.87] |

### Additional queries / Case-aware literal-first-line metric

| Suite | All orders | Correction off | Original contextual setting | Original correction off | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4800 | 0.4597 | 0.4790 | 0.4617 | 0.6260 |
| Novel15 | 0.6409 | 0.6124 | 0.6382 | 0.6124 | 0.8391 |
| Compound10 | 0.4133 | 0.3993 | 0.4120 | 0.3980 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | exhaustive_cass − frozen_cass | +0.10 [-0.27, +0.50] |
| Non-development LOTO20 | exhaustive_cass − exhaustive_off | +2.03 [+0.97, +3.30] |
| Novel15 | exhaustive_cass − frozen_cass | +0.27 [-0.44, +1.07] |
| Novel15 | exhaustive_cass − exhaustive_off | +2.84 [+1.11, +4.93] |
| Compound10 | exhaustive_cass − frozen_cass | +0.13 [-0.47, +0.73] |
| Compound10 | exhaustive_cass − exhaustive_off | +1.40 [+0.60, +2.13] |

## Demonstration-only serving-tier selection

Compare the mean fourfold answer NLL of fixed CASS with three-shot ICL, using only the supplied four demonstrations. Route to ICL4 only when its ICL3 validation loss is strictly lower. There is no fitted margin or evaluation-label selection. The fixed CASS setting is used, not the adaptive grid minimum. This separate variant requires112 logical extraction sequences and eight teacher-forced forwards; it does not inherit the earlier adaptation timing. Its residual-policy comparator uses the submitted0.7 threshold without retuning.

| Suite | Demo-loss ICL4 fraction | Residual-policy ICL4 fraction |
|---|---:|---:|
| Non-development LOTO20 | 0.883 | 0.150 |
| Novel15 | 0.933 | 0.600 |
| Compound10 | 0.833 | 0.200 |

### Original queries / Submitted prefix metric

| Suite | Demo-loss route | Same route, correction off | Residual0.7 route | Pure contextual steering | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.6376 | 0.6339 | 0.5083 | 0.5022 | 0.6456 |
| Novel15 | 0.8527 | 0.8478 | 0.7881 | 0.6555 | 0.8480 |
| Compound10 | 0.5563 | 0.5543 | 0.4774 | 0.4694 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | teacher_route − frozen_cass | +13.54 [+7.07, +21.10] |
| Non-development LOTO20 | teacher_route − residual_route | +12.93 [+6.27, +20.63] |
| Non-development LOTO20 | teacher_route − teacher_route_off | +0.37 [-0.00, +0.97] |
| Non-development LOTO20 | teacher_route − icl4 | -0.80 [-1.57, -0.20] |
| Novel15 | teacher_route − frozen_cass | +19.72 [+8.89, +34.12] |
| Novel15 | teacher_route − residual_route | +6.46 [+0.91, +15.87] |
| Novel15 | teacher_route − teacher_route_off | +0.49 [+0.00, +1.43] |
| Novel15 | teacher_route − icl4 | +0.47 [-1.53, +2.93] |
| Compound10 | teacher_route − frozen_cass | +8.69 [+3.20, +15.95] |
| Compound10 | teacher_route − residual_route | +7.89 [+2.07, +15.42] |
| Compound10 | teacher_route − teacher_route_off | +0.20 [+0.00, +0.47] |
| Compound10 | teacher_route − icl4 | -0.07 [-0.80, +0.60] |

### Original queries / Case-aware literal-first-line metric

| Suite | Demo-loss route | Same route, correction off | Residual0.7 route | Pure contextual steering | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.6244 | 0.6217 | 0.4814 | 0.4727 | 0.6347 |
| Novel15 | 0.8474 | 0.8425 | 0.7797 | 0.6448 | 0.8467 |
| Compound10 | 0.5543 | 0.5510 | 0.4307 | 0.4227 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | teacher_route − frozen_cass | +15.17 [+8.60, +22.93] |
| Non-development LOTO20 | teacher_route − residual_route | +14.30 [+7.43, +22.30] |
| Non-development LOTO20 | teacher_route − teacher_route_off | +0.27 [-0.03, +0.73] |
| Non-development LOTO20 | teacher_route − icl4 | -1.03 [-1.90, -0.33] |
| Novel15 | teacher_route − frozen_cass | +20.26 [+9.46, +34.57] |
| Novel15 | teacher_route − residual_route | +6.77 [+1.02, +16.27] |
| Novel15 | teacher_route − teacher_route_off | +0.49 [+0.00, +1.43] |
| Novel15 | teacher_route − icl4 | +0.07 [-1.53, +1.73] |
| Compound10 | teacher_route − frozen_cass | +13.16 [+5.20, +22.48] |
| Compound10 | teacher_route − residual_route | +12.36 [+4.00, +22.05] |
| Compound10 | teacher_route − teacher_route_off | +0.33 [+0.00, +0.80] |
| Compound10 | teacher_route − icl4 | -0.27 [-0.80, +0.00] |

| Suite | Demo-loss ICL4 fraction | Residual-policy ICL4 fraction |
|---|---:|---:|
| Non-development LOTO20 | 0.883 | 0.150 |
| Novel15 | 0.933 | 0.600 |
| Compound10 | 0.833 | 0.200 |

### Additional queries / Submitted prefix metric

| Suite | Demo-loss route | Same route, correction off | Residual0.7 route | Pure contextual steering | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.6343 | 0.6300 | 0.5070 | 0.5050 | 0.6357 |
| Novel15 | 0.8453 | 0.8391 | 0.7747 | 0.6458 | 0.8391 |
| Compound10 | 0.5707 | 0.5680 | 0.4700 | 0.4640 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | teacher_route − frozen_cass | +12.93 [+6.43, +20.73] |
| Non-development LOTO20 | teacher_route − residual_route | +12.73 [+6.17, +20.60] |
| Non-development LOTO20 | teacher_route − teacher_route_off | +0.43 [-0.13, +1.27] |
| Non-development LOTO20 | teacher_route − icl4 | -0.13 [-0.93, +0.53] |
| Novel15 | teacher_route − frozen_cass | +19.96 [+10.18, +33.42] |
| Novel15 | teacher_route − residual_route | +7.07 [+1.87, +15.20] |
| Novel15 | teacher_route − teacher_route_off | +0.62 [+0.00, +1.73] |
| Novel15 | teacher_route − icl4 | +0.62 [-0.53, +2.40] |
| Compound10 | teacher_route − frozen_cass | +10.67 [+4.40, +19.20] |
| Compound10 | teacher_route − residual_route | +10.07 [+4.00, +18.67] |
| Compound10 | teacher_route − teacher_route_off | +0.27 [+0.00, +0.67] |
| Compound10 | teacher_route − icl4 | +0.53 [-0.40, +2.00] |

### Additional queries / Case-aware literal-first-line metric

| Suite | Demo-loss route | Same route, correction off | Residual0.7 route | Pure contextual steering | ICL4 |
|---|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.6233 | 0.6200 | 0.4850 | 0.4790 | 0.6260 |
| Novel15 | 0.8427 | 0.8351 | 0.7698 | 0.6382 | 0.8391 |
| Compound10 | 0.5687 | 0.5653 | 0.4193 | 0.4120 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | teacher_route − frozen_cass | +14.43 [+7.97, +22.10] |
| Non-development LOTO20 | teacher_route − residual_route | +13.83 [+7.23, +21.63] |
| Non-development LOTO20 | teacher_route − teacher_route_off | +0.33 [-0.13, +1.03] |
| Non-development LOTO20 | teacher_route − icl4 | -0.27 [-1.03, +0.37] |
| Novel15 | teacher_route − frozen_cass | +20.44 [+10.58, +33.82] |
| Novel15 | teacher_route − residual_route | +7.29 [+1.96, +15.47] |
| Novel15 | teacher_route − teacher_route_off | +0.76 [+0.00, +2.13] |
| Novel15 | teacher_route − icl4 | +0.36 [-0.53, +1.60] |
| Compound10 | teacher_route − frozen_cass | +15.67 [+6.73, +26.40] |
| Compound10 | teacher_route − residual_route | +14.93 [+5.80, +25.87] |
| Compound10 | teacher_route − teacher_route_off | +0.33 [+0.00, +0.80] |
| Compound10 | teacher_route − icl4 | +0.33 [-0.60, +1.60] |

## Shared-rank mechanism ablation

This is a fixed-setting mechanism ablation of the contextual extension. The original null contrast, gain, schedule and correction rule stay frozen while the number of removed shared directions changes at each selected layer. Bases, sparse supports and norm calibration are rebuilt consistently for each rank. No evaluation label selects a rank, and the ranks do not each receive separately tuned hyperparameters. The rank1 predictions are the main contextual results, with matched input checks and a numerical reconstruction check.

### Original queries / Submitted prefix metric

| Suite | Rank0 | Rank0 off | Rank1 | Rank1 off | Rank2 | Rank2 off | Rank4 | Rank4 off | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4499 | 0.4182 | 0.5022 | 0.4872 | 0.4752 | 0.4595 | 0.3594 | 0.3585 | 0.6456 |
| Novel15 | 0.5328 | 0.4904 | 0.6555 | 0.6290 | 0.6408 | 0.6267 | 0.5303 | 0.5188 | 0.8480 |
| Compound10 | 0.4116 | 0.3774 | 0.4694 | 0.4544 | 0.4786 | 0.4607 | 0.3410 | 0.3324 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | rank0 − rank1 | -5.23 [-10.57, +0.13] |
| Non-development LOTO20 | rank2 − rank1 | -2.70 [-9.37, +1.70] |
| Non-development LOTO20 | rank4 − rank1 | -14.28 [-27.17, -3.84] |
| Non-development LOTO20 | rank0 − rank0_off | +3.17 [+1.67, +4.80] |
| Non-development LOTO20 | rank1 − rank1_off | +1.50 [+0.03, +3.03] |
| Non-development LOTO20 | rank2 − rank2_off | +1.57 [+0.30, +3.20] |
| Non-development LOTO20 | rank4 − rank4_off | +0.10 [-0.63, +0.90] |
| Novel15 | rank0 − rank1 | -12.27 [-18.72, -5.46] |
| Novel15 | rank2 − rank1 | -1.47 [-4.48, +1.29] |
| Novel15 | rank4 − rank1 | -12.52 [-21.98, -5.18] |
| Novel15 | rank0 − rank0_off | +4.24 [+2.27, +6.68] |
| Novel15 | rank1 − rank1_off | +2.65 [+0.57, +5.29] |
| Novel15 | rank2 − rank2_off | +1.41 [-0.06, +3.31] |
| Novel15 | rank4 − rank4_off | +1.15 [+0.54, +1.82] |
| Compound10 | rank0 − rank1 | -5.78 [-9.60, -1.84] |
| Compound10 | rank2 − rank1 | +0.92 [-0.40, +2.49] |
| Compound10 | rank4 − rank1 | -12.84 [-32.07, -1.55] |
| Compound10 | rank0 − rank0_off | +3.42 [+1.69, +5.35] |
| Compound10 | rank1 − rank1_off | +1.50 [+0.40, +2.67] |
| Compound10 | rank2 − rank2_off | +1.79 [+0.47, +3.32] |
| Compound10 | rank4 − rank4_off | +0.86 [-0.33, +2.12] |

### Original queries / Case-aware literal-first-line metric

| Suite | Rank0 | Rank0 off | Rank1 | Rank1 off | Rank2 | Rank2 off | Rank4 | Rank4 off | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.3888 | 0.3525 | 0.4727 | 0.4560 | 0.4490 | 0.4340 | 0.3336 | 0.3306 | 0.6347 |
| Novel15 | 0.5137 | 0.4650 | 0.6448 | 0.6184 | 0.6350 | 0.6187 | 0.5169 | 0.5045 | 0.8467 |
| Compound10 | 0.2756 | 0.2387 | 0.4227 | 0.3991 | 0.4353 | 0.4127 | 0.3190 | 0.3071 | 0.5570 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | rank0 − rank1 | -8.38 [-16.43, -1.40] |
| Non-development LOTO20 | rank2 − rank1 | -2.37 [-7.93, +1.33] |
| Non-development LOTO20 | rank4 − rank1 | -13.91 [-26.40, -4.01] |
| Non-development LOTO20 | rank0 − rank0_off | +3.64 [+2.13, +5.17] |
| Non-development LOTO20 | rank1 − rank1_off | +1.67 [+0.33, +3.07] |
| Non-development LOTO20 | rank2 − rank2_off | +1.50 [+0.26, +3.00] |
| Non-development LOTO20 | rank4 − rank4_off | +0.30 [-0.50, +1.17] |
| Novel15 | rank0 − rank1 | -13.11 [-19.60, -6.14] |
| Novel15 | rank2 − rank1 | -0.98 [-3.84, +1.74] |
| Novel15 | rank4 − rank1 | -12.80 [-21.56, -5.60] |
| Novel15 | rank0 − rank0_off | +4.87 [+2.81, +7.39] |
| Novel15 | rank1 − rank1_off | +2.65 [+0.57, +5.28] |
| Novel15 | rank2 − rank2_off | +1.63 [+0.10, +3.60] |
| Novel15 | rank4 − rank4_off | +1.24 [+0.63, +1.89] |
| Compound10 | rank0 − rank1 | -14.71 [-25.98, -5.78] |
| Compound10 | rank2 − rank1 | +1.25 [-0.53, +3.18] |
| Compound10 | rank4 − rank1 | -10.37 [-31.00, +3.52] |
| Compound10 | rank0 − rank0_off | +3.69 [+1.47, +6.07] |
| Compound10 | rank1 − rank1_off | +2.36 [+0.96, +3.90] |
| Compound10 | rank2 − rank2_off | +2.26 [+0.80, +3.67] |
| Compound10 | rank4 − rank4_off | +1.19 [-0.21, +2.65] |

### Additional queries / Submitted prefix metric

| Suite | Rank0 | Rank0 off | Rank1 | Rank1 off | Rank2 | Rank2 off | Rank4 | Rank4 off | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4593 | 0.4320 | 0.5050 | 0.4867 | 0.4810 | 0.4583 | 0.3627 | 0.3557 | 0.6357 |
| Novel15 | 0.5458 | 0.5036 | 0.6458 | 0.6209 | 0.6289 | 0.6173 | 0.5218 | 0.5124 | 0.8391 |
| Compound10 | 0.3893 | 0.3647 | 0.4640 | 0.4587 | 0.4693 | 0.4567 | 0.3327 | 0.3247 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | rank0 − rank1 | -4.57 [-9.77, +1.43] |
| Non-development LOTO20 | rank2 − rank1 | -2.40 [-9.60, +1.93] |
| Non-development LOTO20 | rank4 − rank1 | -14.23 [-28.07, -2.73] |
| Non-development LOTO20 | rank0 − rank0_off | +2.73 [+1.33, +4.27] |
| Non-development LOTO20 | rank1 − rank1_off | +1.83 [+0.57, +3.40] |
| Non-development LOTO20 | rank2 − rank2_off | +2.27 [+1.37, +3.33] |
| Non-development LOTO20 | rank4 − rank4_off | +0.70 [-0.07, +1.43] |
| Novel15 | rank0 − rank1 | -10.00 [-15.78, -4.04] |
| Novel15 | rank2 − rank1 | -1.69 [-4.80, +0.98] |
| Novel15 | rank4 − rank1 | -12.40 [-21.29, -5.24] |
| Novel15 | rank0 − rank0_off | +4.22 [+2.31, +6.27] |
| Novel15 | rank1 − rank1_off | +2.49 [+0.71, +4.67] |
| Novel15 | rank2 − rank2_off | +1.16 [+0.04, +2.40] |
| Novel15 | rank4 − rank4_off | +0.93 [+0.18, +1.96] |
| Compound10 | rank0 − rank1 | -7.47 [-12.20, -2.73] |
| Compound10 | rank2 − rank1 | +0.53 [-0.60, +1.60] |
| Compound10 | rank4 − rank1 | -13.13 [-32.67, -1.67] |
| Compound10 | rank0 − rank0_off | +2.47 [+1.00, +4.27] |
| Compound10 | rank1 − rank1_off | +0.53 [-0.27, +1.27] |
| Compound10 | rank2 − rank2_off | +1.27 [+0.27, +2.33] |
| Compound10 | rank4 − rank4_off | +0.80 [-0.67, +2.40] |

### Additional queries / Case-aware literal-first-line metric

| Suite | Rank0 | Rank0 off | Rank1 | Rank1 off | Rank2 | Rank2 off | Rank4 | Rank4 off | ICL4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-development LOTO20 | 0.4000 | 0.3703 | 0.4790 | 0.4617 | 0.4560 | 0.4320 | 0.3417 | 0.3327 | 0.6260 |
| Novel15 | 0.5338 | 0.4889 | 0.6382 | 0.6124 | 0.6227 | 0.6111 | 0.5142 | 0.5049 | 0.8391 |
| Compound10 | 0.2493 | 0.2120 | 0.4120 | 0.3980 | 0.4153 | 0.3960 | 0.3007 | 0.2867 | 0.5653 |

| Suite | Paired comparison | Difference, pp [95% CI] |
|---|---|---:|
| Non-development LOTO20 | rank0 − rank1 | -7.90 [-16.20, -0.30] |
| Non-development LOTO20 | rank2 − rank1 | -2.30 [-9.00, +1.73] |
| Non-development LOTO20 | rank4 − rank1 | -13.73 [-27.03, -2.80] |
| Non-development LOTO20 | rank0 − rank0_off | +2.97 [+1.50, +4.60] |
| Non-development LOTO20 | rank1 − rank1_off | +1.73 [+0.60, +3.17] |
| Non-development LOTO20 | rank2 − rank2_off | +2.40 [+1.47, +3.43] |
| Non-development LOTO20 | rank4 − rank4_off | +0.90 [+0.10, +1.67] |
| Novel15 | rank0 − rank1 | -10.44 [-16.00, -4.62] |
| Novel15 | rank2 − rank1 | -1.56 [-4.67, +1.07] |
| Novel15 | rank4 − rank1 | -12.40 [-20.84, -5.29] |
| Novel15 | rank0 − rank0_off | +4.49 [+2.67, +6.49] |
| Novel15 | rank1 − rank1_off | +2.58 [+0.71, +4.89] |
| Novel15 | rank2 − rank2_off | +1.16 [+0.04, +2.40] |
| Novel15 | rank4 − rank4_off | +0.93 [+0.18, +1.96] |
| Compound10 | rank0 − rank1 | -16.27 [-26.27, -7.67] |
| Compound10 | rank2 − rank1 | +0.33 [-1.13, +1.67] |
| Compound10 | rank4 − rank1 | -11.13 [-30.67, +1.40] |
| Compound10 | rank0 − rank0_off | +3.73 [+1.53, +6.20] |
| Compound10 | rank1 − rank1_off | +1.40 [+0.40, +2.40] |
| Compound10 | rank2 − rank2_off | +1.93 [+0.67, +3.20] |
| Compound10 | rank4 − rank4_off | +1.40 [-0.13, +3.07] |

## Source generation hashes

```json
{
  "llama_native_confirm": "a85c9dc0ebccc8a373bbb980c098ef42fb408e318040e407783e5fe8ed6fb1d7",
  "llama_native_fresh": "ffd81c6a888ace58c21cff1eff9cba143562ea6c0ecdab89c20c9f0d0f8ad8f4",
  "position_confirm": "7fd4ab1b4d6b6a11bb18ae30f8305958970cbad530f1c6ca36a9bc9d61ee278c",
  "position_fresh": "a876c03cf796ad1b6a7bfadb0b5ca0b695fef66300cf735f5a51cc647d44c712",
  "projector_confirm": "0214ddba08e71a0e3f4e7975a0576a28610bae8c3240ddc95c0a51155d23cc6c",
  "projector_fresh": "d2b9684edca1b227575c68fc7bfbebc6ef131fd318f2a692d0f12daff1127d59",
  "exhaustive_orders_confirm": "180873c0fb0ce1782f59ee82463c6a75df9e329da9dcbf08074072e5fac13424",
  "exhaustive_orders_fresh": "30c62200eabb3d3890de72ca0667d39a36ae65de53a5a61a856a5ab236448802",
  "teacher_route_confirm": "31c9b3ba9799f44199c4fa02a16c0a65a74c50e13ba91abc8c3f31bb403ff8c5",
  "teacher_route_fresh": "21eeb4fb813300a2a15dccbea11616d82cf515843442e7f963cebf494616fe98",
  "shared_rank_confirm": "71cae8fb45a9e0c84965672defbf1a9ed39df36ffc185723e73013ab34523493",
  "shared_rank_fresh": "486869a647c401dd604bd99138cabbcc2cb99415f004b3d2364d77a14ed538ef"
}
```
