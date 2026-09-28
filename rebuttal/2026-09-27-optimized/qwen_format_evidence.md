# Qwen output-format diagnosis

Post-hoc output diagnosis only; no parsing rule, setting, or accuracy value is changed.

The original metric accepts a normalized answer prefix. The two complete-first-line checks respectively normalize punctuation or preserve it literally; both retain task-relevant case. Suffix-only cases are literal mismatches repaired solely by removing final periods, question marks or exclamation marks. This diagnostic classification uses references and is not a proposed serving-time postprocessor.

## qwen3-4b_contrast_confirm

| Suite | Method | Case-aware prefix | Normalized full line | Literal full line | Prefix-only | Punctuation mismatch | Suffix-only mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| heldout_known | best_correction | 0.3864 | 0.2610 | 0.2137 | 0.1254 | 0.0473 | 0.0457 |
| heldout_known | matched_no_correction | 0.3841 | 0.2320 | 0.1767 | 0.1520 | 0.0553 | 0.0550 |
| heldout_known | plain_original | 0.0380 | 0.0097 | 0.0097 | 0.0283 | 0.0000 | 0.0000 |
| heldout_known | icl4 | 0.5451 | 0.5360 | 0.5323 | 0.0091 | 0.0037 | 0.0000 |
| novel | best_correction | 0.4461 | 0.4181 | 0.4114 | 0.0280 | 0.0067 | 0.0067 |
| novel | matched_no_correction | 0.4360 | 0.3989 | 0.3886 | 0.0371 | 0.0103 | 0.0103 |
| novel | plain_original | 0.0495 | 0.0351 | 0.0347 | 0.0144 | 0.0004 | 0.0004 |
| novel | icl4 | 0.7039 | 0.7015 | 0.6997 | 0.0024 | 0.0018 | 0.0018 |
| compound | best_correction | 0.4357 | 0.3791 | 0.1656 | 0.0566 | 0.2135 | 0.2101 |
| compound | matched_no_correction | 0.3994 | 0.3167 | 0.1322 | 0.0827 | 0.1845 | 0.1838 |
| compound | plain_original | 0.0050 | 0.0036 | 0.0036 | 0.0013 | 0.0000 | 0.0000 |
| compound | icl4 | 0.5047 | 0.4987 | 0.4924 | 0.0060 | 0.0063 | 0.0053 |

Values first average queries within each task/seed, then average the three seeds and equally weight tasks. Error-category rates are descriptive fractions, not new accuracy metrics.

Examples are the first six encountered per suite/category at seed20, with their raw strings retained in `qwen_format_diagnostics.json`.

Source SHA256: `272b98f955ca5cae8ed7c43379fb94dd68aa85f39e0d5a0a519d1b1a9e7dfaa0`.

## qwen3-4b_contrast_fresh

| Suite | Method | Case-aware prefix | Normalized full line | Literal full line | Prefix-only | Punctuation mismatch | Suffix-only mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| heldout_known | best_correction | 0.3830 | 0.2553 | 0.2233 | 0.1277 | 0.0320 | 0.0313 |
| heldout_known | matched_no_correction | 0.3653 | 0.2187 | 0.1807 | 0.1467 | 0.0380 | 0.0370 |
| heldout_known | plain_original | 0.0320 | 0.0130 | 0.0130 | 0.0190 | 0.0000 | 0.0000 |
| heldout_known | icl4 | 0.5333 | 0.5183 | 0.5157 | 0.0150 | 0.0027 | 0.0003 |
| novel | best_correction | 0.4258 | 0.4080 | 0.4000 | 0.0178 | 0.0080 | 0.0080 |
| novel | matched_no_correction | 0.4080 | 0.3876 | 0.3778 | 0.0204 | 0.0098 | 0.0098 |
| novel | plain_original | 0.0449 | 0.0387 | 0.0387 | 0.0062 | 0.0000 | 0.0000 |
| novel | icl4 | 0.6929 | 0.6916 | 0.6902 | 0.0013 | 0.0013 | 0.0013 |
| compound | best_correction | 0.4240 | 0.3620 | 0.1793 | 0.0620 | 0.1827 | 0.1820 |
| compound | matched_no_correction | 0.4060 | 0.3213 | 0.1633 | 0.0847 | 0.1580 | 0.1580 |
| compound | plain_original | 0.0120 | 0.0020 | 0.0020 | 0.0100 | 0.0000 | 0.0000 |
| compound | icl4 | 0.5093 | 0.5033 | 0.5027 | 0.0060 | 0.0007 | 0.0007 |

Values first average queries within each task/seed, then average the three seeds and equally weight tasks. Error-category rates are descriptive fractions, not new accuracy metrics.

Examples are the first six encountered per suite/category at seed20, with their raw strings retained in `qwen_format_diagnostics.json`.

Source SHA256: `45077f7acc4913a2c093440cb41163a22df7601c4818f1f8f766cf1d5a81b51c`.

## qwen25-3b_contrast_confirm

| Suite | Method | Case-aware prefix | Normalized full line | Literal full line | Prefix-only | Punctuation mismatch | Suffix-only mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| heldout_known | best_correction | 0.3224 | 0.2974 | 0.2970 | 0.0250 | 0.0003 | 0.0003 |
| heldout_known | matched_no_correction | 0.3217 | 0.2904 | 0.2900 | 0.0313 | 0.0003 | 0.0003 |
| heldout_known | plain_original | 0.0190 | 0.0087 | 0.0067 | 0.0103 | 0.0020 | 0.0000 |
| heldout_known | icl4 | 0.5321 | 0.5280 | 0.5270 | 0.0041 | 0.0010 | 0.0000 |
| novel | best_correction | 0.3505 | 0.3376 | 0.3376 | 0.0129 | 0.0000 | 0.0000 |
| novel | matched_no_correction | 0.3518 | 0.3387 | 0.3387 | 0.0131 | 0.0000 | 0.0000 |
| novel | plain_original | 0.0161 | 0.0067 | 0.0067 | 0.0094 | 0.0000 | 0.0000 |
| novel | icl4 | 0.7113 | 0.7113 | 0.7113 | 0.0000 | 0.0000 | 0.0000 |
| compound | best_correction | 0.2296 | 0.2230 | 0.2230 | 0.0067 | 0.0000 | 0.0000 |
| compound | matched_no_correction | 0.2256 | 0.2210 | 0.2210 | 0.0047 | 0.0000 | 0.0000 |
| compound | plain_original | 0.0027 | 0.0000 | 0.0000 | 0.0027 | 0.0000 | 0.0000 |
| compound | icl4 | 0.4511 | 0.4498 | 0.4498 | 0.0013 | 0.0000 | 0.0000 |

Values first average queries within each task/seed, then average the three seeds and equally weight tasks. Error-category rates are descriptive fractions, not new accuracy metrics.

Examples are the first six encountered per suite/category at seed20, with their raw strings retained in `qwen_format_diagnostics.json`.

Source SHA256: `a56c51ede57cccd9f4bb191e7a0eb64eeac9abe28461ed83ea6a7f1fd852b346`.

## qwen25-3b_contrast_fresh

| Suite | Method | Case-aware prefix | Normalized full line | Literal full line | Prefix-only | Punctuation mismatch | Suffix-only mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| heldout_known | best_correction | 0.3143 | 0.3030 | 0.3030 | 0.0113 | 0.0000 | 0.0000 |
| heldout_known | matched_no_correction | 0.3170 | 0.3020 | 0.3020 | 0.0150 | 0.0000 | 0.0000 |
| heldout_known | plain_original | 0.0197 | 0.0097 | 0.0077 | 0.0100 | 0.0020 | 0.0000 |
| heldout_known | icl4 | 0.5340 | 0.5277 | 0.5270 | 0.0063 | 0.0007 | 0.0000 |
| novel | best_correction | 0.3636 | 0.3569 | 0.3569 | 0.0067 | 0.0000 | 0.0000 |
| novel | matched_no_correction | 0.3538 | 0.3471 | 0.3471 | 0.0067 | 0.0000 | 0.0000 |
| novel | plain_original | 0.0138 | 0.0071 | 0.0071 | 0.0067 | 0.0000 | 0.0000 |
| novel | icl4 | 0.7156 | 0.7156 | 0.7156 | 0.0000 | 0.0000 | 0.0000 |
| compound | best_correction | 0.2313 | 0.2187 | 0.2180 | 0.0127 | 0.0007 | 0.0000 |
| compound | matched_no_correction | 0.2460 | 0.2360 | 0.2353 | 0.0100 | 0.0007 | 0.0000 |
| compound | plain_original | 0.0040 | 0.0000 | 0.0000 | 0.0040 | 0.0000 | 0.0000 |
| compound | icl4 | 0.4613 | 0.4613 | 0.4613 | 0.0000 | 0.0000 | 0.0000 |

Values first average queries within each task/seed, then average the three seeds and equally weight tasks. Error-category rates are descriptive fractions, not new accuracy metrics.

Examples are the first six encountered per suite/category at seed20, with their raw strings retained in `qwen_format_diagnostics.json`.

Source SHA256: `7f44588c2fe2ecc2ef3a0b5f25fdda01620e6322532616c58d2fb55f068d7a38`.
