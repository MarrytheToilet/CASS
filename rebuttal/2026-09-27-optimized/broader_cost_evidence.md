# Cached serving across all seven broader NLP tasks

Device: NVIDIA GeForce RTX 5090, physical GPU3; Llama-3.1-8B-Instruct, BF16, PyTorch 2.13.0+cu130.

All seven previously specified NLP tasks are included. The frozen contextual operator and ICL4 use the same uncompiled greedy engine, a reusable prefix cache, a 32-token limit and first-answer stopping. Every timed repeat processes the same first 32 queries; each row gives workload throughput, not individual request latency. Prefix copying and expansion are timed. Different generated answer lengths remain part of the workload. Settings are fixed and seed 20 uses the same demonstrations as the broader accuracy study.

Independent bootstrap of seven within-run grouped timing repeats; not between-run uncertainty or per-request tail latency. GPU board only; 100ms sampling, CPU/system energy excluded.

| Task | Batch | CASS queries/s | ICL4 queries/s | Throughput ratio | CASS J/32 queries | ICL4 J/32 queries | Workload time saving, s [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| news-topic | 1 | 21.54 | 24.43 | 0.881 | 430.466 | 374.367 | [-0.1860, -0.1659] |
| news-topic | 4 | 47.42 | 70.07 | 0.677 | 216.113 | 161.288 | [-0.2203, -0.2165] |
| news-topic | 8 | 59.99 | 96.89 | 0.619 | 188.351 | 135.315 | [-0.2037, -0.2025] |
| sentiment | 1 | 24.90 | 25.23 | 0.987 | 354.014 | 353.036 | [-0.0237, -0.0098] |
| sentiment | 4 | 85.91 | 86.16 | 0.997 | 98.810 | 103.027 | [-0.0033, 0.0002] |
| sentiment | 8 | 163.21 | 160.92 | 1.014 | 61.327 | 65.184 | [0.0025, 0.0031] |
| commonsense-choice | 1 | 13.39 | 24.50 | 0.547 | 748.106 | 373.486 | [-1.1003, -1.0793] |
| commonsense-choice | 4 | 32.16 | 76.16 | 0.422 | 282.340 | 139.224 | [-0.5773, -0.5742] |
| commonsense-choice | 8 | 53.61 | 113.29 | 0.473 | 190.756 | 109.482 | [-0.3153, -0.3119] |
| reading-comprehension | 1 | 9.05 | 10.99 | 0.823 | 1289.337 | 1072.127 | [-0.6326, -0.6108] |
| reading-comprehension | 4 | 10.14 | 15.55 | 0.652 | 1077.841 | 819.591 | [-1.1019, -1.0886] |
| reading-comprehension | 8 | 13.32 | 17.81 | 0.748 | 907.763 | 774.696 | [-0.6083, -0.6023] |
| entity-person | 1 | 13.69 | 13.30 | 1.030 | 731.011 | 762.690 | [0.0561, 0.1021] |
| entity-person | 4 | 23.19 | 28.86 | 0.804 | 386.179 | 317.003 | [-0.2844, -0.2560] |
| entity-person | 8 | 29.40 | 44.15 | 0.666 | 317.570 | 235.384 | [-0.3731, -0.3582] |
| entity-organization | 1 | 6.24 | 16.02 | 0.390 | 1704.889 | 611.586 | [-3.1445, -3.1124] |
| entity-organization | 4 | 10.65 | 39.30 | 0.271 | 820.032 | 226.013 | [-2.1998, -2.1777] |
| entity-organization | 8 | 16.71 | 59.58 | 0.281 | 541.185 | 171.051 | [-1.3874, -1.3721] |
| entity-location | 1 | 20.03 | 17.39 | 1.152 | 459.864 | 553.539 | [0.2257, 0.2476] |
| entity-location | 4 | 44.01 | 39.39 | 1.117 | 209.492 | 231.325 | [0.0798, 0.0937] |
| entity-location | 8 | 70.44 | 57.86 | 1.217 | 146.047 | 180.331 | [0.0981, 0.1009] |

## Warm task setup

Models and dictionaries are already resident. Adaptation executes 28 sequences in seven batch-4 calls. Both methods’ prefix construction is measured separately; CASS adaptation includes extraction, sparse coding and operator construction. These batch-4 times are distinct from the two-call short-task measurements. The JSON retains model/library load times and nominal amortization estimates; a finite reliable break-even is not established when the serving-time saving includes zero.

| Task | Adaptation, s | Adaptation, GPU J | CASS prefix, s | ICL4 prefix, s | Same support on re-extraction | Maximum signature difference |
|---|---:|---:|---:|---:|---|---:|
| news-topic | 0.5334 | 257.072 | 0.0243 | 0.0536 | True | 0 |
| sentiment | 0.2414 | 87.202 | 0.0190 | 0.0262 | True | 0 |
| commonsense-choice | 0.4524 | 206.617 | 0.0220 | 0.0420 | True | 0 |
| reading-comprehension | 1.5314 | 809.990 | 0.0402 | 0.1513 | True | 0 |
| entity-person | 0.2400 | 98.117 | 0.0208 | 0.0286 | True | 0 |
| entity-organization | 0.3061 | 115.936 | 0.0206 | 0.0290 | True | 0 |
| entity-location | 0.3178 | 127.251 | 0.0212 | 0.0306 | True | 0 |

## Generation and score parity

Batch 4 compares all 100 query outputs to the saved accuracy run. Batches 1/8 compare the first 32 outputs to fresh uncached generation at that batch size. This check does not assume cached and uncached BF16 outputs are identical. Some saved ICL4 accuracy baselines predate first-answer stopping and contain continuations after the answer. Their raw-string differences therefore include those removed continuations; parsed first-answer differences and score differences are recorded separately. All timed arms use the same first-answer stopping contract. All raw before/after strings and per-query scores are retained.

| Task | Method | Batch | Queries | Different raw strings | Different first answers | Different scores | Uncached score | Cached score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| news-topic | cass | 1 | 32 | 1 | 1 | 1 | 0.5938 | 0.6250 |
| news-topic | icl4 | 1 | 32 | 0 | 0 | 0 | 0.7500 | 0.7500 |
| news-topic | cass | 4 | 100 | 3 | 3 | 1 | 0.5200 | 0.5300 |
| news-topic | icl4 | 4 | 100 | 100 | 2 | 1 | 0.6400 | 0.6500 |
| news-topic | cass | 8 | 32 | 1 | 1 | 0 | 0.5938 | 0.5938 |
| news-topic | icl4 | 8 | 32 | 0 | 0 | 0 | 0.7500 | 0.7500 |
| sentiment | cass | 1 | 32 | 2 | 0 | 0 | 0.8750 | 0.8750 |
| sentiment | icl4 | 1 | 32 | 0 | 0 | 0 | 0.9688 | 0.9688 |
| sentiment | cass | 4 | 100 | 3 | 0 | 0 | 0.9300 | 0.9300 |
| sentiment | icl4 | 4 | 100 | 100 | 0 | 0 | 0.9700 | 0.9700 |
| sentiment | cass | 8 | 32 | 1 | 0 | 0 | 0.8750 | 0.8750 |
| sentiment | icl4 | 8 | 32 | 0 | 0 | 0 | 0.9688 | 0.9688 |
| commonsense-choice | cass | 1 | 32 | 1 | 1 | 1 | 0.8125 | 0.7812 |
| commonsense-choice | icl4 | 1 | 32 | 1 | 1 | 1 | 0.7812 | 0.7500 |
| commonsense-choice | cass | 4 | 100 | 3 | 3 | 1 | 0.7000 | 0.6900 |
| commonsense-choice | icl4 | 4 | 100 | 100 | 0 | 0 | 0.7300 | 0.7300 |
| commonsense-choice | cass | 8 | 32 | 1 | 1 | 1 | 0.8125 | 0.7812 |
| commonsense-choice | icl4 | 8 | 32 | 1 | 1 | 1 | 0.7812 | 0.7500 |
| reading-comprehension | cass | 1 | 32 | 2 | 2 | 1 | 0.8506 | 0.8591 |
| reading-comprehension | icl4 | 1 | 32 | 0 | 0 | 0 | 0.9074 | 0.9074 |
| reading-comprehension | cass | 4 | 100 | 0 | 0 | 0 | 0.8748 | 0.8748 |
| reading-comprehension | icl4 | 4 | 100 | 100 | 0 | 0 | 0.9278 | 0.9278 |
| reading-comprehension | cass | 8 | 32 | 1 | 1 | 0 | 0.8591 | 0.8591 |
| reading-comprehension | icl4 | 8 | 32 | 0 | 0 | 0 | 0.9074 | 0.9074 |
| entity-person | cass | 1 | 32 | 0 | 0 | 0 | 0.7244 | 0.7244 |
| entity-person | icl4 | 1 | 32 | 0 | 0 | 0 | 0.9375 | 0.9375 |
| entity-person | cass | 4 | 100 | 2 | 2 | 1 | 0.7325 | 0.7359 |
| entity-person | icl4 | 4 | 100 | 100 | 0 | 0 | 0.9667 | 0.9667 |
| entity-person | cass | 8 | 32 | 1 | 1 | 1 | 0.7036 | 0.7244 |
| entity-person | icl4 | 8 | 32 | 0 | 0 | 0 | 0.9375 | 0.9375 |
| entity-organization | cass | 1 | 32 | 5 | 5 | 3 | 0.3983 | 0.4414 |
| entity-organization | icl4 | 1 | 32 | 0 | 0 | 0 | 0.7604 | 0.7604 |
| entity-organization | cass | 4 | 100 | 9 | 9 | 4 | 0.3044 | 0.3046 |
| entity-organization | icl4 | 4 | 100 | 100 | 2 | 1 | 0.7207 | 0.7107 |
| entity-organization | cass | 8 | 32 | 2 | 2 | 2 | 0.4097 | 0.4300 |
| entity-organization | icl4 | 8 | 32 | 1 | 1 | 0 | 0.7604 | 0.7604 |
| entity-location | cass | 1 | 32 | 0 | 0 | 0 | 0.5312 | 0.5312 |
| entity-location | icl4 | 1 | 32 | 1 | 1 | 0 | 0.8750 | 0.8750 |
| entity-location | cass | 4 | 100 | 0 | 0 | 0 | 0.4700 | 0.4700 |
| entity-location | icl4 | 4 | 100 | 100 | 4 | 1 | 0.8400 | 0.8300 |
| entity-location | cass | 8 | 32 | 0 | 0 | 0 | 0.5312 | 0.5312 |
| entity-location | icl4 | 8 | 32 | 4 | 4 | 1 | 0.8750 | 0.8438 |

Source SHA256: `983a0155f34bf15c86700ed59f2897d39dff3007b71ca86f7bbca0d1b1b58ba1`.
