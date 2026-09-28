#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
export CASS_MODELS_DIR=/root/autodl-tmp/CASS-rebuttal-20260927/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
case "${1:?pass latency or controls}" in
  latency)
    export CUDA_VISIBLE_DEVICES=2 CASS_POWER_GPU=2
    exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python rebuttal/2026-09-27/benchmark_latency.py
    ;;
  controls)
    export CUDA_VISIBLE_DEVICES=3
    exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python rebuttal/2026-09-27/compound_controls.py
    ;;
  qwen)
    export CUDA_VISIBLE_DEVICES=2
    exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python rebuttal/2026-09-27/qwen_probe.py
    ;;
  *) exit 2 ;;
esac
