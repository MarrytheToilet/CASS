#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
export CASS_MODELS_DIR=/root/autodl-tmp/CASS-rebuttal-20260927/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${1:?physical GPU 2 or 3 required}"
cass_python=/root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python
case "$1" in
  2) exec "$cass_python" optimization/2026-09-27-night/llama_dev.py ;;
  3) exec "$cass_python" optimization/2026-09-27-night/qwen_scan.py ;;
  *) exit 2 ;;
esac
