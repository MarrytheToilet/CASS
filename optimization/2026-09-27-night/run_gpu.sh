#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
cass_gpu="${1:?GPU required}"
shift
case "$cass_gpu" in 2|3) ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="$cass_gpu" CASS_POWER_GPU="$cass_gpu"
export CASS_MODELS_DIR=/root/autodl-tmp/CASS-rebuttal-20260927/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/CASS-rebuttal-20260927/optimization/2026-09-27-night/inductor_cache
/root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python optimization/2026-09-27-night/record_run.py "$cass_gpu" "$@"
exec /root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python "$@"
