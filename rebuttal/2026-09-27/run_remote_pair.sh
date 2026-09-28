#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
export CASS_MODELS_DIR=/root/autodl-tmp/CASS-rebuttal-20260927/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cass_python=/root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python
cass_dir=rebuttal/2026-09-27
case "${1:?pass 2 or 3}" in
  2)
    export CUDA_VISIBLE_DEVICES=2 CASS_POWER_GPU=2
    "$cass_python" "$cass_dir/reextract_remote.py" llama31-8b
    "$cass_python" "$cass_dir/benchmark_latency.py"
    ;;
  3)
    export CUDA_VISIBLE_DEVICES=3
    "$cass_python" "$cass_dir/reextract_remote.py" qwen3-4b
    "$cass_python" "$cass_dir/qwen_probe.py"
    until test -f "$cass_dir/reextract_llama31-8b_done.json"; do sleep 15; done
    "$cass_python" "$cass_dir/compound_controls.py"
    ;;
  *) exit 2 ;;
esac
