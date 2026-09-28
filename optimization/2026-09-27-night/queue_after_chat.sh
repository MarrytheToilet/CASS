#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
cass_dir=optimization/2026-09-27-night
until test -f "$cass_dir/qwen_chat_dev_done.json"; do sleep 5; done
sleep 5
bash "$cass_dir/run_gpu.sh" 3 "$cass_dir/qwen_confirm.py"
bash "$cass_dir/run_gpu.sh" 3 "$cass_dir/confirm_llama.py"
bash "$cass_dir/run_gpu.sh" 3 "$cass_dir/llama_variants.py"
bash "$cass_dir/run_gpu.sh" 3 "$cass_dir/adaptive_calibration.py"
bash "$cass_dir/run_gpu.sh" 3 "$cass_dir/benchmark_optimized.py"
bash "$cass_dir/run_gpu.sh" 3 "$cass_dir/extract_scale_bank.py"
