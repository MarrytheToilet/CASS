#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
cass_dir=optimization/2026-09-27-night
until test -f "$cass_dir/llama_dev_done.json"; do sleep 5; done
sleep 5
exec bash "$cass_dir/run_gpu.sh" 2 "$cass_dir/broader_run.py"
