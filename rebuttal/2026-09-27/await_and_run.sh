#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/CASS-rebuttal-20260927
cass_dir=rebuttal/2026-09-27
cass_python=/root/autodl-tmp/CASS-rebuttal-20260927/.venv/bin/python
case "${1:?pass 2 or 3}" in
  2)
    until test -f "$cass_dir/llama_transfer_receipt.json"; do sleep 15; done
    "$cass_python" "$cass_dir/check_model_hashes.py" Llama-3.1-8B-Instruct
    ;;
  3)
    until test -f "$cass_dir/model_download_receipts.json"; do sleep 15; done
    "$cass_python" "$cass_dir/check_model_hashes.py" Qwen3-4B
    ;;
  *) exit 2 ;;
esac
exec bash "$cass_dir/run_remote_pair.sh" "$1"
