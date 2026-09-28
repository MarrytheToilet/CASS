"""Package only CASS code, public task data, and required activation layers.

Secrets, .env, git history, and unrelated projects are never included.
The compact activation snapshots preserve the selected tensors exactly.
"""
import io
import argparse
import json
import sys
import tarfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cass.tasks import ALL_TASKS
from cass.compound import COMPOUND_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only",action="store_true")
    args = parser.parse_args()
    manifest = []
    target = HERE / ("remote_code.tar.gz" if args.code_only else "remote_inputs.tar.gz")
    with tarfile.open(target, "w:gz") as tar:
        def add(path):
            rel = path.relative_to(ROOT)
            tar.add(path, arcname=str(rel), recursive=False)
            manifest.append(str(rel))
        def blob(rel, data):
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            manifest.append(rel)
        for path in (ROOT/"src/cass").glob("*.py"):
            add(path)
        for path in (ROOT/"third_party/function_vectors/dataset_files").rglob("*.json"):
            add(path)
        for path in (ROOT/"data/synthetic").glob("*.json"):
            add(path)
        for name in ["analyze_evidence.py","benchmark_latency.py","compound_controls.py","launch_remote.sh","qwen_probe.py","reextract_remote.py","run_remote_pair.sh"]:
            add(HERE/name)
        for name in ["compound_fixed_route_per_task.csv", "matched_results.json"]:
            add(HERE/name)
        for model in ["llama31-8b","qwen3-4b"]:
            r = ROOT/"results"/model
            add(r/"injection_hparams.json")
            add(r/"baselines.json")
            if args.code_only:
                continue
            layers = json.loads((r/"injection_hparams.json").read_text())["layers"]
            for task in ALL_TASKS:
                source = torch.load(r/"activations"/f"{task}.pt",map_location="cpu",weights_only=True)
                compact = dict(task=task,layers=layers,G_by_layer={l:source["G"][:,l].clone() for l in layers})
                buf = io.BytesIO(); torch.save(compact,buf)
                blob(f"results/{model}/activations_compact/{task}.pt",buf.getvalue())
        r = ROOT/"results/llama31-8b"
        for task in ([] if args.code_only else COMPOUND_REGISTRY):
            for seed in [0,1,2]:
                source = torch.load(r/"zcache"/f"{task}_k4_s{seed}.pt",map_location="cpu",weights_only=True)
                compact = torch.zeros_like(source)
                compact[:,[12,16]] = source[:,[12,16]]
                buf = io.BytesIO(); torch.save(compact,buf)
                blob(f"results/llama31-8b/zcache/{task}_k4_s{seed}.pt",buf.getvalue())
        blob("rebuttal/2026-09-27/bundle_manifest.json",json.dumps(manifest,indent=2).encode())
    print(target, target.stat().st_size, "bytes", len(manifest), "files")


if __name__ == "__main__":
    main()
