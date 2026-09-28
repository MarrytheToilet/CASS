"""Re-extract the original fixed protocol locally on the authorized GPU server.

Used because transferring dense activation tensors was slower than recomputing
them. These fresh GPU controls are separate from checkpoint-only CPU reanalysis.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cass.models import HookedLM
from cass.extract import extract_and_save
from cass.tasks import ALL_TASKS,load_task
from cass.compound import COMPOUND_REGISTRY,load_compound
from cass.zcache import get_z


def main():
    model = sys.argv[1]
    assert model in ["llama31-8b","qwen3-4b"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ["2","3"]
    torch.set_num_threads(1)
    start = time.perf_counter()
    hlm = HookedLM(model)
    for t in ALL_TASKS:
        extract_and_save(hlm,load_task(t),n_pairs=100,n_shots=10,seed=0,batch_size=8)
        print(model,t,"extracted",round(time.perf_counter()-start,1),flush=True)
    if model == "llama31-8b":
        for t in COMPOUND_REGISTRY:
            for seed in [0,1,2]:
                get_z(hlm,load_compound(t),4,seed)
            print(model,t,"z cached",round(time.perf_counter()-start,1),flush=True)
    meta = dict(model=model,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                transformers=transformers.__version__,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                elapsed_seconds=time.perf_counter()-start,n_pairs=100,n_shots=10,batch_size=8,
                extraction_seed=0,fresh_remote_extraction=True)
    (HERE/f"reextract_{model}_done.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__ == "__main__":
    main()
