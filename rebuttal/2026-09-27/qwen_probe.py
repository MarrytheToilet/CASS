"""Bounded Qwen3-4B oracle-injection diagnosis on the original 8-task panel.

Known-task oracle subspaces isolate injection from few-shot coding. Fixed
layers/scalars; no configuration is selected using these probe accuracies.
Compare one/two layers and last-one/last-four/all valid prefill positions.
All interventions continue at every decoding step. GPU selected externally.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_evidence import dictionary
from cass.evaluate import accuracy
from cass.models import HookedLM
from cass.pipeline import oracle_ops
from cass.tasks import load_task, zs_prompt, icl_prompt

TASKS = ["antonym", "present-past", "country-capital", "person-sport",
         "english-french", "next-item", "choose-first-of-list", "animal-from-list"]


@torch.no_grad()
def generate_positions(hlm, prompts, ops, layers, positions, batch_size=25):
    texts = []
    for i in range(0, len(prompts), batch_size):
        enc = hlm.tok(prompts[i:i+batch_size], return_tensors="pt", padding=True).to("cuda")
        mask0 = enc.attention_mask.bool()
        handles = []
        for operation, layer in zip(ops, layers):
            def hook(module, inputs, output, op=operation):
                h = output[0] if isinstance(output,tuple) else output
                if h.shape[1] == 1:
                    mask = torch.ones(h.shape[:2],device=h.device,dtype=torch.bool)
                else:
                    mask = mask0.clone()
                    if positions != "all":
                        mask[:,:-int(positions)] = False
                h[mask] = op(h[mask]).to(h.dtype)
            handles.append(hlm.layers[layer-1].register_forward_hook(hook))
        try:
            result = hlm.model.generate(**enc,max_new_tokens=8,do_sample=False,pad_token_id=hlm.tok.pad_token_id)
            texts += hlm.tok.batch_decode(result[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()
    return texts


def main():
    torch.set_num_threads(1)
    _, D = dictionary("qwen3-4b")
    hlm = HookedLM("qwen3-4b")
    metadata = dict(model="qwen3-4b",tasks=TASKS,layers=D.layers,scalars=[1,2,1],
                    torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    n_queries="all original held-out queries, up to 50",demonstration_seed=0,
                    calibration="none; diagnostic grid fixed before running",
                    scope="known-task oracle; cannot establish a unique architecture-level failure cause")
    (HERE/"qwen_probe_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    path = HERE/"qwen_probe.csv"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["task"],r["method"]) for r in csv.DictReader(f)}
    with path.open("a",newline="") as f, (HERE/"qwen_probe_generations.jsonl").open("a") as g:
        w = csv.DictWriter(f,fieldnames=["task","method","n_queries","acc"])
        if not done:
            w.writeheader()
        for name in TASKS:
            task = load_task(name)
            prompts = [zs_prompt(x) for x,_ in task.eval_queries]
            targets = [y for _,y in task.eval_queries]
            rng = np.random.default_rng(4)
            ex = [task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
            ops,layers = oracle_ops(D,name,1.,2.,1.)
            conditions = [("zero_shot",None,None,None), ("icl4",None,None,None),
                          ("pair_last1",ops,layers,"1"),("pair_last4",ops,layers,"4"),
                          ("pair_all_valid",ops,layers,"all"),
                          (f"single_{layers[0]}_last1",[ops[0]],[layers[0]],"1"),
                          (f"single_{layers[1]}_last1",[ops[1]],[layers[1]],"1")]
            for method, oo,ll,pp in conditions:
                if (name,method) in done:
                    continue
                if method == "zero_shot":
                    preds = hlm.generate(prompts,batch_size=25)
                elif method == "icl4":
                    preds = hlm.generate([icl_prompt(ex,x) for x,_ in task.eval_queries],batch_size=25)
                else:
                    preds = generate_positions(hlm,prompts,oo,ll,pp)
                acc = accuracy(preds,targets)
                w.writerow(dict(task=name,method=method,n_queries=len(targets),acc=acc)); f.flush()
                for (x,y),pred in zip(task.eval_queries,preds):
                    g.write(json.dumps(dict(task=name,method=method,input=x,target=y,pred=pred))+"\n")
                g.flush()
                print(name,method,round(acc,4),flush=True)


if __name__ == "__main__":
    main()
