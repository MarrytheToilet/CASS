"""Hendel-replace baseline on the 10 compound tasks (honesty check)."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cass.config import results_dir
from cass.compound import COMPOUND_REGISTRY, load_compound
from cass.evaluate import accuracy, dump_preds
from cass.models import HookedLM
from cass.steer import _to_torch
from cass.tasks import icl_prompt, zs_prompt

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
fout = open(out / "hendel_compound.csv", "w", newline="")
w = csv.DictWriter(fout, fieldnames=["compound", "seed", "acc"])
w.writeheader()
def make_replace_op(vec):
    v = _to_torch(vec, "cuda")
    def op(h):
        return v.unsqueeze(0).expand_as(h).clone()
    return op
for cname in COMPOUND_REGISTRY:
    comp = load_compound(cname)
    q = comp.eval_queries
    targets = [y for _, y in q]
    prompts = [zs_prompt(x) for x, _ in q]
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(100 * seed + K)
        idx = rng.choice(len(comp.fewshot_pool), K, replace=False)
        ex = [comp.fewshot_pool[i] for i in idx]
        kp = []
        for j, (x, _) in enumerate(ex):
            shots = [e for i, e in enumerate(ex) if i != j]
            kp.append(icl_prompt(shots, x))
        H = hlm.last_token_hiddens(kp, batch_size=8)
        ops = [make_replace_op(H[:, l].mean(0).numpy()) for l in LAYERS]
        preds = hlm.generate(prompts, batch_size=25, op=ops,
                             layer=list(LAYERS))
        w.writerow(dict(compound=cname, seed=seed,
                        acc=accuracy(preds, targets, case_sensitive=True)))
        fout.flush()
    print(cname, "done", flush=True)
fout.close()
print("HENDEL COMPOUND DONE")
