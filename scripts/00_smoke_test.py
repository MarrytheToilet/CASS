"""Pipeline acceptance check (IDEA.md 7/4 gate).

For 5 tasks: ZS / 10-shot ICL accuracy, then inject the oracle mean diff
vector at several candidate layers into zero-shot prompts. Pass if the best
layer recovers >=70% of the (ICL - ZS) gap on most tasks.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cass.config import results_dir
from cass.evaluate import accuracy
from cass.extract import extract_task
from cass.models import HookedLM
from cass.steer import make_additive_op
from cass.tasks import load_task, icl_prompt, zs_prompt

MODEL = "llama31-8b"
TASKS = ["antonym", "country-capital", "english-french", "present-past",
         "choose-first-of-list"]
CAND_LAYERS = [6, 9, 12, 14, 16, 20, 24]
N_EVAL = 30
N_PAIRS = 50

def main():
    t0 = time.time()
    hlm = HookedLM(MODEL)
    print(f"loaded {MODEL}: L={hlm.L} d={hlm.d} ({time.time()-t0:.0f}s)", flush=True)
    report = {}
    for tname in TASKS:
        task = load_task(tname)
        queries = task.eval_queries[:N_EVAL]
        targets = [y for _, y in queries]
        rng = np.random.default_rng(0)

        zs_prompts = [zs_prompt(x) for x, _ in queries]
        icl_prompts = []
        pool = task.dict_pool
        for i, (x, _) in enumerate(queries):
            idx = rng.choice(len(pool), 10, replace=False)
            icl_prompts.append(icl_prompt([pool[j] for j in idx], x))

        acc_zs = accuracy(hlm.generate(zs_prompts, batch_size=16), targets)
        acc_icl = accuracy(hlm.generate(icl_prompts, batch_size=8), targets)

        G = extract_task(hlm, task, n_pairs=N_PAIRS, batch_size=6)  # [n, L+1, d]
        layer_accs = {}
        for l in CAND_LAYERS:
            mu = G[:, l, :].mean(0).numpy()
            op = make_additive_op(mu, gamma=1.0)
            preds = hlm.generate(zs_prompts, batch_size=16, op=op, layer=l)
            layer_accs[l] = accuracy(preds, targets)
        best_l = max(layer_accs, key=layer_accs.get)
        gap = acc_icl - acc_zs
        rec = (layer_accs[best_l] - acc_zs) / gap if gap > 0.02 else float("nan")
        report[tname] = dict(zs=acc_zs, icl=acc_icl, inj=layer_accs,
                             best_layer=best_l, recovery=rec)
        print(f"{tname}: zs={acc_zs:.2f} icl={acc_icl:.2f} "
              f"inj@{best_l}={layer_accs[best_l]:.2f} recovery={rec:.2f} "
              f"all={ {l: round(a,2) for l,a in layer_accs.items()} }", flush=True)

    out = results_dir(MODEL) / "smoke_test.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out}")

if __name__ == "__main__":
    main()
