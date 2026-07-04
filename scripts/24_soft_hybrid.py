"""Soft interpolation hybrid: h <- h + beta*(v_prompt - h) + gated CASS
correction. beta=0 -> pure CASS; beta=1 -> replacement + correction.
Tested on 8 representative + 5 contrastively-unsteerable tasks."""
import csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from cass.config import results_dir
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, z_list_from_Z, _support_weights, _gated_op
from cass.steer import _to_torch
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt
from cass.zcache import get_z

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
TASKS = ["antonym", "present-past", "country-capital", "person-sport",
         "english-french", "next-item", "choose-first-of-list",
         "animal-from-list",
         "person-instrument", "capitalize-last-letter", "word-length",
         "next-capital-letter", "choose-last-of-list"]
BETAS = [0.3, 0.5, 0.7, 1.0]
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}
fout = open(out / "soft_hybrid.csv", "a", newline="")
w = csv.DictWriter(fout, fieldnames=["task", "seed", "beta", "acc"])
import os
if os.path.getsize(out / "soft_hybrid.csv") == 0:
    w.writeheader()

def soft_op(gop, v, beta):
    vt = _to_torch(v, "cuda")
    def op(h):
        h2 = gop(h.float())
        return h2 + beta * (vt.unsqueeze(0) - h.float())
    return op

t0 = time.time()
for tstar in TASKS:
    task = load_task(tstar)
    D = build_multilayer_dictionary(
        {l: {t: G[l][t] for t in ALL_TASKS if t != tstar} for l in LAYERS},
        r0=1)
    q = task.eval_queries[:25]
    targets = [y for _, y in q]
    prompts = [zs_prompt(x) for x, _ in q]
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(100 * seed + K)
        idx = rng.choice(len(task.fewshot_pool), K, replace=False)
        ex = [task.fewshot_pool[i] for i in idx]
        kp = [icl_prompt([e for i, e in enumerate(ex) if i != j], x)
              for j, (x, _) in enumerate(ex)]
        H = hlm.last_token_hiddens(kp, batch_size=8)
        V = {l: H[:, l].mean(0).numpy() for l in LAYERS}
        Z = get_z(hlm, task, K, seed)
        z_list = z_list_from_Z(D, Z)
        z_mean = np.mean(z_list, axis=0)
        code = code_for(D, z_list)
        # build gated CASS per-layer ops manually (delta = rescaled z)
        delta = z_mean.copy()
        if code.support:
            ws = _support_weights(code)
            target = float(sum(wi * np.linalg.norm(D.anchors[n])
                               for wi, n in zip(ws, code.support)))
            dn = np.linalg.norm(delta)
            if dn > 1e-8:
                delta *= target / dn
            mu_full = sum(wi * D.anchors[n] for wi, n in zip(ws, code.support))
            gate = max(0.0, float(delta @ mu_full /
                       (np.linalg.norm(delta) * np.linalg.norm(mu_full)
                        + 1e-12)))
        db = D.split(delta)
        for beta in BETAS:
            ops = []
            for l in LAYERS:
                if code.support:
                    Dl = D.per_layer[l]
                    mu_l = sum(wi * Dl.anchors[n]
                               for wi, n in zip(ws, code.support))
                    B_l = np.concatenate([Dl.bases[n] for n in code.support],
                                         axis=1)
                    gop = _gated_op(db[l], B_l, mu_l, gate, 1.0, 2.0, 1.0)
                else:
                    gop = lambda h: h + _to_torch(db[l], "cuda").unsqueeze(0)
                ops.append(soft_op(gop, V[l], beta))
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=ops,
                                        layer=list(LAYERS)), targets)
            w.writerow(dict(task=tstar, seed=seed, beta=beta, acc=acc))
            fout.flush()
    print(f"{tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)
fout.close()
print("SOFT HYBRID DONE")
