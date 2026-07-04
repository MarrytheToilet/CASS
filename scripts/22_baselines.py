"""Literature-grounded baselines on LOTO-32 + Novel-15 (same k=4 examples):
  hendel_replace : task vector = mean last-token hidden of k-shot prompts,
                   REPLACES the query hidden state (Hendel et al., 2023)
  icv_pc1        : top principal component of the contrastive diffs,
                   additive injection (ICV-style)
  retrieval      : nearest dictionary skill by cos(z, anchor), apply that
                   skill's oracle affine operator (ELICIT-style)
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cass.config import results_dir
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G, extract_fewshot_z
from cass.models import HookedLM
from cass.pipeline import oracle_ops
from cass.steer import make_additive_op, _to_torch
from cass.tasks import ALL_TASKS, load_task, icl_prompt, synthetic_tasks, \
    zs_prompt
from cass.zcache import get_z

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}
D_full = build_multilayer_dictionary(G, r0=1)

path = out / "baselines_lit.csv"
done = set()
if path.exists():
    with open(path) as f:
        done = {(r["suite"], r["method"], r["task"], r["seed"])
                for r in csv.DictReader(f)}
fout = open(path, "a", newline="")
w = csv.DictWriter(fout, fieldnames=["suite", "method", "task", "seed",
                                     "acc"])
if not done:
    w.writeheader()


def examples_for(task, seed):
    rng = np.random.default_rng(100 * seed + K)
    idx = rng.choice(len(task.fewshot_pool), K, replace=False)
    return [task.fewshot_pool[i] for i in idx]


def make_replace_op(vec, device="cuda"):
    v = _to_torch(vec, device)

    def op(h):
        return v.unsqueeze(0).expand_as(h).clone()
    return op


def eval_ops(task, ops, lys):
    q = task.eval_queries[:50]
    return accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                 batch_size=25, op=ops, layer=lys),
                    [y for _, y in q])


t0 = time.time()
for suite, names in [("loto", ALL_TASKS), ("novel", list(synthetic_tasks()))]:
    for tname in names:
        task = load_task(tname)
        D = (build_multilayer_dictionary(
            {l: {t: G[l][t] for t in ALL_TASKS if t != tname}
             for l in LAYERS}, r0=1) if suite == "loto" else D_full)
        for seed in [0, 1, 2]:
            need = [m for m in ["hendel_replace", "icv_pc1", "retrieval"]
                    if (suite, m, tname, str(seed)) not in done]
            if not need:
                continue
            examples = examples_for(task, seed)

            if "hendel_replace" in need:
                # mean last-token hidden of k-shot prompts (leave-self-out)
                prompts = []
                for j, (x, _) in enumerate(examples):
                    shots = [e for i, e in enumerate(examples) if i != j]
                    prompts.append(icl_prompt(shots, x))
                H = hlm.last_token_hiddens(prompts, batch_size=8)  # [k,L+1,d]
                ops = [make_replace_op(H[:, l].mean(0).numpy())
                       for l in LAYERS]
                w.writerow(dict(suite=suite, method="hendel_replace",
                                task=tname, seed=seed,
                                acc=eval_ops(task, ops, list(LAYERS))))
                fout.flush()

            if "icv_pc1" in need or "retrieval" in need:
                Z = get_z(hlm, task, K, seed)  # [k, L+1, d] mean-of-6 diffs

            if "icv_pc1" in need:
                # top PC of per-example diffs, sign-aligned to the mean,
                # rescaled to anchor norm like zvec, additive
                ops = []
                for l in LAYERS:
                    Zl = Z[:, l].numpy().astype(np.float64)
                    Zl = np.stack([D.per_layer[l].project_out_shared(z)
                                   for z in Zl])
                    U, s, _ = np.linalg.svd(Zl - Zl.mean(0), full_matrices=False)
                    pc = _pc = np.linalg.svd(Zl, full_matrices=False)[2][0]
                    if pc @ Zl.mean(0) < 0:
                        pc = -pc
                    norm = np.median([np.linalg.norm(
                        D.per_layer[l].anchors[t]) for t in D.task_names])
                    ops.append(make_additive_op(pc * norm, gamma=1.0))
                w.writerow(dict(suite=suite, method="icv_pc1", task=tname,
                                seed=seed,
                                acc=eval_ops(task, ops, list(LAYERS))))
                fout.flush()

            if "retrieval" in need:
                z = np.concatenate([D.per_layer[l].project_out_shared(
                    Z[:, l].numpy().astype(np.float64).mean(0))
                    for l in LAYERS])
                sims = {t: abs(float(z @ D.anchors[t] /
                        (np.linalg.norm(z) * np.linalg.norm(D.anchors[t])
                         + 1e-12))) for t in D.task_names}
                nearest = max(sims, key=sims.get)
                ops, lys = oracle_ops(D, nearest, 1.0, 2.0, 1.0)
                w.writerow(dict(suite=suite, method="retrieval", task=tname,
                                seed=seed, acc=eval_ops(task, ops, lys)))
                fout.flush()
        print(f"{suite}/{tname} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
fout.close()
print("LIT BASELINES DONE")
