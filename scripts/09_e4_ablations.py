"""E4 ablation matrix on 8 representative tasks (LOTO protocol, k=4 default).

Axes: r0, solver, injection form, layer, k, lambda, n (dictionary samples).
Usage: python 09_e4_ablations.py [model] [axis1,axis2,...|all]
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.dictionary import build_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, op_for
from cass.solver import lambda_max
from cass.tasks import ALL_TASKS, load_task, zs_prompt
from cass.zcache import get_z

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
AXES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"])
REP_TASKS = ["antonym", "present-past", "country-capital", "person-sport",
             "english-french", "next-item", "choose-first-of-list",
             "animal-from-list"]
SEEDS = [0, 1, 2]
N_EVAL = 25
K_DEFAULT = 4

out = results_dir(MODEL)
hp = json.load(open(out / "injection_hparams.json"))
L0, GAMMA, BETA, AMAX = hp["layer"], hp["gamma"], hp["beta"], hp["alpha_max"]

hlm = HookedLM(MODEL)
rows_path = out / "e4_ablations.csv"
done = set()
if rows_path.exists():
    with open(rows_path) as f:
        done = {(r["axis"], r["value"], r["task"], r["seed"])
                for r in csv.DictReader(f)}
fieldnames = ["axis", "value", "task", "seed", "acc", "eps", "support_size",
              "support"]
fout = open(rows_path, "a", newline="")
writer = csv.DictWriter(fout, fieldnames=fieldnames)
if not done:
    writer.writeheader()

_G_cache = {}
def G_at(layer):
    if layer not in _G_cache:
        _G_cache[layer] = {t: load_G(MODEL, t, layer).numpy()
                           for t in ALL_TASKS}
    return _G_cache[layer]


def run_condition(axis, value, tstar, seed, *, r0=2, solver="group_lasso",
                  injection="affine", layer=None, k=K_DEFAULT, lam=None,
                  n_samples=None):
    if (axis, str(value), tstar, str(seed)) in done:
        return
    layer = layer or L0
    task = load_task(tstar)
    G = G_at(layer)
    G_lo = {}
    for t in ALL_TASKS:
        if t == tstar:
            continue
        g = G[t]
        G_lo[t] = g[:n_samples] if n_samples else g
    D = build_dictionary(G_lo, r0=r0)
    Z = get_z(hlm, task, k, seed)
    z_list = [D.project_out_shared(Z[j, layer].numpy().astype(np.float64))
              for j in range(Z.shape[0])]
    if isinstance(lam, float):  # lambda expressed as fraction of lambda_max
        z_mean = np.mean(z_list, axis=0)
        code = code_for(D, z_list, solver=solver,
                        lam=lam * lambda_max(D, z_mean))
    else:
        code = code_for(D, z_list, solver=solver)
    op = op_for(D, code, GAMMA, BETA, AMAX, injection=injection)
    queries = task.eval_queries[:N_EVAL]
    acc = accuracy(hlm.generate([zs_prompt(x) for x, _ in queries],
                                batch_size=25, op=op, layer=layer),
                   [y for _, y in queries])
    writer.writerow(dict(axis=axis, value=value, task=tstar, seed=seed,
                         acc=acc, eps=round(code.residual, 4),
                         support_size=len(code.support),
                         support="|".join(code.support[:8])))
    fout.flush()


def sweep(axis):
    t0 = time.time()
    grids = {
        "r0": [0, 1, 2, 4, 8, 16],
        "solver": ["group_lasso", "ls", "omp", "simplex"],
        "injection": ["affine", "additive", "projection"],
        "layer": [L0 - 4, L0 - 2, L0, L0 + 2, L0 + 4],
        "k": [1, 2, 4, 8],
        "lam": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
        "n": [25, 50, 100],
    }
    for value in grids[axis]:
        for tstar in REP_TASKS:
            for seed in SEEDS:
                kw = {}
                if axis == "r0":
                    kw["r0"] = value
                elif axis == "solver":
                    kw["solver"] = value
                elif axis == "injection":
                    kw["injection"] = value
                elif axis == "layer":
                    kw["layer"] = value
                elif axis == "k":
                    kw["k"] = value
                elif axis == "lam":
                    kw["lam"] = float(value)
                elif axis == "n":
                    kw["n_samples"] = value
                run_condition(axis, value, tstar, seed, **kw)
        print(f"[{axis}={value}] done ({(time.time()-t0)/60:.1f} min)",
              flush=True)


if __name__ == "__main__":
    axes = list(("r0 solver injection layer k lam n".split())
                if AXES == ["all"] else AXES)
    for ax in axes:
        sweep(ax)
    fout.close()
    print("E4 DONE")
