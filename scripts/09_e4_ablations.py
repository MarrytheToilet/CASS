"""E4 ablation matrix on 8 representative tasks (LOTO protocol, k=4 default),
multi-layer hybrid pipeline.

Axes:
  r0        : 0/1/2/4/8/16 (common-component rank)
  solver    : group_lasso / ls / omp / simplex (support + code)
  delta     : hybrid(z) / recon / blend / zonly-additive (direction source)
  layers    : [12] / [16] / [12,16] / [12,14,16]
  k         : 1 / 2 / 4 / 8
  smax      : 3 / 5 / 8 / 31 (support cap)
  n         : 25 / 50 / 100 (dictionary samples per task)
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
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z, _support_weights
from cass.solver import solve_capped
from cass.tasks import ALL_TASKS, load_task, zs_prompt
from cass.zcache import get_z

import os
MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
AXES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"])
REP_TASKS = (list(ALL_TASKS) if os.environ.get("CASS_REP") == "all" else
             ["antonym", "present-past", "country-capital", "person-sport",
              "english-french", "next-item", "choose-first-of-list",
              "animal-from-list"])
SEEDS = [0, 1, 2]
N_EVAL = 25
K_DEFAULT = 4

out = results_dir(MODEL)
hp = json.load(open(out / "injection_hparams.json"))
LAYERS, GAMMA, BETA, AMAX = (hp["layers"], hp["gamma"], hp["beta"],
                             hp["alpha_max"])

hlm = HookedLM(MODEL)
rows_path = out / ("e4_ablations_all32.csv" if os.environ.get("CASS_REP") == "all" else "e4_ablations.csv")
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


def run_condition(axis, value, tstar, seed, *, r0=1, solver="group_lasso",
                  delta_src="hybrid", layers=None, k=K_DEFAULT, s_max=5,
                  n_samples=None):
    if (axis, str(value), tstar, str(seed)) in done:
        return
    layers = layers or LAYERS
    task = load_task(tstar)
    D = build_multilayer_dictionary(
        {l: {t: (G_at(l)[t][:n_samples] if n_samples else G_at(l)[t])
             for t in ALL_TASKS if t != tstar} for l in layers}, r0=r0)
    Z = get_z(hlm, task, k, seed)
    z_list = z_list_from_Z(D, Z)
    z_mean = np.mean(z_list, axis=0)
    if solver == "group_lasso":
        code = solve_capped(D, z_list, s_max=s_max)
    else:
        code = code_for(D, z_list, solver=solver)

    if delta_src == "hybrid":
        ops, lys = ops_for(D, code, GAMMA, BETA, AMAX, delta_vec=z_mean)
    elif delta_src == "recon":
        ops, lys = ops_for(D, code, GAMMA, BETA, AMAX)
    elif delta_src == "blend":
        if code.support:
            w = _support_weights(code)
            blend = sum(wi * D.anchors[n] for wi, n in zip(w, code.support))
        else:
            blend = z_mean
        ops, lys = ops_for(D, code, GAMMA, BETA, AMAX, delta_vec=blend)
    elif delta_src == "zonly":
        ops, lys = ops_for(D, code, GAMMA, BETA, AMAX, injection="additive",
                           delta_vec=z_mean)
    elif delta_src == "projection":
        ops, lys = ops_for(D, code, GAMMA, BETA, AMAX, injection="projection")
    else:
        raise ValueError(delta_src)

    queries = task.eval_queries[:N_EVAL]
    acc = accuracy(hlm.generate([zs_prompt(x) for x, _ in queries],
                                batch_size=25, op=ops, layer=lys),
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
        "delta": ["hybrid", "recon", "blend", "zonly", "projection"],
        "layers": ["12", "16", "12+16", "12+14+16"],
        "k": [1, 2, 4, 8],
        "smax": [3, 5, 8, 31],
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
                elif axis == "delta":
                    kw["delta_src"] = value
                elif axis == "layers":
                    kw["layers"] = [int(x) for x in str(value).split("+")]
                elif axis == "k":
                    kw["k"] = value
                elif axis == "smax":
                    kw["s_max"] = value
                elif axis == "n":
                    kw["n_samples"] = value
                run_condition(axis, value, tstar, seed, **kw)
        print(f"[{axis}={value}] done ({(time.time()-t0)/60:.1f} min)",
              flush=True)


if __name__ == "__main__":
    axes = list(("r0 solver delta layers k smax n".split())
                if AXES == ["all"] else AXES)
    for ax in axes:
        sweep(ax)
    fout.close()
    print("E4 DONE")
