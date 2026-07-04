"""Review-response experiments (8 representative tasks, LOTO protocol).

(a) denoise-vs-dictionary disentangle:
      z single-pair (m=1) additive | z denoised (m=6) additive | full CASS
(b) shared-component specificity control:
      r0=1 (top shared PC) vs removing a RANDOM unit direction (5 draws)
      vs r0=0
(c) hyperparameter robustness: perturb (gamma, beta, alpha_max) around
      (1,2,1); nearby layer pairs already covered by the E4 'layers' axis.
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.dictionary import build_dictionary, build_multilayer_dictionary, \
    MultiLayerDictionary
from cass.evaluate import accuracy
from cass.extract import load_G, extract_fewshot_z
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z
from cass.tasks import ALL_TASKS, load_task, zs_prompt

MODEL = "llama31-8b"
REP_TASKS = ["antonym", "present-past", "country-capital", "person-sport",
             "english-french", "next-item", "choose-first-of-list",
             "animal-from-list"]
SEEDS = [0, 1, 2]
K = 4
N_EVAL = 25
LAYERS = [12, 16]

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hlm = HookedLM(MODEL)
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in LAYERS}

    rows_path = out / "review_response.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["exp"], r["cond"], r["task"], r["seed"])
                    for r in csv.DictReader(f)}
    fout = open(rows_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=["exp", "cond", "task", "seed",
                                              "acc"])
    if not done:
        writer.writeheader()

    def emit(exp, cond, task, seed, acc):
        writer.writerow(dict(exp=exp, cond=cond, task=task, seed=seed,
                             acc=acc))
        fout.flush()

    def eval_op(task, ops, lys):
        q = task.eval_queries[:N_EVAL]
        return accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                     batch_size=25, op=ops, layer=lys),
                        [y for _, y in q])

    def z_for(task, seed, n_reps):
        rng = np.random.default_rng(100 * seed + K)
        idx = rng.choice(len(task.fewshot_pool), K, replace=False)
        examples = [task.fewshot_pool[i] for i in idx]
        return extract_fewshot_z(hlm, examples, seed=seed, n_reps=n_reps)

    # ---------- (a) denoise vs dictionary ----------
    for tstar in REP_TASKS:
        task = load_task(tstar)
        D = build_multilayer_dictionary(
            {l: {t: G[l][t] for t in ALL_TASKS if t != tstar}
             for l in LAYERS}, r0=1)
        for seed in SEEDS:
            conds = [c for c in ["z_m1_add", "z_m6_add", "cass_m1",
                                 "cass_m6"]
                     if ("denoise", c, tstar, str(seed)) not in done]
            if not conds:
                continue
            Z1 = z_for(task, seed, 1)
            Z6 = z_for(task, seed, 6)
            for cond, Z in [("z_m1_add", Z1), ("z_m6_add", Z6),
                            ("cass_m1", Z1), ("cass_m6", Z6)]:
                if cond not in conds:
                    continue
                zl = z_list_from_Z(D, Z)
                zm = np.mean(zl, axis=0)
                code = code_for(D, zl)
                if cond.endswith("_add"):
                    ops, lys = ops_for(D, code, 1.0, 2.0, 1.0,
                                       injection="additive", delta_vec=zm)
                else:
                    ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=zm)
                emit("denoise", cond, tstar, seed, eval_op(task, ops, lys))
        print(f"(a) {tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------- (b) shared-component specificity ----------
    def build_with_projection(exclude, U0_by_layer):
        per = {}
        for l in LAYERS:
            Dl = build_dictionary({t: G[l][t] for t in ALL_TASKS
                                   if t != exclude}, r0=0)
            U0 = U0_by_layer[l]
            for n in Dl.task_names:
                Gt = (G[l][n].astype(np.float64).T
                      - U0 @ (U0.T @ G[l][n].astype(np.float64).T))
                U, s, _ = np.linalg.svd(Gt, full_matrices=False)
                from cass.dictionary import rank_by_energy
                r = rank_by_energy(s, 0.90, 16)
                Dl.bases[n] = U[:, :r]
                Dl.anchors[n] = Gt.mean(1)
                Dl.U0 = U0
            per[l] = Dl
        return MultiLayerDictionary(per)

    rng0 = np.random.default_rng(7)
    for tstar in REP_TASKS:
        task = load_task(tstar)
        for draw in range(3):
            cond = f"random_dir_{draw}"
            if ("control", cond, tstar, "0") in done:
                continue
            U0r = {l: np.linalg.qr(rng0.standard_normal((hlm.d, 1)))[0]
                   for l in LAYERS}
            D = build_with_projection(tstar, U0r)
            Z = z_for(task, 0, 6)
            zl = z_list_from_Z(D, Z)
            zm = np.mean(zl, axis=0)
            code = code_for(D, zl)
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=zm)
            emit("control", cond, tstar, 0, eval_op(task, ops, lys))
        print(f"(b) {tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------- (c) hyperparameter robustness ----------
    HP = [(1.0, 2.0, 1.0), (0.75, 2.0, 1.0), (1.25, 2.0, 1.0),
          (1.0, 1.5, 1.0), (1.0, 3.0, 1.0), (1.0, 2.0, 0.75),
          (1.0, 2.0, 1.25)]
    for tstar in REP_TASKS:
        task = load_task(tstar)
        D = build_multilayer_dictionary(
            {l: {t: G[l][t] for t in ALL_TASKS if t != tstar}
             for l in LAYERS}, r0=1)
        for seed in SEEDS[:2]:
            Z = z_for(task, seed, 6)
            zl = z_list_from_Z(D, Z)
            zm = np.mean(zl, axis=0)
            code = code_for(D, zl)
            for g, b, am in HP:
                cond = f"g{g}_b{b}_a{am}"
                if ("hparam", cond, tstar, str(seed)) in done:
                    continue
                ops, lys = ops_for(D, code, g, b, am, delta_vec=zm)
                emit("hparam", cond, tstar, seed, eval_op(task, ops, lys))
        print(f"(c) {tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    fout.close()
    print(f"REVIEW-RESPONSE DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
