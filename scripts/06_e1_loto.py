"""E1: Leave-one-task-out synthesis (go/no-go experiment), multi-layer version.

For each held-out task t*: rebuild the multi-layer dictionary WITHOUT t*
(incl. per-layer U0), extract z from k examples, solve joint group LASSO
(support shared across layers), inject per-layer affine operators, evaluate.
Also records oracle (own-subspace, both layers) and naive (anchor average).
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
from cass.evaluate import accuracy, dump_preds
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, oracle_ops, naive_ops, z_list_from_Z
from cass.tasks import ALL_TASKS, load_task, zs_prompt
from cass.zcache import get_z

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
KS = [1, 2, 4]
SEEDS = [0, 1, 2, 3, 4]
R0 = 2

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layers, gamma, beta, amax = (hp["layers"], hp["gamma"], hp["beta"],
                                 hp["alpha_max"])
    baselines = json.load(open(out / "baselines.json"))
    hlm = HookedLM(MODEL)
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in layers}

    rows_path = out / "e1_loto.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["task"], r["k"], r["seed"], r["mode"])
                    for r in csv.DictReader(f)}
    fieldnames = ["task", "family", "k", "seed", "mode", "acc", "eps",
                  "support_size", "support", "lam", "delta_norm"]
    fout = open(rows_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    def emit(**kw):
        writer.writerow(kw)
        fout.flush()

    D_full = build_multilayer_dictionary(G, r0=R0)
    for ti, tstar in enumerate(ALL_TASKS):
        task = load_task(tstar)
        queries = task.eval_queries
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]
        inputs = [x for x, _ in queries]
        gens_path = out / "e1_gens.jsonl"

        D = build_multilayer_dictionary(
            {l: {t: G[l][t] for t in ALL_TASKS if t != tstar} for l in layers},
            r0=R0)

        if (tstar, "0", "0", "oracle") not in done:
            ops, lys = oracle_ops(D_full, tstar, gamma, beta, amax)
            preds = hlm.generate(prompts, batch_size=25, op=ops, layer=lys)
            dump_preds(gens_path, f"{tstar}|oracle", inputs, preds, targets)
            acc = accuracy(preds, targets)
            emit(task=tstar, family=task.family, k=0, seed=0, mode="oracle",
                 acc=acc, eps=0, support_size=1, support=tstar, lam=0,
                 delta_norm=float(np.linalg.norm(D_full.anchors[tstar])))

        if (tstar, "0", "0", "naive") not in done:
            ops, lys = naive_ops(D)
            preds = hlm.generate(prompts, batch_size=25, op=ops, layer=lys)
            dump_preds(gens_path, f"{tstar}|naive", inputs, preds, targets)
            acc = accuracy(preds, targets)
            emit(task=tstar, family=task.family, k=0, seed=0, mode="naive",
                 acc=acc, eps=1, support_size=len(D.task_names), support="all",
                 lam=0, delta_norm=0)

        for k in KS:
            for seed in SEEDS:
                Z = None
                # cass = hybrid (z direction + dictionary support/affine);
                # cass_recon = pure dictionary reconstruction (analysis);
                # zvec = raw z additive, no dictionary (few-shot TV baseline)
                modes = ["cass"] + (["cass_recon", "zvec"] if seed < 3 else [])
                for mode in modes:
                    if (tstar, str(k), str(seed), mode) in done:
                        continue
                    if Z is None:
                        Z = get_z(hlm, task, k, seed)
                        z_list = z_list_from_Z(D, Z)
                        z_mean = np.mean(z_list, axis=0)
                        code = code_for(D, z_list)
                    if mode == "cass":
                        ops, lys = ops_for(D, code, gamma, beta, amax,
                                           delta_vec=z_mean)
                    elif mode == "cass_recon":
                        ops, lys = ops_for(D, code, gamma, beta, amax)
                    else:  # zvec: additive raw z at same layers, same rescale
                        ops, lys = ops_for(D, code, gamma, beta, amax,
                                           injection="additive",
                                           delta_vec=z_mean)
                    preds = hlm.generate(prompts, batch_size=25, op=ops,
                                         layer=lys)
                    dump_preds(gens_path, f"{tstar}|{mode}|k{k}s{seed}",
                               inputs, preds, targets)
                    acc = accuracy(preds, targets)
                    emit(task=tstar, family=task.family, k=k, seed=seed,
                         mode=mode, acc=acc, eps=round(code.residual, 4),
                         support_size=len(code.support),
                         support="|".join(code.support[:8]),
                         lam=round(code.lam, 4),
                         delta_norm=round(float(np.linalg.norm(z_mean)), 2))
        bl = baselines[tstar]
        print(f"[{ti+1}/{len(ALL_TASKS)}] {tstar} done "
              f"(zs={bl['zs']:.2f} icl={bl['icl_mean']:.2f}) "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()
    print(f"E1 DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
