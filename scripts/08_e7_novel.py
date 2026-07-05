"""E7: truly out-of-distribution unseen tasks (API-synthesized, not from the
Todd benchmark). Dictionary = the 32 original skills; each synthetic task is
evaluated as unseen (cass hybrid / zvec / oracle / naive), k=4, 5 seeds.
Phase 2: extended scale curve with the combined 47-skill pool.
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.dictionary import build_multilayer_dictionary, build_dictionary
from cass.evaluate import accuracy, dump_preds
from cass.extract import extract_and_save, load_G
from cass.models import HookedLM
from cass.pipeline import (code_for, ops_for, oracle_ops, naive_ops,
                           z_list_from_Z)
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt, \
    synthetic_tasks
from cass.zcache import get_z

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
SEEDS = [0, 1, 2, 3, 4]
K = 4
R0 = 1

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layers, gamma, beta, amax = (hp["layers"], hp["gamma"], hp["beta"],
                                 hp["alpha_max"])
    hlm = HookedLM(MODEL)
    synth = list(synthetic_tasks())

    # extraction + baselines for synthetic tasks
    baselines = {}
    bl_path = out / "baselines_synth.json"
    if bl_path.exists():
        baselines = json.load(open(bl_path))
    for name in synth:
        task = load_task(name)
        extract_and_save(hlm, task, n_pairs=100, batch_size=6)
        if name not in baselines:
            queries = task.eval_queries
            targets = [y for _, y in queries]
            zs = accuracy(hlm.generate([zs_prompt(x) for x, _ in queries],
                                       batch_size=16), targets)
            rng = np.random.default_rng(0)
            icl = accuracy(hlm.generate(
                [icl_prompt([task.dict_pool[j] for j in
                             rng.choice(len(task.dict_pool), 10,
                                        replace=False)], x)
                 for x, _ in queries], batch_size=8), targets)
            baselines[name] = dict(zs=zs, icl_mean=icl)
            json.dump(baselines, open(bl_path, "w"), indent=2)
        print(f"extracted {name} (zs={baselines[name]['zs']:.2f} "
              f"icl={baselines[name]['icl_mean']:.2f}) "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # dictionary: original 32 tasks only
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in layers}
    D = build_multilayer_dictionary(G, r0=R0)

    rows_path = out / "e7_novel.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["task"], r["seed"], r["mode"])
                    for r in csv.DictReader(f)}
    fieldnames = ["task", "family", "seed", "mode", "acc", "eps",
                  "support_size", "support"]
    fout = open(rows_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not done:
        writer.writeheader()
    gens_path = out / "e7_gens.jsonl"

    for name in synth:
        task = load_task(name)
        queries = task.eval_queries
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]
        inputs = [x for x, _ in queries]

        # oracle: own subspace from its extraction, projected by D's U0
        if (name, "0", "oracle") not in done:
            Do = build_multilayer_dictionary(
                {l: {**G[l], name: load_G(MODEL, name, l).numpy()}
                 for l in layers}, r0=R0)
            ops, lys = oracle_ops(Do, name, gamma, beta, amax)
            preds = hlm.generate(prompts, batch_size=25, op=ops, layer=lys)
            dump_preds(gens_path, f"{name}|oracle", inputs, preds, targets)
            writer.writerow(dict(task=name, family=task.family, seed=0,
                                 mode="oracle",
                                 acc=accuracy(preds, targets), eps=0,
                                 support_size=1, support=name))
            fout.flush()
        if (name, "0", "naive") not in done:
            ops, lys = naive_ops(D)
            preds = hlm.generate(prompts, batch_size=25, op=ops, layer=lys)
            writer.writerow(dict(task=name, family=task.family, seed=0,
                                 mode="naive",
                                 acc=accuracy(preds, targets), eps=1,
                                 support_size=0, support="all"))
            fout.flush()

        for seed in SEEDS:
            Z = None
            for mode in (["cass", "zvec"] if seed < 3 else ["cass"]):
                if (name, str(seed), mode) in done:
                    continue
                if Z is None:
                    Z = get_z(hlm, task, K, seed)
                    z_list = z_list_from_Z(D, Z)
                    z_mean = np.mean(z_list, axis=0)
                    code = code_for(D, z_list)
                if mode == "cass":
                    ops, lys = ops_for(D, code, gamma, beta, amax,
                                       delta_vec=z_mean)
                else:
                    ops, lys = ops_for(D, code, gamma, beta, amax,
                                       injection="additive",
                                       delta_vec=z_mean)
                preds = hlm.generate(prompts, batch_size=25, op=ops,
                                     layer=lys)
                dump_preds(gens_path, f"{name}|{mode}|s{seed}", inputs,
                           preds, targets)
                writer.writerow(dict(task=name, family=task.family,
                                     seed=seed, mode=mode,
                                     acc=accuracy(preds, targets),
                                     eps=round(code.residual, 4),
                                     support_size=len(code.support),
                                     support="|".join(code.support[:8])))
                fout.flush()
        print(f"{name} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()

    # phase 2: extended scale curve (47-skill pool, original 6 held-out)
    pool_all = ALL_TASKS + synth
    Gx = {l: {t: load_G(MODEL, t, l).numpy() for t in pool_all}
          for l in layers}
    rows2 = out / "e5_scale_ext.csv"
    done2 = set()
    if rows2.exists():
        with open(rows2) as f:
            done2 = {(r["size"], r["draw"], r["task"], r["seed"])
                     for r in csv.DictReader(f)}
    f2 = open(rows2, "a", newline="")
    w2 = csv.DictWriter(f2, fieldnames=["size", "draw", "task", "seed",
                                        "acc", "eps", "support_size"])
    if not done2:
        w2.writeheader()
    HELD = ["antonym", "country-capital", "english-french", "present-past",
            "next-item", "person-sport"]
    for tstar in HELD:
        task = load_task(tstar)
        queries = task.eval_queries[:25]
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]
        cand = [t for t in pool_all if t != tstar]
        for size in [5, 10, 20, 30, 40, 46]:
            for draw in range(5):
                rng = np.random.default_rng(100 * size + draw)
                subset = list(rng.choice(cand, min(size, len(cand)),
                                         replace=False))
                Ds = build_multilayer_dictionary(
                    {l: {t: Gx[l][t] for t in subset} for l in layers},
                    r0=R0)
                for seed in [0, 1]:
                    if (str(size), str(draw), tstar, str(seed)) in done2:
                        continue
                    Z = get_z(hlm, task, K, seed)
                    z_list = z_list_from_Z(Ds, Z)
                    z_mean = np.mean(z_list, axis=0)
                    code = code_for(Ds, z_list)
                    ops, lys = ops_for(Ds, code, gamma, beta, amax,
                                       delta_vec=z_mean)
                    acc = accuracy(hlm.generate(prompts, batch_size=25,
                                                op=ops, layer=lys), targets)
                    w2.writerow(dict(size=size, draw=draw, task=tstar,
                                     seed=seed, acc=acc,
                                     eps=round(code.residual, 4),
                                     support_size=len(code.support)))
                    f2.flush()
        print(f"scale-ext {tstar} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    f2.close()
    print(f"E7 DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
