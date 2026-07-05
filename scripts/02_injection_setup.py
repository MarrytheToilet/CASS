"""Injection setup for a model, two stages:
  scan : add the oracle mean diff vector to zero-shot prompts at each
         candidate layer x gamma; the steerability check of RQ5 (a family
         whose best recovery stays near zero is contrastively unsteerable).
  pair : pick the affine injection layer pair by oracle recovery on 6 tune
         tasks (candidates depth-proportional to the main model's {12,16}/32)
         and freeze injection_hparams.json.
Usage: python 02_injection_setup.py [model] [scan|pair|all]
"""
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
from cass.steer import make_additive_op, make_affine_op
from cass.tasks import ALL_TASKS, load_task, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"
SCAN_TASKS = ["antonym", "present-past", "singular-plural", "capitalize",
              "country-capital", "person-sport", "product-company",
              "english-french", "english-spanish",
              "next-item", "choose-first-of-list", "alphabetically-first"]
TUNE_TASKS = ["antonym", "present-past", "country-capital", "english-french",
              "next-item", "choose-first-of-list"]
GAMMAS = [1.0, 2.0, 3.0]
N_EVAL = 25


def scan(hlm, baselines):
    t0 = time.time()
    layers = list(range(6, min(24, hlm.L - 4) + 1, 2))
    results = {}  # task -> {"l,g": acc}
    for tname in SCAN_TASKS:
        task = load_task(tname)
        queries = task.eval_queries[:N_EVAL]
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]
        G = load_G(MODEL, tname)
        results[tname] = {}
        for l in layers:
            mu = G[:, l, :].mean(0).numpy()
            for g in GAMMAS:
                op = make_additive_op(mu, gamma=g)
                acc = accuracy(hlm.generate(prompts, batch_size=25,
                                            max_new_tokens=6, op=op, layer=l),
                               targets)
                results[tname][f"{l},{g}"] = acc
        best = max(results[tname], key=results[tname].get)
        print(f"{tname}: best={best} acc={results[tname][best]:.2f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        json.dump(results, open(results_dir(MODEL) / "layer_scan.json", "w"),
                  indent=2)

    # aggregate mean recovery per (layer, gamma)
    agg = {}
    for l in layers:
        for g in GAMMAS:
            recs = []
            for tname in SCAN_TASKS:
                b = baselines[tname]
                gap = b["icl_mean"] - b["zs"]
                if gap > 0.05:
                    recs.append((results[tname][f"{l},{g}"] - b["zs"]) / gap)
            agg[f"{l},{g}"] = float(np.mean(recs))
    best = max(agg, key=agg.get)
    summary = dict(agg=agg, best=best, best_layer=int(best.split(",")[0]),
                   best_gamma=float(best.split(",")[1]))
    json.dump(summary,
              open(results_dir(MODEL) / "layer_scan_summary.json", "w"),
              indent=2)
    print(f"BEST (layer,gamma)={best} mean recovery={agg[best]:.3f}")


def pair(hlm, baselines):
    t0 = time.time()
    out = results_dir(MODEL)
    L = hlm.L
    lo, hi = round(12 / 32 * L), round(16 / 32 * L)
    pairs = [(lo, hi), (lo - 2, hi), (lo, hi + 2), (lo - 1, hi + 1),
             (lo + 1, hi - 1)]
    pairs = sorted({(max(2, a), min(L - 2, b)) for a, b in pairs})
    layers_needed = sorted({l for p in pairs for l in p})
    D = {l: build_dictionary({t: load_G(MODEL, t, l).numpy()
                              for t in ALL_TASKS}, r0=1)
         for l in layers_needed}

    results = {}
    for p in pairs:
        recs = []
        for tname in TUNE_TASKS:
            task = load_task(tname)
            queries = task.eval_queries[:N_EVAL]
            targets = [y for _, y in queries]
            prompts = [zs_prompt(x) for x, _ in queries]
            ops = [make_affine_op(D[l].anchors[tname], D[l].bases[tname],
                                  D[l].anchors[tname], gamma=1.0, beta=2.0,
                                  alpha_max=1.0) for l in p]
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=ops,
                                        layer=list(p)), targets)
            bl = baselines[tname]
            gap = bl["icl_mean"] - bl["zs"]
            recs.append((acc - bl["zs"]) / gap if gap > 0.05 else np.nan)
        results[str(p)] = float(np.nanmean(recs))
        print(f"{p}: mean oracle recovery {results[str(p)]:.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    best = max(results, key=results.get)
    best_pair = [int(x) for x in best.strip("()").split(",")]
    hp = dict(layers=best_pair, gamma=1.0, beta=2.0, alpha_max=1.0,
              scheme="multilayer_affine",
              source=f"02 injection setup: recovery {results[best]:.3f}",
              all_pairs=results)
    json.dump(hp, open(out / "injection_hparams.json", "w"), indent=2)
    print(f"BEST {best} -> injection_hparams.json")


if __name__ == "__main__":
    hlm = HookedLM(MODEL)
    baselines = json.load(open(results_dir(MODEL) / "baselines.json"))
    if STAGE in ("scan", "all"):
        scan(hlm, baselines)
    if STAGE in ("pair", "all"):
        pair(hlm, baselines)
    print("DONE")
