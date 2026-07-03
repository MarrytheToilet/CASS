"""Injection layer scan: add oracle mean diff vector (per task) to zero-shot
prompts at each candidate layer x gamma; pick l* maximizing mean recovery."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.steer import make_additive_op
from cass.tasks import load_task, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
SCAN_TASKS = ["antonym", "present-past", "singular-plural", "capitalize",
              "country-capital", "person-sport", "product-company",
              "english-french", "english-spanish",
              "next-item", "choose-first-of-list", "alphabetically-first"]
GAMMAS = [1.0, 2.0, 3.0]
N_EVAL = 25

def main():
    t0 = time.time()
    hlm = HookedLM(MODEL)
    layers = list(range(6, min(24, hlm.L - 4) + 1, 2))
    baselines = json.load(open(results_dir(MODEL) / "baselines.json"))
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
    json.dump(summary, open(results_dir(MODEL) / "layer_scan_summary.json", "w"),
              indent=2)
    print(f"BEST (layer,gamma)={best} mean recovery={agg[best]:.3f}")
    print(f"DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
