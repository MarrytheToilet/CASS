"""One-time coarse grid for injection hyperparameters (gamma, beta, alpha_max)
using ORACLE subspace injection on 3 base tasks; frozen globally afterwards."""
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.evaluate import accuracy
from cass.models import HookedLM
from cass.steer import make_affine_op
from cass.tasks import load_task, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
TUNE_TASKS = ["antonym", "country-capital", "english-french"]
GRID = dict(gamma=[0.5, 1.0, 2.0], beta=[2.0, 4.0, 8.0], alpha_max=[1.0, 2.0, 3.0])
N_EVAL = 25

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    layer = json.load(open(out / "layer_scan_summary.json"))["best_layer"]
    D = pickle.load(open(out / f"dictionary_l{layer}_r02.pkl", "rb"))
    hlm = HookedLM(MODEL)
    baselines = json.load(open(out / "baselines.json"))

    results = {}
    for g, b, am in itertools.product(GRID["gamma"], GRID["beta"],
                                      GRID["alpha_max"]):
        recs = []
        for tname in TUNE_TASKS:
            task = load_task(tname)
            queries = task.eval_queries[:N_EVAL]
            targets = [y for _, y in queries]
            prompts = [zs_prompt(x) for x, _ in queries]
            op = make_affine_op(D.anchors[tname], D.bases[tname],
                                D.anchors[tname], gamma=g, beta=b, alpha_max=am)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                        layer=layer), targets)
            bl = baselines[tname]
            gap = bl["icl_mean"] - bl["zs"]
            recs.append((acc - bl["zs"]) / gap if gap > 0.05 else acc)
        key = f"g{g}_b{b}_a{am}"
        results[key] = dict(mean_recovery=float(np.mean(recs)),
                            per_task=[float(r) for r in recs])
        print(f"{key}: {np.mean(recs):.3f} {recs}", flush=True)

    best = max(results, key=lambda k: results[k]["mean_recovery"])
    g, b, am = (float(x[1:]) for x in best.split("_"))
    summary = dict(results=results, best=best, gamma=g, beta=b, alpha_max=am,
                   layer=layer)
    json.dump(summary, open(out / "injection_hparams.json", "w"), indent=2)
    print(f"BEST {best} recovery={results[best]['mean_recovery']:.3f} "
          f"({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
