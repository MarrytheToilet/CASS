"""Injection repair study: find an injection scheme whose ORACLE recovery is
high across families with NO per-task tuning. Candidates:
  - additive gamma=2 at single layers (12/14/16)
  - multi-layer additive at {12,16} and {10,13,16}
  - affine at single layers, several (beta, alpha_max)
  - multi-layer affine at {12,16}
Evaluated on 8 tasks x 25 queries against ICL gap.
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

MODEL = "llama31-8b"
TASKS = ["antonym", "present-past", "country-capital", "person-sport",
         "english-french", "next-item", "choose-first-of-list", "capitalize"]
N_EVAL = 25

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    baselines = json.load(open(out / "baselines.json"))
    hlm = HookedLM(MODEL)

    layers_needed = [10, 12, 13, 14, 16]
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in layers_needed}
    D = {l: build_dictionary(G[l], r0=2) for l in layers_needed}

    def additive(t, l, g):
        return make_additive_op(D[l].raw_means[t], gamma=g)

    def affine(t, l, g, b, am):
        return make_affine_op(D[l].anchors[t], D[l].bases[t], D[l].anchors[t],
                              gamma=g, beta=b, alpha_max=am)

    schemes = {
        "add_l12_g2": lambda t: (additive(t, 12, 2.0), 12),
        "add_l14_g2": lambda t: (additive(t, 14, 2.0), 14),
        "add_l16_g2": lambda t: (additive(t, 16, 2.0), 16),
        "add_l12+16_g1.5": lambda t: ([additive(t, 12, 1.5),
                                       additive(t, 16, 1.5)], [12, 16]),
        "add_l10+13+16_g1.2": lambda t: ([additive(t, l, 1.2)
                                          for l in (10, 13, 16)], [10, 13, 16]),
        "aff_l14_b2_a1": lambda t: (affine(t, 14, 1.0, 2.0, 1.0), 14),
        "aff_l16_b2_a1": lambda t: (affine(t, 16, 1.0, 2.0, 1.0), 16),
        "aff_l12+16_b2_a1": lambda t: ([affine(t, 12, 1.0, 2.0, 1.0),
                                        affine(t, 16, 1.0, 2.0, 1.0)], [12, 16]),
        "aff_l12+16_b4_a2": lambda t: ([affine(t, 12, 1.0, 4.0, 2.0),
                                        affine(t, 16, 1.0, 4.0, 2.0)], [12, 16]),
        "aff_l12+16_g0.5_b2_a1": lambda t: ([affine(t, 12, 0.5, 2.0, 1.0),
                                             affine(t, 16, 0.5, 2.0, 1.0)],
                                            [12, 16]),
    }

    results = {}
    for name, make in schemes.items():
        recs, accs = [], {}
        for tname in TASKS:
            task = load_task(tname)
            queries = task.eval_queries[:N_EVAL]
            targets = [y for _, y in queries]
            prompts = [zs_prompt(x) for x, _ in queries]
            op, layer = make(tname)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                        layer=layer), targets)
            bl = baselines[tname]
            gap = bl["icl_mean"] - bl["zs"]
            rec = (acc - bl["zs"]) / gap if gap > 0.05 else np.nan
            recs.append(rec)
            accs[tname] = round(acc, 3)
        mean_rec = float(np.nanmean(recs))
        results[name] = dict(mean_recovery=mean_rec, accs=accs,
                             recs=[round(float(r), 3) for r in recs])
        print(f"{name}: mean_rec={mean_rec:.3f} accs={accs} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    best = max(results, key=lambda k: results[k]["mean_recovery"])
    json.dump(dict(results=results, best=best),
              open(out / "injection_repair.json", "w"), indent=2)
    print(f"\nBEST: {best} -> {results[best]['mean_recovery']:.3f}")

if __name__ == "__main__":
    main()
