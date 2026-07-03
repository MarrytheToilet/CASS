"""Cross-model setup: pick the injection layer pair for a new model by oracle
affine recovery on 6 tune tasks, then freeze injection_hparams.json.
Candidates are depth-proportional to the main model's {12,16}/32."""
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
from cass.steer import make_affine_op
from cass.tasks import ALL_TASKS, load_task, zs_prompt

MODEL = sys.argv[1]
TUNE_TASKS = ["antonym", "present-past", "country-capital", "english-french",
              "next-item", "choose-first-of-list"]
N_EVAL = 25

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    baselines = json.load(open(out / "baselines.json"))
    hlm = HookedLM(MODEL)
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
    for pair in pairs:
        recs = []
        for tname in TUNE_TASKS:
            task = load_task(tname)
            queries = task.eval_queries[:N_EVAL]
            targets = [y for _, y in queries]
            prompts = [zs_prompt(x) for x, _ in queries]
            ops = [make_affine_op(D[l].anchors[tname], D[l].bases[tname],
                                  D[l].anchors[tname], gamma=1.0, beta=2.0,
                                  alpha_max=1.0) for l in pair]
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=ops,
                                        layer=list(pair)), targets)
            bl = baselines[tname]
            gap = bl["icl_mean"] - bl["zs"]
            recs.append((acc - bl["zs"]) / gap if gap > 0.05 else np.nan)
        results[str(pair)] = float(np.nanmean(recs))
        print(f"{pair}: mean oracle recovery {results[str(pair)]:.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    best = max(results, key=results.get)
    best_pair = [int(x) for x in best.strip("()").split(",")]
    hp = dict(layers=best_pair, gamma=1.0, beta=2.0, alpha_max=1.0,
              scheme="multilayer_affine",
              source=f"05c cross-model setup: recovery {results[best]:.3f}",
              all_pairs=results)
    json.dump(hp, open(out / "injection_hparams.json", "w"), indent=2)
    print(f"BEST {best} -> injection_hparams.json")

if __name__ == "__main__":
    main()
