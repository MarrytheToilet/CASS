"""Extract contrastive activations for all tasks (all layers) and compute
ZS / 10-shot ICL baseline accuracies."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.evaluate import accuracy
from cass.extract import extract_and_save
from cass.models import HookedLM
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
N_PAIRS = 100
ICL_SEEDS = [0, 1, 2]

def main():
    t0 = time.time()
    hlm = HookedLM(MODEL)
    out_dir = results_dir(MODEL)
    baselines = {}
    bl_path = out_dir / "baselines.json"
    if bl_path.exists():
        baselines = json.load(open(bl_path))

    for i, tname in enumerate(ALL_TASKS):
        task = load_task(tname)
        extract_and_save(hlm, task, n_pairs=N_PAIRS, batch_size=6)
        if tname not in baselines:
            queries = task.eval_queries
            targets = [y for _, y in queries]
            zs = accuracy(hlm.generate([zs_prompt(x) for x, _ in queries],
                                       batch_size=16), targets)
            icls = []
            pool = task.dict_pool
            for seed in ICL_SEEDS:
                rng = np.random.default_rng(seed)
                prompts = [icl_prompt([pool[j] for j in
                                       rng.choice(len(pool), 10, replace=False)], x)
                           for x, _ in queries]
                icls.append(accuracy(hlm.generate(prompts, batch_size=8), targets))
            baselines[tname] = dict(zs=zs, icl_mean=float(np.mean(icls)),
                                    icl_all=icls, family=task.family)
            json.dump(baselines, open(bl_path, "w"), indent=2)
        b = baselines[tname]
        print(f"[{i+1}/{len(ALL_TASKS)}] {tname}: zs={b['zs']:.2f} "
              f"icl={b['icl_mean']:.2f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
