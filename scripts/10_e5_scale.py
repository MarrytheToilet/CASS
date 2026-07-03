"""E5: dictionary-size scaling (multi-layer hybrid). Random sub-dictionaries
of size T', fixed 6 held-out tasks, accuracy vs T'."""
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
from cass.pipeline import code_for, ops_for, z_list_from_Z
from cass.tasks import ALL_TASKS, load_task, zs_prompt
from cass.zcache import get_z

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
HELD_OUT = ["antonym", "country-capital", "english-french", "present-past",
            "next-item", "person-sport"]
SIZES = [5, 10, 15, 20, 25, 31]
DRAWS = [0, 1, 2, 3, 4]
SEEDS = [0, 1]
K = 4
N_EVAL = 25

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layers, gamma, beta, amax = (hp["layers"], hp["gamma"], hp["beta"],
                                 hp["alpha_max"])
    hlm = HookedLM(MODEL)
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in layers}

    rows_path = out / "e5_scale.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["size"], r["draw"], r["task"], r["seed"])
                    for r in csv.DictReader(f)}
    fieldnames = ["size", "draw", "task", "seed", "acc", "eps", "support_size"]
    fout = open(rows_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    for tstar in HELD_OUT:
        task = load_task(tstar)
        queries = task.eval_queries[:N_EVAL]
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]
        pool = [t for t in ALL_TASKS if t != tstar]
        for size in SIZES:
            for draw in DRAWS:
                rng = np.random.default_rng(10 * size + draw)
                subset = list(rng.choice(pool, min(size, len(pool)),
                                         replace=False))
                D = build_multilayer_dictionary(
                    {l: {t: G[l][t] for t in subset} for l in layers}, r0=2)
                for seed in SEEDS:
                    if (str(size), str(draw), tstar, str(seed)) in done:
                        continue
                    Z = get_z(hlm, task, K, seed)
                    z_list = z_list_from_Z(D, Z)
                    z_mean = np.mean(z_list, axis=0)
                    code = code_for(D, z_list)
                    ops, lys = ops_for(D, code, gamma, beta, amax,
                                       delta_vec=z_mean)
                    acc = accuracy(hlm.generate(prompts, batch_size=25,
                                                op=ops, layer=lys), targets)
                    writer.writerow(dict(size=size, draw=draw, task=tstar,
                                         seed=seed, acc=acc,
                                         eps=round(code.residual, 4),
                                         support_size=len(code.support)))
                    fout.flush()
        print(f"{tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()
    print(f"E5 DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
