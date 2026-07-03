"""E5: dictionary-size scaling curve. Random sub-dictionaries of size T',
fixed 6 held-out tasks, recovery vs T'."""
import csv
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
from cass.pipeline import code_for, op_for
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
    layer, gamma, beta, amax = (hp["layer"], hp["gamma"], hp["beta"],
                                hp["alpha_max"])
    hlm = HookedLM(MODEL)
    G = {t: load_G(MODEL, t, layer).numpy() for t in ALL_TASKS}

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
                D = build_dictionary({t: G[t] for t in subset}, r0=2)
                for seed in SEEDS:
                    if (str(size), str(draw), tstar, str(seed)) in done:
                        continue
                    Z = get_z(hlm, task, K, seed)
                    z_list = [D.project_out_shared(
                        Z[j, layer].numpy().astype(np.float64))
                        for j in range(Z.shape[0])]
                    code = code_for(D, z_list)
                    op = op_for(D, code, gamma, beta, amax)
                    acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                                layer=layer), targets)
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
