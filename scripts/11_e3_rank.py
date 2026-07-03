"""E3: rank scan. Oracle affine injection with the task's own subspace
truncated to rank r in {1,2,4,8,16}. Slow-spectral-decay (high-rank) tasks
should need larger r; r=1 reproduces single-vector failures."""
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
from cass.steer import make_affine_op
from cass.tasks import ALL_TASKS, load_task, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
RANKS = [1, 2, 4, 8, 16]
N_EVAL = 50

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layer, gamma, beta, amax = (hp["layer"], hp["gamma"], hp["beta"],
                                hp["alpha_max"])
    hlm = HookedLM(MODEL)
    G = {t: load_G(MODEL, t, layer).numpy() for t in ALL_TASKS}
    # rank-16 dictionary with full spectra retained
    D = build_dictionary(G, r0=2, tau=1.0, r_max=16)

    rows = []
    for tname in ALL_TASKS:
        task = load_task(tname)
        queries = task.eval_queries[:N_EVAL]
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]
        s = D.spectra[tname]
        e = (s ** 2) / (s ** 2).sum()
        r90 = int(np.searchsorted(np.cumsum(e), 0.9) + 1)  # effective rank
        for r in RANKS:
            U = D.bases[tname][:, :r]
            mu = D.anchors[tname]
            mu_r = U @ (U.T @ mu)   # anchor restricted to the rank-r subspace
            op = make_affine_op(mu_r, U, mu_r, gamma=gamma, beta=beta,
                                alpha_max=amax)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                        layer=layer), targets)
            rows.append(dict(task=tname, family=task.family, rank=r, acc=acc,
                             effective_rank_90=r90,
                             top1_energy=round(float(e[0]), 4)))
        print(f"{tname}: r90={r90} " +
              " ".join(f"r{r}={rows[-len(RANKS)+i]['acc']:.2f}"
                       for i, r in enumerate(RANKS)) +
              f" ({(time.time()-t0)/60:.1f} min)", flush=True)
        with open(out / "e3_rank.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"E3 DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
