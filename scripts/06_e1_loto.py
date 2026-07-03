"""E1: Leave-one-task-out synthesis (go/no-go experiment).

For each held-out task t*: rebuild dictionary WITHOUT t* (incl. U0), extract z
from k examples, solve group LASSO, inject affine operator, evaluate.
Also records oracle (own-subspace injection) and naive (mean of all other
anchors) per task.
"""
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
from cass.extract import load_G, extract_fewshot_z
from cass.models import HookedLM
from cass.solver import solve
from cass.steer import make_additive_op, make_affine_op
from cass.tasks import ALL_TASKS, load_task, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
KS = [1, 2, 4]
SEEDS = [0, 1, 2, 3, 4]
R0 = 2
RESCALE = True   # scale synthesized delta to weighted support-anchor norm

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layer, gamma, beta, amax = (hp["layer"], hp["gamma"], hp["beta"],
                                hp["alpha_max"])
    baselines = json.load(open(out / "baselines.json"))
    hlm = HookedLM(MODEL)
    G_all = {t: load_G(MODEL, t, layer).numpy() for t in ALL_TASKS}

    rows_path = out / "e1_loto.csv"
    done = set()
    if rows_path.exists():
        with open(rows_path) as f:
            done = {(r["task"], r["k"], r["seed"], r["mode"])
                    for r in csv.DictReader(f)}
    fieldnames = ["task", "family", "k", "seed", "mode", "acc", "eps",
                  "support_size", "support", "lam", "delta_norm"]
    fout = open(rows_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    def emit(**kw):
        writer.writerow(kw)
        fout.flush()

    for ti, tstar in enumerate(ALL_TASKS):
        task = load_task(tstar)
        queries = task.eval_queries
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]

        D_full = build_dictionary(G_all, r0=R0)
        D = build_dictionary({t: G_all[t] for t in ALL_TASKS if t != tstar},
                             r0=R0)

        # oracle: own-subspace affine injection (full dictionary's U0)
        if (tstar, "0", "0", "oracle") not in done:
            op = make_affine_op(D_full.anchors[tstar], D_full.bases[tstar],
                                D_full.anchors[tstar], gamma=gamma, beta=beta,
                                alpha_max=amax)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                        layer=layer), targets)
            emit(task=tstar, family=task.family, k=0, seed=0, mode="oracle",
                 acc=acc, eps=0, support_size=1, support=tstar, lam=0,
                 delta_norm=float(np.linalg.norm(D_full.anchors[tstar])))

        # naive: mean of the other tasks' anchors, additive
        if (tstar, "0", "0", "naive") not in done:
            delta = np.mean([D.anchors[t] for t in D.task_names], axis=0)
            op = make_additive_op(delta, gamma=2.0)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                        layer=layer), targets)
            emit(task=tstar, family=task.family, k=0, seed=0, mode="naive",
                 acc=acc, eps=1, support_size=len(D.task_names), support="all",
                 lam=0, delta_norm=float(np.linalg.norm(delta)))

        for k in KS:
            for seed in SEEDS:
                if (tstar, str(k), str(seed), "cass") in done:
                    continue
                rng = np.random.default_rng(100 * seed + k)
                idx = rng.choice(len(task.fewshot_pool), k, replace=False)
                examples = [task.fewshot_pool[i] for i in idx]
                Z = extract_fewshot_z(hlm, examples, seed=seed)  # [k, L+1, d]
                z_list = [D.project_out_shared(Z[j, layer].numpy().astype(
                    np.float64)) for j in range(len(examples))]
                code = solve(D, z_list)
                delta = code.delta
                dn = np.linalg.norm(delta)
                if RESCALE and code.support and dn > 1e-8:
                    w = np.array([np.linalg.norm(code.coeffs[n])
                                  for n in code.support])
                    w = w / w.sum()
                    target_norm = float(sum(
                        wi * np.linalg.norm(D.anchors[n])
                        for wi, n in zip(w, code.support)))
                    delta = delta * (target_norm / dn)
                if code.support:
                    w = np.array([np.linalg.norm(code.coeffs[n])
                                  for n in code.support])
                    w = w / w.sum()
                    mu_S = sum(wi * D.anchors[n]
                               for wi, n in zip(w, code.support))
                    op = make_affine_op(delta, code, mu_S, gamma=gamma,
                                        beta=beta, alpha_max=amax,
                                        dictionary=D)
                else:
                    op = make_additive_op(delta, gamma=gamma)
                acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                            layer=layer), targets)
                emit(task=tstar, family=task.family, k=k, seed=seed,
                     mode="cass", acc=acc, eps=round(code.residual, 4),
                     support_size=len(code.support),
                     support="|".join(code.support[:8]),
                     lam=round(code.lam, 4),
                     delta_norm=round(float(np.linalg.norm(delta)), 2))
        bl = baselines[tstar]
        print(f"[{ti+1}/{len(ALL_TASKS)}] {tstar} done "
              f"(zs={bl['zs']:.2f} icl={bl['icl_mean']:.2f}) "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()
    print(f"E1 DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
