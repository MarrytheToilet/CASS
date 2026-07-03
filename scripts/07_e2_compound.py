"""E2: compound tasks -- can group-sparse coding recover the constituent
skills, and does the synthesized operator steer the compound behavior?

Compound tasks are NOT in the dictionary. Metrics: support precision/recall
against the true components, coefficient heatmap data, accuracy vs baselines
(ELICIT-style retrieval of the single nearest task; naive 2-anchor average).
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.compound import COMPOUND_REGISTRY, compound_components, load_compound
from cass.dictionary import build_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G, extract_fewshot_z
from cass.models import HookedLM
from cass.solver import solve
from cass.steer import make_additive_op, make_affine_op
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
K = 4
SEEDS = [0, 1, 2, 3, 4]
R0 = 2

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layer, gamma, beta, amax = (hp["layer"], hp["gamma"], hp["beta"],
                                hp["alpha_max"])
    hlm = HookedLM(MODEL)
    G_all = {t: load_G(MODEL, t, layer).numpy() for t in ALL_TASKS}
    D = build_dictionary(G_all, r0=R0)

    rows = []
    coeff_matrix = {}   # compound -> {task: mean ||c_t||}
    for cname in COMPOUND_REGISTRY:
        comp = load_compound(cname)
        comps = compound_components(cname)
        queries = comp.eval_queries
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]

        # baselines: zs, 10-shot icl
        acc_zs = accuracy(hlm.generate(prompts, batch_size=25), targets)
        rng0 = np.random.default_rng(0)
        icl_prompts = [icl_prompt(
            [comp.dict_pool[j] for j in
             rng0.choice(len(comp.dict_pool), 10, replace=False)], x)
            for x, _ in queries]
        acc_icl = accuracy(hlm.generate(icl_prompts, batch_size=8), targets)

        coeff_acc = {t: [] for t in ALL_TASKS}
        for seed in SEEDS:
            rng = np.random.default_rng(200 + seed)
            idx = rng.choice(len(comp.fewshot_pool), K, replace=False)
            examples = [comp.fewshot_pool[i] for i in idx]
            Z = extract_fewshot_z(hlm, examples, seed=seed)
            z_list = [D.project_out_shared(Z[j, layer].numpy().astype(
                np.float64)) for j in range(len(examples))]
            code = solve(D, z_list)
            for t in ALL_TASKS:
                coeff_acc[t].append(float(np.linalg.norm(code.coeffs[t])))

            hits = set(code.support) & set(comps)
            prec = len(hits) / len(code.support) if code.support else 0.0
            rec = len(hits) / len(comps)

            delta = code.delta
            dn = np.linalg.norm(delta)
            if code.support and dn > 1e-8:
                w = np.array([np.linalg.norm(code.coeffs[n])
                              for n in code.support])
                w = w / w.sum()
                tnorm = float(sum(wi * np.linalg.norm(D.anchors[n])
                                  for wi, n in zip(w, code.support)))
                delta = delta * (tnorm / dn)
                mu_S = sum(wi * D.anchors[n]
                           for wi, n in zip(w, code.support))
                op = make_affine_op(delta, code, mu_S, gamma=gamma, beta=beta,
                                    alpha_max=amax, dictionary=D)
            else:
                op = make_additive_op(delta, gamma=gamma)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=op,
                                        layer=layer), targets)

            # ELICIT-style retrieval baseline: single nearest dictionary task
            z = np.mean(z_list, axis=0)
            sims = {t: abs(float(np.dot(z, D.anchors[t]) /
                     (np.linalg.norm(z) * np.linalg.norm(D.anchors[t]) + 1e-12)))
                    for t in ALL_TASKS}
            nearest = max(sims, key=sims.get)
            op_r = make_affine_op(D.anchors[nearest], D.bases[nearest],
                                  D.anchors[nearest], gamma=gamma, beta=beta,
                                  alpha_max=amax)
            acc_retr = accuracy(hlm.generate(prompts, batch_size=25, op=op_r,
                                             layer=layer), targets)

            # naive: average of the two true components' anchors
            delta_n = np.mean([D.anchors[c] for c in comps], axis=0)
            acc_naive = accuracy(hlm.generate(
                prompts, batch_size=25, op=make_additive_op(delta_n, gamma=2.0),
                layer=layer), targets)

            rows.append(dict(compound=cname, seed=seed, acc_zs=acc_zs,
                             acc_icl=acc_icl, acc_cass=acc, acc_retrieval=acc_retr,
                             retrieved=nearest, acc_naive=acc_naive,
                             support="|".join(code.support[:8]),
                             support_precision=round(prec, 3),
                             support_recall=round(rec, 3),
                             eps=round(code.residual, 4)))
            print(rows[-1], flush=True)
        coeff_matrix[cname] = {t: float(np.mean(v))
                               for t, v in coeff_acc.items()}
        with open(out / "e2_compound.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        json.dump(coeff_matrix, open(out / "e2_coeff_matrix.json", "w"),
                  indent=2)
    print(f"E2 DONE in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
