"""E2: compound tasks (multi-layer hybrid pipeline).

Compound tasks are NOT in the dictionary. Metrics: support precision/recall
against true components, coefficient matrix (heatmap), accuracy vs baselines
(ELICIT-style nearest-task retrieval, naive component-anchor average, ICL).
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
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, oracle_ops, z_list_from_Z
from cass.steer import make_additive_op
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt
from cass.zcache import get_z

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
K = 4
SEEDS = [0, 1, 2, 3, 4]
R0 = 2

def main():
    t0 = time.time()
    out = results_dir(MODEL)
    hp = json.load(open(out / "injection_hparams.json"))
    layers, gamma, beta, amax = (hp["layers"], hp["gamma"], hp["beta"],
                                 hp["alpha_max"])
    hlm = HookedLM(MODEL)
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in layers}
    D = build_multilayer_dictionary(G, r0=R0)

    rows = []
    coeff_matrix = {}
    for cname in COMPOUND_REGISTRY:
        comp = load_compound(cname)
        comps = compound_components(cname)
        queries = comp.eval_queries
        targets = [y for _, y in queries]
        prompts = [zs_prompt(x) for x, _ in queries]

        acc_zs = accuracy(hlm.generate(prompts, batch_size=25), targets)
        rng0 = np.random.default_rng(0)
        icl_prompts = [icl_prompt(
            [comp.dict_pool[j] for j in
             rng0.choice(len(comp.dict_pool), 10, replace=False)], x)
            for x, _ in queries]
        acc_icl = accuracy(hlm.generate(icl_prompts, batch_size=8), targets)

        coeff_acc = {t: [] for t in ALL_TASKS}
        for seed in SEEDS:
            Z = get_z(hlm, comp, K, seed)
            z_list = z_list_from_Z(D, Z)
            z_mean = np.mean(z_list, axis=0)
            code = code_for(D, z_list)
            for t in ALL_TASKS:
                coeff_acc[t].append(float(np.linalg.norm(code.coeffs[t])))

            hits = set(code.support) & set(comps)
            prec = len(hits) / len(code.support) if code.support else 0.0
            rec = len(hits) / len(comps)

            ops, lys = ops_for(D, code, gamma, beta, amax, delta_vec=z_mean)
            acc = accuracy(hlm.generate(prompts, batch_size=25, op=ops,
                                        layer=lys), targets)

            # ELICIT-style retrieval: nearest single dictionary task
            sims = {t: abs(float(z_mean @ D.anchors[t] /
                    (np.linalg.norm(z_mean) * np.linalg.norm(D.anchors[t])
                     + 1e-12))) for t in ALL_TASKS}
            nearest = max(sims, key=sims.get)
            ops_r, lys_r = oracle_ops(D, nearest, gamma, beta, amax)
            acc_retr = accuracy(hlm.generate(prompts, batch_size=25, op=ops_r,
                                             layer=lys_r), targets)

            # naive: average of the true components' anchors, additive
            ops_n = []
            for l in layers:
                Dl = D.per_layer[l]
                ops_n.append(make_additive_op(
                    np.mean([Dl.anchors[c] for c in comps], axis=0),
                    gamma=1.5))
            acc_naive = accuracy(hlm.generate(prompts, batch_size=25,
                                              op=ops_n, layer=list(layers)),
                                 targets)

            rows.append(dict(compound=cname, seed=seed, acc_zs=acc_zs,
                             acc_icl=acc_icl, acc_cass=acc,
                             acc_retrieval=acc_retr, retrieved=nearest,
                             acc_naive=acc_naive,
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
