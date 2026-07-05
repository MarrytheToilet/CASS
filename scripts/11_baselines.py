"""Baselines (same k=4 examples as CASS), four stages:
  lit       : LOTO-32 + Novel-15 -> baselines_lit.csv
      hendel_replace : task vector = mean last-token hidden of k-shot
                       prompts, REPLACES the query hidden state (Hendel 2023)
      icv_pc1        : top principal component of the contrastive diffs,
                       additive injection (ICV-style)
      retrieval      : nearest dictionary skill by cos(z, anchor), apply that
                       skill's oracle affine operator (ELICIT-style)
  blend32   : anchor-blend (conceptor-style) on all 32 LOTO tasks
              -> improvements.csv (exp=baseline, cond=blend)
  icl4      : 4-shot ICL with the SAME demonstrations -> icl4.csv
  hendel_c  : Hendel-replace on the 10 compound tasks -> hendel_compound.csv
Usage: python 11_baselines.py [lit|blend32|icl4|hendel_c|all]
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.compound import COMPOUND_REGISTRY, load_compound
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, oracle_ops, z_list_from_Z, \
    _support_weights
from cass.steer import make_additive_op, _to_torch
from cass.tasks import ALL_TASKS, load_task, icl_prompt, synthetic_tasks, \
    zs_prompt
from cass.zcache import get_z

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
STAGES = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["all"])
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}


def examples_for(task, seed):
    rng = np.random.default_rng(100 * seed + K)
    idx = rng.choice(len(task.fewshot_pool), K, replace=False)
    return [task.fewshot_pool[i] for i in idx]


def make_replace_op(vec, device="cuda"):
    v = _to_torch(vec, device)

    def op(h):
        return v.unsqueeze(0).expand_as(h).clone()
    return op


def eval_ops(task, ops, lys, cs=False):
    q = task.eval_queries[:50]
    return accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                 batch_size=25, op=ops, layer=lys),
                    [y for _, y in q], case_sensitive=cs)


def open_csv(path, fieldnames, done_key):
    done = set()
    if path.exists():
        with open(path) as f:
            done = {done_key(r) for r in csv.DictReader(f)}
    fout = open(path, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if not done:
        w.writeheader()
    return done, fout, w


def loto_dictionary(tstar):
    return build_multilayer_dictionary(
        {l: {t: G[l][t] for t in ALL_TASKS if t != tstar} for l in LAYERS},
        r0=1)


def lit():
    D_full = build_multilayer_dictionary(G, r0=1)
    done, fout, w = open_csv(
        out / "baselines_lit.csv", ["suite", "method", "task", "seed", "acc"],
        lambda r: (r["suite"], r["method"], r["task"], r["seed"]))
    t0 = time.time()
    for suite, names in [("loto", ALL_TASKS),
                         ("novel", list(synthetic_tasks()))]:
        for tname in names:
            task = load_task(tname)
            D = loto_dictionary(tname) if suite == "loto" else D_full
            for seed in [0, 1, 2]:
                need = [m for m in ["hendel_replace", "icv_pc1", "retrieval"]
                        if (suite, m, tname, str(seed)) not in done]
                if not need:
                    continue
                examples = examples_for(task, seed)

                if "hendel_replace" in need:
                    # mean last-token hidden of k-shot prompts (leave-self-out)
                    prompts = []
                    for j, (x, _) in enumerate(examples):
                        shots = [e for i, e in enumerate(examples) if i != j]
                        prompts.append(icl_prompt(shots, x))
                    H = hlm.last_token_hiddens(prompts, batch_size=8)
                    ops = [make_replace_op(H[:, l].mean(0).numpy())
                           for l in LAYERS]
                    w.writerow(dict(suite=suite, method="hendel_replace",
                                    task=tname, seed=seed,
                                    acc=eval_ops(task, ops, list(LAYERS))))
                    fout.flush()

                if "icv_pc1" in need or "retrieval" in need:
                    Z = get_z(hlm, task, K, seed)  # [k, L+1, d]

                if "icv_pc1" in need:
                    # top PC of per-example diffs, sign-aligned to the mean,
                    # rescaled to the median anchor norm, additive
                    ops = []
                    for l in LAYERS:
                        Zl = Z[:, l].numpy().astype(np.float64)
                        Zl = np.stack([D.per_layer[l].project_out_shared(z)
                                       for z in Zl])
                        pc = np.linalg.svd(Zl, full_matrices=False)[2][0]
                        if pc @ Zl.mean(0) < 0:
                            pc = -pc
                        norm = np.median([np.linalg.norm(
                            D.per_layer[l].anchors[t]) for t in D.task_names])
                        ops.append(make_additive_op(pc * norm, gamma=1.0))
                    w.writerow(dict(suite=suite, method="icv_pc1", task=tname,
                                    seed=seed,
                                    acc=eval_ops(task, ops, list(LAYERS))))
                    fout.flush()

                if "retrieval" in need:
                    z = np.concatenate([D.per_layer[l].project_out_shared(
                        Z[:, l].numpy().astype(np.float64).mean(0))
                        for l in LAYERS])
                    sims = {t: abs(float(z @ D.anchors[t] /
                            (np.linalg.norm(z) * np.linalg.norm(D.anchors[t])
                             + 1e-12))) for t in D.task_names}
                    nearest = max(sims, key=sims.get)
                    ops, lys = oracle_ops(D, nearest, 1.0, 2.0, 1.0)
                    w.writerow(dict(suite=suite, method="retrieval",
                                    task=tname, seed=seed,
                                    acc=eval_ops(task, ops, lys)))
                    fout.flush()
            print(f"{suite}/{tname} done ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    fout.close()
    print("LIT BASELINES DONE")


def blend32():
    done, fout, w = open_csv(
        out / "improvements.csv", ["exp", "cond", "task", "seed", "acc"],
        lambda r: (r["exp"], r["cond"], r["task"], r["seed"]))
    for tstar in ALL_TASKS:
        task = load_task(tstar)
        D = loto_dictionary(tstar)
        for seed in [0, 1, 2]:
            if ("baseline", "blend", tstar, str(seed)) in done:
                continue
            Z = get_z(hlm, task, K, seed)
            z_list = z_list_from_Z(D, Z)
            code = code_for(D, z_list)
            if code.support:
                ws = _support_weights(code)
                blend = sum(wi * D.anchors[n]
                            for wi, n in zip(ws, code.support))
            else:
                blend = np.mean(z_list, axis=0)
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=blend)
            w.writerow(dict(exp="baseline", cond="blend", task=tstar,
                            seed=seed, acc=eval_ops(task, ops, lys)))
            fout.flush()
        print(tstar, "done", flush=True)
    fout.close()
    print("BLEND DONE")


def icl4():
    done, fout, w = open_csv(
        out / "icl4.csv", ["suite", "task", "seed", "acc"],
        lambda r: (r["task"], r["seed"]))
    for suite, names in [("loto", ALL_TASKS),
                         ("novel", list(synthetic_tasks()))]:
        for tname in names:
            task = load_task(tname)
            q = task.eval_queries[:50]
            for seed in [0, 1, 2]:
                if (tname, str(seed)) in done:
                    continue
                shots = examples_for(task, seed)
                prompts = [icl_prompt(shots, x) for x, _ in q]
                acc = accuracy(hlm.generate(prompts, batch_size=16),
                               [y for _, y in q])
                w.writerow(dict(suite=suite, task=tname, seed=seed, acc=acc))
                fout.flush()
            print(tname, "done", flush=True)
    fout.close()
    print("ICL4 DONE")


def hendel_c():
    fout = open(out / "hendel_compound.csv", "w", newline="")
    w = csv.DictWriter(fout, fieldnames=["compound", "seed", "acc"])
    w.writeheader()
    for cname in COMPOUND_REGISTRY:
        comp = load_compound(cname)
        q = comp.eval_queries
        targets = [y for _, y in q]
        prompts = [zs_prompt(x) for x, _ in q]
        for seed in [0, 1, 2, 3, 4]:
            ex = examples_for(comp, seed)
            kp = []
            for j, (x, _) in enumerate(ex):
                shots = [e for i, e in enumerate(ex) if i != j]
                kp.append(icl_prompt(shots, x))
            H = hlm.last_token_hiddens(kp, batch_size=8)
            ops = [make_replace_op(H[:, l].mean(0).numpy()) for l in LAYERS]
            preds = hlm.generate(prompts, batch_size=25, op=ops,
                                 layer=list(LAYERS))
            w.writerow(dict(compound=cname, seed=seed,
                            acc=accuracy(preds, targets,
                                         case_sensitive=True)))
            fout.flush()
        print(cname, "done", flush=True)
    fout.close()
    print("HENDEL COMPOUND DONE")


if __name__ == "__main__":
    stages = (["lit", "blend32", "icl4", "hendel_c"] if STAGES == ["all"]
              else STAGES)
    for s in stages:
        {"lit": lit, "blend32": blend32, "icl4": icl4,
         "hendel_c": hendel_c}[s]()
