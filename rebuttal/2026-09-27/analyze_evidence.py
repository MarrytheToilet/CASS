"""Read-only reanalysis of existing checkpoints for the September reviews.

Writes only under this directory. Common seeds, task-level paired bootstrap,
and exactly shared routing assignments isolate the dictionary correction.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for small matrix operations.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cass.dictionary import build_multilayer_dictionary, subcoherence_matrix
from cass.pipeline import z_list_from_Z
from cass.solver import group_lasso
from cass.tasks import ALL_TASKS

SOURCE = ROOT / "results" / "llama31-8b"
SEEDS = [0, 1, 2]
THRESHOLDS = np.linspace(1.0, 8.0, 57)


def write_json(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2) + "\n")


def load_activations(model, layers):
    result = {l: {} for l in layers}
    for task in ALL_TASKS:
        compact = ROOT / "results" / model / "activations_compact" / f"{task}.pt"
        path = compact if compact.exists() else ROOT / "results" / model / "activations" / f"{task}.pt"
        blob = torch.load(path, map_location="cpu", weights_only=True)
        for l in layers:
            result[l][task] = (blob["G_by_layer"][l] if "G_by_layer" in blob
                              else blob["G"][:, l]).float().numpy()
    return result


def dictionary(model="llama31-8b"):
    hp = json.loads((ROOT / "results" / model / "injection_hparams.json").read_text())
    G = load_activations(model, hp["layers"])
    return G, build_multilayer_dictionary(G, r0=1)


def paired(a, b, nboot=50000):
    """Equal task weights, average matched seeds first, then resample tasks."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    diff = a - b
    rng = np.random.default_rng(20260927)
    boot = diff[rng.integers(len(diff), size=(nboot, len(diff)))].mean(axis=1)
    return dict(n_tasks=len(diff), a=float(a.mean()), b=float(b.mean()),
                diff=float(diff.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                wins=int((diff > 1e-12).sum()), ties=int((abs(diff) <= 1e-12).sum()),
                losses=int((diff < -1e-12).sum()))


def seed_table(frame, methods, expected_tasks):
    frame = frame[frame.seed.isin(SEEDS)].copy()
    assert not frame.duplicated(["task", "seed", "method"]).any()
    table = frame.pivot(index=["task", "seed"], columns="method", values="acc")
    assert set(table.columns) == set(methods), table.columns
    assert not table.isna().any().any(), table[table.isna().any(axis=1)]
    assert table.index.get_level_values("task").nunique() == expected_tasks
    assert (table.groupby(level="task").size() == len(SEEDS)).all()
    return table


def optimal_threshold(df):
    means = [np.where(df.znorm < t, df["replace"], df.cass).mean() for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(means))])


def cached_norms(D, names):
    out = {}
    for name in names:
        vals = []
        for seed in SEEDS:
            Z = torch.load(SOURCE / "zcache" / f"{name}_k4_s{seed}.pt",
                           map_location="cpu", weights_only=True).float()
            vals.append(float(np.linalg.norm(np.mean(z_list_from_Z(D, Z), axis=0))))
        out[name] = float(np.mean(vals))
    return pd.Series(out)


def routed_results(task_df, replace_mask, suite, label):
    methods = [x for x in task_df.columns if x in ["cass", "zvec", "true_support"]]
    out = {m: np.where(replace_mask, task_df["replace"], task_df[m]) for m in methods}
    res = dict(n_tasks=len(task_df), n_replace=int(replace_mask.sum()),
               n_compose=int((~replace_mask).sum()),
               accuracies={m: float(v.mean()) for m, v in out.items()},
               cass_minus_zvec=paired(out["cass"], out["zvec"]),
               compose_only_cass_minus_zvec=paired(task_df.loc[~replace_mask, "cass"],
                                                   task_df.loc[~replace_mask, "zvec"]))
    if "true_support" in out:
        res["true_minus_inferred"] = paired(out["true_support"], out["cass"])
        res["true_minus_zvec"] = paired(out["true_support"], out["zvec"])
    export = task_df.copy()
    export["route"] = np.where(replace_mask, "replace", "compose")
    for m, v in out.items():
        export[f"routed_{m}"] = v
    export.to_csv(HERE / f"{suite}_{label}_per_task.csv")
    return res


def matching():
    lit = pd.read_csv(SOURCE / "baselines_lit.csv")
    e1 = pd.read_csv(SOURCE / "e1_loto.csv")
    e7 = pd.read_csv(SOURCE / "e7_novel.csv")
    e2 = pd.read_csv(SOURCE / "e2_compound.csv")
    gaps = pd.read_csv(SOURCE / "fill_gaps.csv")
    hc = pd.read_csv(SOURCE / "hendel_compound.csv")
    c = e1[(e1.k == 4) & (e1["mode"] == "cass")]
    original = c.groupby("task").agg(cass=("acc", "mean"), znorm=("delta_norm", "mean"))
    original["replace"] = lit[(lit.suite == "loto") & (lit.method == "hendel_replace")].groupby("task").acc.mean()
    frozen = optimal_threshold(original)
    nested = pd.Series({t: optimal_threshold(original.drop(t)) for t in original.index})
    original_route = original.znorm < nested
    z_policy = original.copy()
    z_policy["cass"] = e1[(e1.k == 4) & (e1["mode"] == "zvec")].groupby("task").acc.mean()
    frozen_z = optimal_threshold(z_policy)
    nested_z = pd.Series({t: optimal_threshold(z_policy.drop(t)) for t in z_policy.index})
    result = dict(protocol={
        "common_seeds": SEEDS, "bootstrap_replicates": 50000, "bootstrap_seed": 20260927,
        "bootstrap_unit": "task; common seeds averaged within task",
        "primary_routing": "Freeze original CASS assignments from existing 5-seed results; compare arms on seeds 0,1,2",
        "ci_scope": "Conditional on fixed routing assignments and existing checkpoints; does not include threshold-learning uncertainty",
        "replacement": "The exact same replacement task/seed score is used in each arm",
        "frozen_threshold": frozen,
        "original_separately_tuned_z_threshold": frozen_z,
        "zvec_scope": "Shares U0 removal and support-dependent norm scaling with CASS; isolates gated correction, not all uses of dictionary",
    })
    for suite, raw, n in [("loto", e1[e1.k == 4], 32), ("novel", e7, 15)]:
        arms = raw[raw["mode"].isin(["cass", "zvec"])][["task", "seed", "mode", "acc"]].rename(columns={"mode": "method"})
        repl = lit[(lit.suite == suite) & (lit.method == "hendel_replace")][["task", "seed", "method", "acc"]].copy()
        repl["method"] = "replace"
        table = seed_table(pd.concat([arms, repl]), ["cass", "zvec", "replace"], n)
        table.to_csv(HERE / f"{suite}_matched_seeds.csv")
        df = table.groupby(level="task").mean()
        result[suite] = dict(unrouted=paired(df.cass, df.zvec))
        if suite == "loto":
            df["znorm"] = original.znorm
            df["threshold"] = nested
            result[suite]["fixed_original_route"] = routed_results(df, original_route, suite, "fixed_route")
            result[suite]["separately_tuned_policy_common_seeds"] = paired(
                np.where(original_route,df["replace"],df.cass),
                np.where(original.znorm < nested_z,df["replace"],df.zvec))
            # Sensitivity: train CASS thresholds with only the same three seeds;
            # each threshold still excludes its target task and is shared by arms.
            matched = df.copy()
            matched["znorm"] = c[c.seed.isin(SEEDS)].groupby("task").delta_norm.mean()
            th = pd.Series({t: optimal_threshold(matched.drop(t)) for t in matched.index})
            result[suite]["common_seed_nested_route"] = routed_results(matched, matched.znorm < th, suite, "matched_nested")
        else:
            _, D = dictionary()
            df["znorm"] = cached_norms(D, df.index)
            df["threshold"] = frozen
            result[suite]["fixed_original_route"] = routed_results(df, df.znorm < frozen, suite, "fixed_route")
            result[suite]["separately_tuned_policy_common_seeds"] = paired(
                np.where(df.znorm < frozen,df["replace"],df.cass),
                np.where(df.znorm < frozen_z,df["replace"],df.zvec))
        print(suite, json.dumps(result[suite]), flush=True)
    comp_arms = e2[["compound", "seed", "acc_cass"]].rename(columns={"compound": "task", "acc_cass": "acc"})
    comp_arms["method"] = "cass"
    extra = gaps[(gaps.suite == "compound") & gaps.method.isin(["zvec", "oracle_support"])][["task", "seed", "method", "acc"]].copy()
    extra["method"] = extra.method.replace({"oracle_support": "true_support"})
    repl = hc.rename(columns={"compound": "task"})
    repl["method"] = "replace"
    table = seed_table(pd.concat([comp_arms, extra, repl]), ["cass", "zvec", "true_support", "replace"], 10)
    table.to_csv(HERE / "compound_matched_seeds.csv")
    df = table.groupby(level="task").mean()
    df["znorm"] = cached_norms(D, df.index)
    df["threshold"] = frozen
    result["compound"] = dict(
        unrouted_cass_minus_zvec=paired(df.cass, df.zvec),
        unrouted_true_minus_inferred=paired(df.true_support, df.cass),
        fixed_original_route=routed_results(df, df.znorm < frozen, "compound", "fixed_route"),
        true_support_provenance="Existing fill_gaps.csv; see scripts/12_attribution.py oracle-support implementation")
    print("compound", json.dumps(result["compound"]), flush=True)
    files = ["e1_loto.csv", "e7_novel.csv", "e2_compound.csv", "fill_gaps.csv", "baselines_lit.csv", "hendel_compound.csv"]
    result["source_sha256"] = {f: hashlib.sha256((SOURCE / f).read_bytes()).hexdigest() for f in files}
    write_json("matched_results.json", result)


def coherence_record(D):
    matrices = [subcoherence_matrix(D.per_layer[l]) for l in D.layers]
    # Block-diagonal stacking: the spectral norm is MAX across layers.
    joint = np.maximum.reduce(matrices)
    ii = np.triu_indices(len(D.task_names), 1)
    mu = float(joint[ii].max())
    a, b = np.unravel_index(np.argmax(joint), joint.shape)
    ranks = [D.bases[t].shape[1] for t in D.task_names]
    qratio = np.sqrt(max(ranks) / min(ranks))
    old = [s for s in range(1, len(D.task_names)+1) if (2*s-1)*mu < 1]
    weighted = [s for s in range(1, len(D.task_names)+1) if (s*qratio+s-1)*mu < 1]
    return dict(layers=D.layers, max_joint=mu, median_joint=float(np.median(joint[ii])),
                max_pair=[D.task_names[a], D.task_names[b]],
                per_layer_max={str(l): float(m.max()) for l, m in zip(D.layers, matrices)},
                min_joint_rank=min(ranks), max_joint_rank=max(ranks),
                original_coherence_certified_s=old,
                weighted_global_sufficient_s=weighted)


def coherences(all_loto=False):
    models = ["llama31-8b", "llama32-3b", "gemma2-2b", "qwen3-4b", "qwen25-3b"]
    result = {}
    for model in models:
        G, D = dictionary(model)
        result[model] = {"full": coherence_record(D)}
        print(model, "full", result[model]["full"], flush=True)
        if model == "llama31-8b" or all_loto:
            records = {}
            for t in ALL_TASKS:
                sub = build_multilayer_dictionary({l: {n: v for n, v in g.items() if n != t} for l, g in G.items()}, r0=1)
                records[t] = coherence_record(sub)
                print(model, "held-out", t, records[t]["max_joint"], flush=True)
            result[model]["loto"] = records
        write_json("coherence_results.json", result)


def counterexample():
    from types import SimpleNamespace
    U1 = np.eye(5)[:, :4]
    U2 = np.array([[0.6], [0.0], [0.0], [0.0], [0.8]])
    D = SimpleNamespace(task_names=["true", "spurious"], bases={"true": U1, "spurious": U2})
    eps = 0.01
    lam = 0.01
    z = np.array([1+eps, 0, 0, 0, 0])
    code = group_lasso(D, z, lam=lam, max_iter=10000, tol=1e-12)
    restricted = np.array([1+eps-2*lam, 0, 0, 0])
    offcorr = float(np.linalg.norm(U2.T @ (z-U1@restricted)))
    assert code.support == ["true", "spurious"]
    assert offcorr > lam
    result = dict(rank_true=4, rank_spurious=1, mu=0.6, epsilon=eps, lam=lam,
                  original_condition_holds=bool(1 < 0.5*(1/0.6+1)),
                  restricted_inactive_correlation=offcorr, allowed_correlation=lam,
                  solution={k: v.tolist() for k, v in code.coeffs.items()},
                  support=code.support,
                  note="Independent counterexample constructed here, not the missing reviewer-supplied example")
    write_json("weighted_counterexample.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["matched", "coherence", "counterexample", "all"])
    p.add_argument("--all-loto", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage in ["counterexample", "all"]:
        counterexample()
    if args.stage in ["matched", "all"]:
        matching()
    if args.stage in ["coherence", "all"]:
        coherences(args.all_loto)
