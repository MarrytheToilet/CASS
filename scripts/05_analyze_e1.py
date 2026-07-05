"""Analyze E1: recovery rates, go/no-go tier, figures (family bars, eps-vs-rho)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cass.config import results_dir, FIGURES_DIR
from cass.tasks import TASK_REGISTRY

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
MIN_GAP = 0.05

def main():
    out = results_dir(MODEL)
    df = pd.read_csv(out / "e1_loto.csv")
    bl = json.load(open(out / "baselines.json"))

    oracle = df[df["mode"] == "oracle"].set_index("task")["acc"].to_dict()
    naive = df[df["mode"] == "naive"].set_index("task")["acc"].to_dict()
    cass = df[df["mode"] == "cass"]

    records = []
    for (task, k), grp in cass.groupby(["task", "k"]):
        zs, icl = bl[task]["zs"], bl[task]["icl_mean"]
        orc = oracle.get(task, np.nan)
        denom = orc - zs
        acc = grp["acc"].mean()
        rho = (acc - zs) / denom if denom > MIN_GAP else np.nan
        rho_icl = (acc - zs) / (icl - zs) if icl - zs > MIN_GAP else np.nan
        records.append(dict(task=task, family=TASK_REGISTRY[task][1], k=k,
                            acc_syn=acc, acc_zs=zs, acc_icl=icl,
                            acc_oracle=orc, acc_naive=naive.get(task, np.nan),
                            rho=rho, rho_icl=rho_icl,
                            eps=grp["eps"].mean(),
                            support_size=grp["support_size"].mean()))
    R = pd.DataFrame(records)
    R.to_csv(out / "e1_summary.csv", index=False)

    for k in sorted(R["k"].unique()):
        sub = R[R["k"] == k].dropna(subset=["rho"])
        med = sub["rho"].median()
        frac = (sub["rho"] >= 0.6).mean()
        print(f"k={k}: median rho={med:.3f}  frac(rho>=0.6)={frac:.2f}  "
              f"n_valid={len(sub)}")
        for fam, g in sub.groupby("family"):
            print(f"   {fam}: median rho={g['rho'].median():.3f} "
                  f"({[f'{t}:{r:.2f}' for t, r in zip(g.task, g.rho)]})")

    # go/no-go on k=4
    sub = R[(R["k"] == 4)].dropna(subset=["rho"])
    med, frac = sub["rho"].median(), (sub["rho"] >= 0.6).mean()
    fam_med = sub.groupby("family")["rho"].median()
    if frac >= 0.5 and med >= 0.5:
        tier = "A (full speed, main narrative)"
    elif (fam_med >= 0.6).sum() >= 2:
        tier = "B (within-family narrative)"
    elif med < 0.3:
        tier = "C (stop-loss: pivot to high-rank repair)"
    else:
        tier = "between B and C -- inspect manually"
    print(f"\nGO/NO-GO (k=4): median rho={med:.3f} frac>=0.6={frac:.2f} "
          f"-> tier {tier}")
    print(f"family medians:\n{fam_med}")

    # naive comparison
    zs_arr = sub["acc_zs"]
    naive_rho = (sub["acc_naive"] - zs_arr) / (sub["acc_oracle"] - zs_arr)
    print(f"\nnaive-average baseline: median rho={naive_rho.median():.3f}")

    # dictionary contribution: cass (hybrid) vs zvec (no dictionary) vs recon
    for mode in ["zvec", "cass_recon"]:
        m = (df[df["mode"] == mode].groupby(["task", "k"])["acc"].mean()
             .rename(mode))
        c = (df[(df["mode"] == "cass") & (df["seed"] < 3)]
             .groupby(["task", "k"])["acc"].mean().rename("cass"))
        j = pd.concat([c, m], axis=1).dropna()
        for k in sorted(j.index.get_level_values("k").unique()):
            jk = j.xs(k, level="k")
            print(f"cass vs {mode} (k={k}): mean acc {jk['cass'].mean():.3f} "
                  f"vs {jk[mode].mean():.3f}  (paired diff "
                  f"{(jk['cass'] - jk[mode]).mean():+.3f}, cass better on "
                  f"{(jk['cass'] > jk[mode]).sum()}/{len(jk)} tasks)")

    # figures
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sub4 = R[R["k"] == 4].sort_values(["family", "task"])
    x = np.arange(len(sub4))
    ax.bar(x - 0.2, sub4["acc_syn"], 0.4, label="CASS (k=4)")
    ax.bar(x + 0.2, sub4["acc_oracle"], 0.4, label="oracle", alpha=0.6)
    ax.plot(x, sub4["acc_icl"], "k_", markersize=12, label="10-shot ICL")
    ax.plot(x, sub4["acc_zs"], "r_", markersize=12, label="zero-shot")
    ax.set_xticks(x)
    ax.set_xticklabels(sub4["task"], rotation=75, fontsize=7)
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"e1_main_{MODEL}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    v = R.dropna(subset=["rho"])
    ax.scatter(v["eps"], v["rho"].clip(-0.2, 1.5), c=v["k"], cmap="viridis", s=25)
    ax.set_xlabel("reconstruction residual eps")
    ax.set_ylabel("recovery rho")
    plt.colorbar(ax.collections[0], label="k")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"e1_eps_vs_rho_{MODEL}.png", dpi=150)
    print(f"\nfigures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
