"""Generate paper figures from result CSVs (no GPU needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cass.config import results_dir, FIGURES_DIR
from cass.compound import COMPOUND_REGISTRY, compound_components
from cass.tasks import ALL_TASKS, TASK_REGISTRY

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
out = results_dir(MODEL)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- Fig: E5 scale curve ----
d5 = pd.read_csv(out / "e5_scale.csv")
fig, ax = plt.subplots(figsize=(4.2, 3))
g = d5.groupby(["size", "draw"])["acc"].mean().reset_index()
m = g.groupby("size")["acc"].agg(["mean", "sem"])
ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], marker="o",
            capsize=3, lw=1.5, color="#2166ac")
ax.set_xlabel("dictionary size $T'$ (skills)")
ax.set_ylabel("accuracy on held-out tasks")
ax.set_xticks(m.index)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e5_scale_{MODEL}.pdf")
plt.savefig(FIGURES_DIR / f"e5_scale_{MODEL}.png", dpi=150)
plt.close()

# ---- Fig: E2 coefficient heatmap ----
cm = json.load(open(out / "e2_coeff_matrix.json"))
compounds = list(cm)
tasks = sorted(ALL_TASKS, key=lambda t: (TASK_REGISTRY[t][1], t))
Mx = np.array([[cm[c][t] for t in tasks] for c in compounds])
Mx = Mx / (Mx.max(axis=1, keepdims=True) + 1e-12)   # row-normalize
fig, ax = plt.subplots(figsize=(9, 3.6))
sns.heatmap(Mx, ax=ax, cmap="Blues", cbar_kws={"label": "norm. coeff"},
            xticklabels=tasks, yticklabels=compounds, linewidths=0.3,
            linecolor="white")
for i, c in enumerate(compounds):
    for comp in compound_components(c):
        if comp in tasks:
            j = tasks.index(comp)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor="#b2182b", lw=1.4))
ax.tick_params(axis="x", labelsize=6, rotation=90)
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e2_heatmap_{MODEL}.pdf")
plt.savefig(FIGURES_DIR / f"e2_heatmap_{MODEL}.png", dpi=150)
plt.close()

# ---- Fig: eps vs rho (from e1 summary) ----
R = pd.read_csv(out / "e1_summary.csv").dropna(subset=["rho"])
fig, ax = plt.subplots(figsize=(4.2, 3))
sc = ax.scatter(R["eps"], R["rho"].clip(-0.2, 1.6), c=R["k"], cmap="viridis",
                s=22, alpha=0.85)
cor = np.corrcoef(R["eps"], R["rho"].clip(-0.2, 1.6))[0, 1]
ax.set_xlabel(r"coding residual $\varepsilon$")
ax.set_ylabel(r"oracle recovery $\rho$")
ax.set_title(f"r = {cor:.2f}", fontsize=9)
plt.colorbar(sc, label="$k$")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e1_eps_vs_rho_{MODEL}.pdf")
plt.savefig(FIGURES_DIR / f"e1_eps_vs_rho_{MODEL}.png", dpi=150)
plt.close()

# ---- Fig: E1 main bars ----
R4 = pd.read_csv(out / "e1_summary.csv")
R4 = R4[R4["k"] == 4].sort_values(["family", "task"])
fig, ax = plt.subplots(figsize=(9, 3))
x = np.arange(len(R4))
ax.bar(x - 0.2, R4["acc_syn"], 0.4, label="CASS (k=4)", color="#2166ac")
ax.bar(x + 0.2, R4["acc_oracle"], 0.4, label="oracle", color="#92c5de")
ax.plot(x, R4["acc_icl"], "_", color="k", markersize=10, label="10-shot ICL")
ax.plot(x, R4["acc_naive"], "x", color="#b2182b", markersize=4,
        label="naive avg")
ax.set_xticks(x)
ax.set_xticklabels(R4["task"], rotation=80, fontsize=6)
ax.set_ylabel("accuracy")
ax.legend(fontsize=7, ncol=4)
prev = None
for i, f in enumerate(R4["family"]):
    if f != prev:
        ax.axvline(i - 0.5, color="gray", lw=0.5, ls=":")
        prev = f
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e1_main_{MODEL}.pdf")
plt.savefig(FIGURES_DIR / f"e1_main_{MODEL}.png", dpi=150)
plt.close()
print("figures written to", FIGURES_DIR)
