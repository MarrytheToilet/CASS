"""Paper figures v2 — light blue / light pink palette (validated:
#5aa7e0 #ef8fb9 #2f6fb3 #cf5a92, CVD dE 19.5, WARN-contrast mitigated by
direct labels + axis ticks). PDF output for LaTeX two-column."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from cass.config import results_dir, FIGURES_DIR
from cass.compound import compound_components
from cass.tasks import ALL_TASKS, TASK_REGISTRY

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
out = results_dir(MODEL)

BLUE, PINK = "#5aa7e0", "#ef8fb9"          # light fills
DBLUE, DPINK = "#2f6fb3", "#cf5a92"        # lines / emphasis
INK, MUTED, GRID = "#3b3b3b", "#8a8a8a", "#e9e9e9"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#ffffff", BLUE, DBLUE])
DIV = LinearSegmentedColormap.from_list("pinkblue",
                                        [DPINK, PINK, "#f2f2f2", BLUE, DBLUE])

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})

FAM_LABEL = {"algorithmic": "algorithmic", "knowledge": "knowledge",
             "linguistic": "linguistic", "selection": "selection",
             "translation": "translation"}


def rounded_bar(ax, x, height, width, color, zorder=3):
    """Bar with a rounded data-end and square baseline."""
    if height <= 0.005:
        return
    r = min(0.012, height / 2)
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=width / 0.05,
        facecolor=color, edgecolor="none", zorder=zorder))


# ---------------- Fig E1: bullet chart ----------------
R = pd.read_csv(out / "e1_summary.csv")
R4 = R[R["k"] == 4].sort_values(["family", "task"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.2, 2.9))
x = np.arange(len(R4))
# family bands (alternating wash)
prev, start = None, 0
for i, f in enumerate(list(R4["family"]) + [None]):
    if f != prev:
        if prev is not None and (start // 1) % 2 == 0:
            pass
        if prev is not None:
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
            ax.text((start + i - 1) / 2, 1.13, FAM_LABEL.get(prev, prev),
                    ha="center", va="top", fontsize=7.5, color=MUTED)
        prev, start = f, i
# oracle: wide light-pink backdrop bar
for i, row in R4.iterrows():
    ax.bar(i, row["acc_oracle"], width=0.82, color=PINK, alpha=0.55,
           edgecolor="none", zorder=1)
    rounded_bar(ax, i, row["acc_syn"], 0.44, BLUE)
    ax.plot([i - 0.41, i + 0.41], [row["acc_icl"]] * 2, color=INK, lw=1.1,
            zorder=4)
    if row["acc_naive"] > 0.02:
        ax.plot(i, row["acc_naive"], "x", color=MUTED, markersize=3.5,
                zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(R4["task"], rotation=62, fontsize=6, ha="right",
                   rotation_mode="anchor")
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.16)
ax.set_xlim(-0.7, len(R4) - 0.3)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="y", zorder=0)
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=BLUE, ec="none"),
    plt.Rectangle((0, 0), 1, 1, fc=PINK, alpha=0.55, ec="none"),
    Line2D([0], [0], color=INK, lw=1.1),
    Line2D([0], [0], marker="x", color=MUTED, lw=0, markersize=4),
]
ax.legend(handles, ["CASS (k=4)", "oracle (own subspace)", "10-shot ICL",
                    "naive composition"],
          ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.26), fontsize=7,
          handlelength=1.4, columnspacing=1.2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e1_main_{MODEL}.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / f"e1_main_{MODEL}.png", dpi=170,
            bbox_inches="tight")
plt.close()

# ---------------- Fig E5: scale curve ----------------
d5 = pd.read_csv(out / "e5_scale_ext.csv" if (out / "e5_scale_ext.csv"
                 ).exists() else out / "e5_scale.csv")
g = d5.groupby(["size", "draw"])["acc"].mean().reset_index()
m = g.groupby("size")["acc"].agg(["mean", "sem"])
# no-dictionary reference from E1 zvec on the same held-out tasks
e1 = pd.read_csv(out / "e1_loto.csv")
held = d5["task"].unique()
zref = e1[(e1["mode"] == "zvec") & (e1["k"] == 4) &
          (e1["task"].isin(held))]["acc"].mean()

fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.grid(axis="y", zorder=0)
ax.fill_between(m.index, m["mean"] - 1.96 * m["sem"],
                m["mean"] + 1.96 * m["sem"], color=BLUE, alpha=0.18, lw=0,
                zorder=1)
ax.plot(m.index, m["mean"], color=DBLUE, lw=2, zorder=3,
        solid_capstyle="round")
ax.plot(m.index, m["mean"], "o", color=DBLUE, markersize=5,
        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
ax.axhline(zref, color=DPINK, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.text(m.index[-1], zref - 0.035, "no dictionary ($z$ only)",
        ha="right", va="top", fontsize=7, color=DPINK)
ax.text(m.index[-1], m["mean"].iloc[-1] + 0.04, "CASS", ha="right",
        fontsize=7.5, color=DBLUE, fontweight="bold")
ax.set_xlabel("dictionary size $T'$ (skills)")
ax.set_ylabel("held-out accuracy")
ax.set_xticks(list(m.index))
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e5_scale_{MODEL}.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / f"e5_scale_{MODEL}.png", dpi=170,
            bbox_inches="tight")
plt.close()

# ---------------- Fig eps vs rho ----------------
Rv = R.dropna(subset=["rho"]).copy()
Rv["rho_c"] = Rv["rho"].clip(-0.2, 1.6)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.grid(zorder=0)
kcols = {1: "#a8cdec", 2: BLUE, 4: DBLUE}      # ordinal blue ramp
for k, col in kcols.items():
    s = Rv[Rv["k"] == k]
    ax.scatter(s["eps"], s["rho_c"], s=26, color=col, label=f"$k={k}$",
               edgecolor="white", linewidth=0.9, zorder=3)
b, a = np.polyfit(Rv["eps"], Rv["rho_c"], 1)
xs = np.linspace(Rv["eps"].min(), Rv["eps"].max(), 20)
ax.plot(xs, a + b * xs, color=DPINK, lw=1.6, ls=(0, (4, 3)), zorder=2)
r = np.corrcoef(Rv["eps"], Rv["rho_c"])[0, 1]
ax.text(0.03, 0.06, f"$r = {r:.2f}$", transform=ax.transAxes, fontsize=8,
        color=DPINK)
ax.set_xlabel(r"coding residual $\varepsilon$")
ax.set_ylabel(r"oracle recovery $\rho$")
ax.legend(loc="upper right", fontsize=7, handletextpad=0.1,
          borderaxespad=0.1)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e1_eps_vs_rho_{MODEL}.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / f"e1_eps_vs_rho_{MODEL}.png", dpi=170,
            bbox_inches="tight")
plt.close()

# ---------------- Fig E2 heatmap ----------------
cm = json.load(open(out / "e2_coeff_matrix.json"))
compounds = list(cm)
tasks = sorted(ALL_TASKS, key=lambda t: (TASK_REGISTRY[t][1], t))
Mx = np.array([[cm[c][t] for t in tasks] for c in compounds])
Mx = Mx / (Mx.max(axis=1, keepdims=True) + 1e-12)
fig, ax = plt.subplots(figsize=(9.2, 3.1))
im = ax.imshow(Mx, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
for i, c in enumerate(compounds):
    for comp in compound_components(c):
        if comp in tasks:
            j = tasks.index(comp)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=DPINK, lw=1.6, zorder=4))
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks, rotation=62, fontsize=5.6, ha="right",
                   rotation_mode="anchor")
ax.set_yticks(range(len(compounds)))
ax.set_yticklabels([c.replace("+", " $\\circ$ ") for c in compounds],
                   fontsize=6.5)
# family separators
prev = None
for j, t in enumerate(tasks):
    f = TASK_REGISTRY[t][1]
    if f != prev and prev is not None:
        ax.axvline(j - 0.5, color="white", lw=1.6)
    prev = f
cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("normalized coefficient", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=DPINK, lw=1.6)]
ax.legend(handles, ["ground-truth constituent"], loc="upper left",
          bbox_to_anchor=(0.0, 1.14), fontsize=7)
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"e2_heatmap_{MODEL}.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / f"e2_heatmap_{MODEL}.png", dpi=170,
            bbox_inches="tight")
plt.close()

# ---------------- Fig cosine matrices (H1) ----------------
from cass.dictionary import build_dictionary, pairwise_cosine
from cass.extract import load_G
G16 = {t: load_G(MODEL, t, 16).numpy() for t in tasks}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax_, r0, title in [(axes[0], 0, "before removal ($r_0{=}0$)"),
                       (axes[1], 1, "after removal ($r_0{=}1$)")]:
    D = build_dictionary(G16, r0=r0)
    C = pairwise_cosine(D.anchors)
    im = ax_.imshow(C, cmap=DIV, vmin=-1, vmax=1)
    ax_.set_title(title, fontsize=8, color=INK)
    ax_.set_xticks([])
    ax_.set_yticks([])
    prev = None
    for j, t in enumerate(tasks):
        f = TASK_REGISTRY[t][1]
        if f != prev and prev is not None:
            ax_.axvline(j - 0.5, color="white", lw=1.2)
            ax_.axhline(j - 0.5, color="white", lw=1.2)
        prev = f
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("pairwise cosine of task means", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
plt.savefig(FIGURES_DIR / f"cosine_matrix_{MODEL}.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / f"cosine_matrix_{MODEL}.png", dpi=170,
            bbox_inches="tight")
plt.close()
print("figures v2 written to", FIGURES_DIR)
