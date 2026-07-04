"""Paper figures v3 — light blue / light pink palette, PNG only.

Design notes: no spines, y-grid only, semantic short task names, two-row
bullet chart for E1, support-trimmed E2 heatmap, binned eps-rho strip plot.
Palette validated (#5aa7e0 #ef8fb9 #2f6fb3 #cf5a92, CVD dE 19.5)."""
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
from cass.compound import compound_components, COMPOUND_REGISTRY
from cass.tasks import ALL_TASKS, TASK_REGISTRY

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
out = results_dir(MODEL)

BLUE, PINK = "#5aa7e0", "#f2aac9"
DBLUE, DPINK = "#2f6fb3", "#cf5a92"
INK, MUTED, GRID = "#3b3b3b", "#8a8a8a", "#ececec"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#ffffff", BLUE, DBLUE])
DIV = LinearSegmentedColormap.from_list("pinkblue",
                                        [DPINK, PINK, "#f4f4f4", BLUE, DBLUE])

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.0,
    "axes.labelcolor": INK, "xtick.color": "#cccccc",
    "ytick.color": "#cccccc", "xtick.labelcolor": INK,
    "ytick.labelcolor": "#6b6b6b", "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": False, "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "xtick.major.size": 0, "ytick.major.size": 0,
})

SHORT = {
    "alphabetically-first": "alpha-first", "alphabetically-last": "alpha-last",
    "choose-first-of-list": "choose-first", "choose-last-of-list":
    "choose-last", "choose-middle-of-list": "choose-middle",
    "next-capital-letter": "next-cap-letter", "word-length": "word-len",
    "next-item": "next-item", "prev-item": "prev-item",
    "country-capital": "capital", "country-currency": "currency",
    "landmark-country": "landmark", "park-country": "park",
    "person-occupation": "occupation", "person-sport": "sport",
    "person-instrument": "instrument", "product-company": "company",
    "antonym": "antonym", "synonym": "synonym", "capitalize": "capitalize",
    "capitalize-first-letter": "cap-1st-letter",
    "capitalize-last-letter": "cap-last-letter",
    "lowercase-first-letter": "lower-1st-letter",
    "present-past": "pres→past", "singular-plural": "sing→plural",
    "english-french": "en→fr", "english-german": "en→de",
    "english-spanish": "en→es",
    "animal-from-list": "animal", "color-from-list": "color",
    "fruit-from-list": "fruit", "verb-from-list": "verb",
}
short = lambda t: SHORT.get(t, t)


def rounded_bar(ax, x, height, width, color, zorder=3):
    if height <= 0.005:
        return
    r = min(0.012, height / 2)
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=width / 0.05, facecolor=color, edgecolor="none",
        zorder=zorder))


PAPER_FIGS = FIGURES_DIR.parent / "paper" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.savefig(FIGURES_DIR / f"{name}_{MODEL}.png", dpi=200,
                bbox_inches="tight")
    import shutil
    shutil.copy(FIGURES_DIR / f"{name}_{MODEL}.png",
                PAPER_FIGS / f"{name}_{MODEL}.png")
    plt.close()


# ---------------- Fig E1: composite (bullet + k-trend + family) ---------
R = pd.read_csv(out / "e1_summary.csv")
R4 = R[R["k"] == 4].sort_values(["family", "task"]).reset_index(drop=True)
e1 = pd.read_csv(out / "e1_loto.csv")
bl = json.load(open(out / "baselines.json"))

fig = plt.figure(figsize=(9.6, 3.7))
gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1], hspace=0.52, wspace=0.2)
axA1 = fig.add_subplot(gs[0, 0])
axA2 = fig.add_subplot(gs[1, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1])

half = int(np.ceil(len(R4) / 2))
for ax, chunk in zip([axA1, axA2], [R4.iloc[:half], R4.iloc[half:]]):
    chunk = chunk.reset_index(drop=True)
    x = np.arange(len(chunk))
    ax.grid(axis="y", zorder=0)
    prev, start = None, 0
    for i, f in enumerate(list(chunk["family"]) + [None]):
        if f != prev:
            if prev is not None:
                ax.text((start + i - 1) / 2, 1.2, prev, ha="center",
                        va="top", fontsize=6.8, color=MUTED, style="italic")
                if f is not None:
                    ax.axvline(i - 0.5, color="#dddddd", lw=0.8, zorder=0)
            prev, start = f, i
    for i, row in chunk.iterrows():
        ax.bar(i, row["acc_oracle"], width=0.72, color=PINK, alpha=0.55,
               edgecolor="none", zorder=1)
        rounded_bar(ax, i, row["acc_syn"], 0.42, BLUE)
        ax.plot([i - 0.4, i + 0.4], [row["acc_icl"]] * 2, color=INK,
                lw=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([short(t) for t in chunk["task"]], rotation=40,
                       fontsize=6.2, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.7, half - 0.3)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("accuracy", fontsize=7.5)
handles = [
    Rectangle((0, 0), 1, 1, fc=BLUE, ec="none"),
    Rectangle((0, 0), 1, 1, fc=PINK, alpha=0.55, ec="none"),
    Line2D([0], [0], color=INK, lw=1.0),
]
axA1.legend(handles, ["CASS (k=4)", "oracle", "10-shot ICL"],
            ncol=3, loc="lower left", bbox_to_anchor=(0.42, 1.16),
            fontsize=7, handlelength=1.2, columnspacing=1.0)
axA1.set_title("A  per-task accuracy", fontsize=8, loc="left",
               color=INK, fontweight="bold", pad=16)

# panel B: median rho vs k for cass / zvec / recon
def rho_by(mode, seeds):
    orc = e1[e1["mode"] == "oracle"].set_index("task")["acc"]
    sub = e1[(e1["mode"] == mode) & (e1["seed"] < seeds)]
    med = {}
    for k, grp in sub.groupby("k"):
        acc = grp.groupby("task")["acc"].mean()
        rhos = []
        for t, a in acc.items():
            zs = bl[t]["zs"]
            if orc.get(t, 0) - zs > 0.05:
                rhos.append((a - zs) / (orc[t] - zs))
        med[k] = np.median(rhos)
    return med

for mode, col, lab in [("cass", DBLUE, "CASS"), ("zvec", MUTED, "$z$ only"),
                       ("cass_recon", DPINK, "reconstruction")]:
    m = rho_by(mode, 3)
    ks = sorted(m)
    axB.plot(ks, [m[k] for k in ks], color=col, lw=2, marker="o",
             markersize=5, markeredgecolor="white", markeredgewidth=1.1,
             label=lab, solid_capstyle="round")
axB.grid(axis="y", zorder=0)
axB.set_xticks([1, 2, 4])
axB.set_xlabel("examples $k$", fontsize=7.5)
axB.set_ylabel(r"median $\rho$", fontsize=7.5)
axB.legend(fontsize=6.5, loc="center right", handlelength=1.4)
axB.set_title("B  recovery vs. $k$", fontsize=8, loc="left", color=INK,
              fontweight="bold")

# panel C: rho by family at k=4 (dot strip + median tick)
fams = sorted(R4["family"].unique())
rngj = np.random.default_rng(1)
Rv = R4.dropna(subset=["rho"])
for fi, f in enumerate(fams):
    v = Rv[Rv["family"] == f]["rho"].clip(0, 1.6)
    axC.scatter(v, fi + rngj.uniform(-0.08, 0.08, len(v)), s=16, color=BLUE,
                alpha=0.75, edgecolor="white", linewidth=0.7, zorder=2)
    axC.plot([v.median()] * 2, [fi - 0.26, fi + 0.26], color=DPINK, lw=2.2,
             zorder=3, solid_capstyle="round")
axC.axvline(1.0, color="#cccccc", lw=0.8, ls=(0, (3, 3)), zorder=1)
axC.text(1.0, -0.72, "oracle", fontsize=6, color=MUTED, ha="center")
axC.set_yticks(range(len(fams)))
axC.set_yticklabels(fams, fontsize=7)
axC.set_xlabel(r"oracle recovery $\rho$  ($k{=}4$)", fontsize=7.5)
axC.grid(axis="x", zorder=0)
axC.set_title("C  recovery by family", fontsize=8, loc="left", color=INK,
              fontweight="bold")
save("e1_main")

# ---------------- Fig E5: scale curve ----------------
d5 = pd.read_csv(out / "e5_scale_ext.csv" if (out / "e5_scale_ext.csv"
                 ).exists() else out / "e5_scale.csv")
g = d5.groupby(["size", "draw"])["acc"].mean().reset_index()
m = g.groupby("size")["acc"].agg(["mean", "sem"])
e1 = pd.read_csv(out / "e1_loto.csv")
held = d5["task"].unique()
zref = e1[(e1["mode"] == "zvec") & (e1["k"] == 4) &
          (e1["task"].isin(held))]["acc"].mean()

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.grid(axis="y", zorder=0)
ax.fill_between(m.index, m["mean"] - 1.96 * m["sem"],
                m["mean"] + 1.96 * m["sem"], color=BLUE, alpha=0.16, lw=0)
ax.plot(m.index, m["mean"], color=DBLUE, lw=2.2, zorder=3,
        solid_capstyle="round")
ax.plot(m.index, m["mean"], "o", color=DBLUE, markersize=5.5,
        markeredgecolor="white", markeredgewidth=1.3, zorder=4)
ax.axhline(zref, color=DPINK, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.text(m.index[-1], zref - 0.03, "no dictionary ($z$ only)", ha="right",
        va="top", fontsize=7.5, color=DPINK)
ax.text(m.index[-1], m["mean"].iloc[-1] + 0.035, "CASS", ha="right",
        fontsize=8.5, color=DBLUE, fontweight="bold")
ax.set_xlabel("dictionary size $T'$ (skills)")
ax.set_ylabel("held-out accuracy")
ax.set_xticks(list(m.index))
ax.set_ylim(0.3, None)
save("e5_scale")

# ---------------- Fig eps -> rho: binned strip ----------------
Rv = R.dropna(subset=["rho"]).copy()
Rv["rho_c"] = Rv["rho"].clip(-0.1, 1.5)
bins = Rv["eps"].quantile([0, 1 / 3, 2 / 3, 1.0]).values
labels = [f"low\n$\\varepsilon\\leq${bins[1]:.2f}",
          f"mid\n{bins[1]:.2f}–{bins[2]:.2f}",
          f"high\n$\\varepsilon>${bins[2]:.2f}"]
Rv["bin"] = pd.cut(Rv["eps"], bins, labels=[0, 1, 2], include_lowest=True)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.grid(axis="y", zorder=0)
for b in [0, 1, 2]:
    s = Rv[Rv["bin"] == b]["rho_c"]
    jx = b + rng.uniform(-0.16, 0.16, len(s))
    ax.scatter(jx, s, s=20, color=BLUE, alpha=0.65, edgecolor="white",
               linewidth=0.7, zorder=2)
    mu, se = s.mean(), s.sem()
    ax.errorbar(b + 0.32, mu, yerr=1.96 * se, fmt="D", color=DPINK,
                markersize=6, capsize=3, elinewidth=1.6, capthick=1.6,
                markeredgecolor="white", markeredgewidth=1, zorder=4)
    ax.text(b + 0.44, mu, f"{mu:.2f}", fontsize=7.5, color=DPINK,
            va="center")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"oracle recovery $\rho$")
ax.set_xlabel(r"coding residual $\varepsilon$ (terciles)")
ax.set_xlim(-0.5, 2.75)
save("e1_eps_vs_rho")

# ---------------- Fig E2: heatmap + execution panel ----------------
cm = json.load(open(out / "e2_coeff_matrix.json"))
compounds = list(cm)
tasks_all = sorted(ALL_TASKS, key=lambda t: (TASK_REGISTRY[t][1], t))
M = np.array([[cm[c][t] for t in tasks_all] for c in compounds])
M = M / (M.max(axis=1, keepdims=True) + 1e-12)
truth = {comp for c in compounds for comp in compound_components(c)}
keep = [j for j, t in enumerate(tasks_all)
        if M[:, j].max() >= 0.2 or t in truth]
tasks = [tasks_all[j] for j in keep]
Mx = M[:, keep]
n_dropped = len(tasks_all) - len(keep)

e2 = pd.read_csv(out / "e2_compound.csv")
agg = e2.groupby("compound").agg(cass=("acc_cass", "mean"),
                                 retr=("acc_retrieval", "mean"),
                                 naive=("acc_naive", "mean"),
                                 icl=("acc_icl", "first"))

fig = plt.figure(figsize=(9.2, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.04)
axH = fig.add_subplot(gs[0])
axE = fig.add_subplot(gs[1], sharey=axH)

axH.imshow(Mx, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
for i, c in enumerate(compounds):
    for comp in compound_components(c):
        if comp in tasks:
            j = tasks.index(comp)
            axH.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=DPINK, lw=1.3, zorder=4))
axH.set_xticks(range(len(tasks)))
axH.set_xticklabels([short(t) for t in tasks], rotation=40, fontsize=6.4,
                    ha="right", rotation_mode="anchor")
axH.set_yticks(range(len(compounds)))
axH.set_yticklabels([" $\\circ$ ".join(short(p) for p in c.split("+"))
                     for c in compounds], fontsize=7)
prev = None
for j, t in enumerate(tasks):
    f = TASK_REGISTRY[t][1]
    if f != prev and prev is not None:
        axH.axvline(j - 0.5, color="white", lw=2)
    prev = f
axH.set_title(f"A  skill identification ({n_dropped} near-zero columns omitted)", fontsize=8, loc="left", color=INK,
              fontweight="bold")

# execution panel: per-compound accuracy, shared rows
y = np.arange(len(compounds))
for i, c in enumerate(compounds):
    a = agg.loc[c]
    axE.plot([0, max(a["icl"], a["cass"], a["retr"], a["naive"])],
             [i, i], color=GRID, lw=0.8, zorder=1)
axE.scatter(agg.loc[compounds, "icl"], y, marker="|", s=90, color=INK,
            lw=1.4, label="10-shot ICL", zorder=3)
axE.scatter(agg.loc[compounds, "naive"], y, marker="x", s=22, color=MUTED,
            label="naive", zorder=3)
axE.scatter(agg.loc[compounds, "retr"], y, s=32, color=PINK,
            edgecolor="white", linewidth=0.8, label="retrieval", zorder=4)
axE.scatter(agg.loc[compounds, "cass"], y, s=42, color=DBLUE,
            edgecolor="white", linewidth=0.9, label="CASS", zorder=5)
axE.tick_params(labelleft=False)
axE.set_xlim(-0.03, 1.05)
axE.set_xlabel("compound accuracy (case-sensitive)", fontsize=7.5)
axE.grid(axis="x", zorder=0)
axE.legend(fontsize=6.3, loc="lower right", handletextpad=0.15,
           borderaxespad=0.1)
axE.set_title("B  compound execution", fontsize=8, loc="left", color=INK,
              fontweight="bold")
axE.invert_yaxis() if False else None
handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=DPINK, lw=1.6)]
axH.legend(handles, ["ground-truth constituent"], loc="lower left",
           bbox_to_anchor=(0.0, 1.12), fontsize=6.5)
save("e2_heatmap")

# ---------------- Fig cosine matrices (H1) ----------------
from cass.dictionary import build_dictionary, pairwise_cosine
from cass.extract import load_G
G16 = {t: load_G(MODEL, t, 16).numpy() for t in tasks_all}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
for ax_, r0, title in [(axes[0], 0, "before removal ($r_0{=}0$)"),
                       (axes[1], 1, "after removal ($r_0{=}1$)")]:
    D = build_dictionary(G16, r0=r0)
    C = pairwise_cosine(D.anchors)
    im = ax_.imshow(C, cmap=DIV, vmin=-1, vmax=1)
    ax_.set_title(title, fontsize=8.5, color=INK)
    ax_.set_xticks([])
    ax_.set_yticks([])
    prev = None
    fams = []
    for j, t in enumerate(tasks_all):
        f = TASK_REGISTRY[t][1]
        if f != prev:
            if prev is not None:
                ax_.axvline(j - 0.5, color="white", lw=1.4)
                ax_.axhline(j - 0.5, color="white", lw=1.4)
            fams.append((f, j))
            prev = f
    fams.append((None, len(tasks_all)))
    if r0 == 0:
        for (f, a), (_, b) in zip(fams[:-1], fams[1:]):
            ax_.text(-1.2, (a + b - 1) / 2, f, ha="right", va="center",
                     fontsize=6.5, color=MUTED, style="italic")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label("pairwise cosine of task means", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)
cb.outline.set_visible(False)
save("cosine_matrix")
print("figures v3 (png only) written to", FIGURES_DIR)
