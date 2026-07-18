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


# ------------- Fig E1: dumbbell + k-trend + cross-model forest -------------
R = pd.read_csv(out / "e1_summary.csv")
R4 = R[R["k"] == 4].sort_values(["family", "task"]).reset_index(drop=True)
e1 = pd.read_csv(out / "e1_loto.csv")
bl = json.load(open(out / "baselines.json"))

fig = plt.figure(figsize=(9.2, 3.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.12], hspace=0.52,
                      wspace=0.34)
axA1 = fig.add_subplot(gs[:, 0])
axA2 = fig.add_subplot(gs[:, 1])
axB = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 2])

half = int(np.ceil(len(R4) / 2))
zv = e1[(e1["mode"] == "zvec") & (e1["k"] == 4) &
        (e1["seed"] < 3)].groupby("task")["acc"].mean()
for ax, chunk in zip([axA1, axA2], [R4.iloc[:half], R4.iloc[half:]]):
    chunk = chunk.reset_index(drop=True)
    n = len(chunk)
    ax.grid(axis="x", zorder=0)
    prev, start = None, 0
    for i, fam in enumerate(list(chunk["family"]) + [None]):
        if fam != prev:
            if prev is not None:
                ax.text(1.02, n - 1 - start + 0.45, prev, ha="right",
                        va="center", fontsize=6.4, color=MUTED,
                        style="italic", zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=0.6, alpha=0.85))
                if fam is not None:
                    ax.axhline(n - 1 - i + 0.5, color="#e3e3e3", lw=0.8,
                               zorder=0)
            prev, start = fam, i
    for i, row in chunk.iterrows():
        yv = n - 1 - i
        lo, hi = sorted([row["acc_syn"], row["acc_oracle"]])
        ax.plot([lo, hi], [yv, yv], color="#c4d8ea", lw=2.2, zorder=2,
                solid_capstyle="round")
        z0 = zv.get(row["task"], np.nan)
        if np.isfinite(z0):
            ax.plot(z0, yv, "o", markersize=4.0, markerfacecolor="white",
                    markeredgecolor=MUTED, markeredgewidth=1.1, zorder=3)
        ax.plot(row["acc_oracle"], yv, "o", color=PINK, markersize=5.6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.plot(row["acc_syn"], yv, "o", color=DBLUE, markersize=5.6,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        ax.plot([row["acc_icl"]] * 2, [yv - 0.34, yv + 0.34], color=INK,
                lw=1.0, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([short(t) for t in chunk["task"]][::-1], fontsize=6.8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.02, 1.04)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("accuracy", fontsize=7.5)
handles = [
    Line2D([0], [0], marker="o", color=DBLUE, lw=0, markersize=5.6,
           markeredgecolor="white"),
    Line2D([0], [0], marker="o", lw=0, markersize=4.0,
           markerfacecolor="white", markeredgecolor=MUTED),
    Line2D([0], [0], marker="o", color=PINK, lw=0, markersize=5.6,
           markeredgecolor="white"),
    Line2D([0], [0], marker="|", color=INK, lw=0, markersize=7,
           markeredgewidth=1.2),
]
axA2.legend(handles, ["CASS (k=4)", "$z$ only", "oracle", "10-shot ICL"],
            ncol=4, loc="lower left", bbox_to_anchor=(-0.30, 1.005),
            fontsize=6.8, handlelength=0.9, columnspacing=0.7,
            handletextpad=0.3, frameon=False)
axA1.set_title("(a)  per-task accuracy", fontsize=8, loc="left",
               color=INK, fontweight="bold", pad=10)

# B: median rho vs k with bootstrap CI band for CASS
def rho_tasks(mode, k, seeds=3):
    orc = e1[e1["mode"] == "oracle"].set_index("task")["acc"]
    acc = e1[(e1["mode"] == mode) & (e1["seed"] < seeds) &
             (e1["k"] == k)].groupby("task")["acc"].mean()
    return np.array([(acc[t] - bl[t]["zs"]) / (orc[t] - bl[t]["zs"])
                     for t in acc.index
                     if orc.get(t, 0) - bl[t]["zs"] > 0.05])

rngb = np.random.default_rng(0)
for mode, col, lab in [("cass", DBLUE, "CASS"), ("zvec", MUTED, "$z$ only"),
                       ("cass_recon", DPINK, "reconstruction")]:
    ks, med, lo, hi = [], [], [], []
    for k in [1, 2, 4]:
        r = rho_tasks(mode, k)
        boots = [np.median(rngb.choice(r, len(r), True)) for _ in range(2000)]
        ks.append(k); med.append(np.median(r))
        lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    axB.fill_between(ks, lo, hi, color=col, alpha=0.14 if mode == "cass"
                     else 0.09, lw=0)
    axB.plot(ks, med, color=col, lw=2, marker="o", markersize=5,
             markeredgecolor="white", markeredgewidth=1.1,
             solid_capstyle="round")
    axB.annotate(lab, (ks[-1], med[-1]), xytext=(5, 0),
                 textcoords="offset points", fontsize=6.6, color=col,
                 va="center")
axB.grid(axis="y", zorder=0)
axB.set_xticks([1, 2, 4])
axB.set_xlim(0.7, 6.3)
axB.set_xlabel("examples $k$", fontsize=7.5, labelpad=1)
axB.set_ylabel(r"median $\rho$", fontsize=7.5)
axB.set_title("(b)  recovery vs. $k$ (band: 95% CI)", fontsize=8, loc="left",
              color=INK, fontweight="bold")

# D: forest plot of dictionary gain across models
models = [("Llama-3.1-8B", 4.9, 1.4, 8.6), ("Llama-3.2-3B", 6.0, 2.9, 9.5),
          ("Gemma-2-2B", 4.2, 2.0, 6.7), ("Qwen2.5-3B", 1.4, 0.2, 2.8)]
for yi, (name, mgain, lo, hi) in enumerate(models):
    axD.plot([lo, hi], [yi, yi], color=BLUE, lw=2.2, zorder=2,
             solid_capstyle="round")
    axD.plot(mgain, yi, "D", color=DBLUE, markersize=6.5,
             markeredgecolor="white", markeredgewidth=1.1, zorder=3)
    axD.text(hi + 0.35, yi, f"+{mgain:.1f}", fontsize=7, color=DBLUE,
             va="center")
axD.axvline(0, color=DPINK, lw=1.1, ls=(0, (4, 3)), zorder=1)
axD.set_yticks(range(len(models)))
axD.set_yticklabels([m[0] for m in models], fontsize=7)
axD.set_xlabel("dictionary gain over $z$-only (acc. pts, 95% CI)",
               fontsize=7.5)
axD.set_xlim(-1.2, 11.5)
axD.invert_yaxis()
axD.grid(axis="x", zorder=0)
axD.set_title("(c)  dictionary gain across models", fontsize=8, loc="left",
              color=INK, fontweight="bold")
save("e1_main")




# ---------------- scale & eps panels (embedded in Fig E2) ----------------
d5 = pd.read_csv(out / "e5_scale_ext.csv" if (out / "e5_scale_ext.csv"
                 ).exists() else out / "e5_scale.csv")
_g5 = d5.groupby(["size", "draw"])["acc"].mean().reset_index()
_m5 = _g5.groupby("size")["acc"].agg(["mean", "sem"])
_e1s = pd.read_csv(out / "e1_loto.csv")
_held = d5["task"].unique()
_zref = _e1s[(_e1s["mode"] == "zvec") & (_e1s["k"] == 4) &
             (_e1s["task"].isin(_held))]["acc"].mean()

def plot_scale(ax):
    m = _m5
    ax.grid(axis="y", zorder=0)
    ax.fill_between(m.index, m["mean"] - 1.96 * m["sem"],
                    m["mean"] + 1.96 * m["sem"], color=BLUE, alpha=0.16,
                    lw=0)
    ax.plot(m.index, m["mean"], color=DBLUE, lw=2, zorder=3,
            solid_capstyle="round")
    ax.plot(m.index, m["mean"], "o", color=DBLUE, markersize=4.5,
            markeredgecolor="white", markeredgewidth=1.1, zorder=4)
    ax.axhline(_zref, color=DPINK, lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.text(m.index[-1], _zref - 0.022, "no dict.", ha="right", va="top",
            fontsize=6.4, color=DPINK)
    ax.text(m.index[-1], m["mean"].iloc[-1] + 0.028, "CASS", ha="right",
            fontsize=7.2, color=DBLUE, fontweight="bold")
    ax.set_xlabel("dictionary size $T'$", fontsize=7)
    ax.set_ylabel("held-out acc.", fontsize=7)
    ax.set_xticks([int(m.index[0]), 20, int(m.index[-1])])
    ax.tick_params(labelsize=6.2)
    ax.set_ylim(0.3, 0.6)

def plot_eps(ax):
    R5 = pd.read_csv(out / "e1_summary.csv")
    Rv = R5[R5["k"] == 4].dropna(subset=["rho"]).copy()
    Rv["rho_c"] = Rv["rho"].clip(-0.1, 1.5)
    bins = Rv["eps"].quantile([0, 1 / 3, 2 / 3, 1.0]).values
    Rv["bin"] = pd.cut(Rv["eps"], bins, labels=[0, 1, 2],
                       include_lowest=True)
    rng5 = np.random.default_rng(0)
    ax.grid(axis="y", zorder=0)
    for b in [0, 1, 2]:
        sv = Rv[Rv["bin"] == b]["rho_c"]
        jx = b + rng5.uniform(-0.16, 0.16, len(sv))
        ax.scatter(jx, sv, s=14, color=BLUE, alpha=0.6, edgecolor="white",
                   linewidth=0.6, zorder=2)
        mu, se = sv.mean(), sv.sem()
        ax.errorbar(b + 0.3, mu, yerr=1.96 * se, fmt="D", color=DPINK,
                    markersize=4.5, capsize=2.5, elinewidth=1.3,
                    capthick=1.3, markeredgecolor="white",
                    markeredgewidth=1, zorder=4)
        ax.text(b + 0.42, mu, f"{mu:.2f}", fontsize=6.2, color=DPINK,
                va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["low", "mid", "high"], fontsize=6.4)
    ax.set_ylabel(r"oracle recovery $\rho$", fontsize=7)
    ax.set_xlabel(r"residual $\varepsilon$ (terciles)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.set_xlim(-0.5, 2.9)

# -------- Fig E2: heatmap + identification quality + execution ----------
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

fig = plt.figure(figsize=(10.6, 2.44))
gs = fig.add_gridspec(2, 5, width_ratios=[1.62, 0.38, 1.02, 0.62, 0.62],
                      height_ratios=[0.20, 1.0], wspace=0.42, hspace=0.10)
axH = fig.add_subplot(gs[1, 0])
axI = fig.add_subplot(gs[1, 1], sharey=axH)
axE = fig.add_subplot(gs[1, 2], sharey=axH)
axS = fig.add_subplot(gs[1, 3])
axP = fig.add_subplot(gs[1, 4])
axT = fig.add_subplot(gs[0, 0], sharex=axH)

axH.imshow(Mx, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
for i, c in enumerate(compounds):
    for comp in compound_components(c):
        if comp in tasks:
            j = tasks.index(comp)
            axH.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=DPINK, lw=1.4, zorder=4))
    for j in range(len(tasks)):
        if Mx[i, j] >= 0.30:
            axH.text(j, i, f"{Mx[i, j]:.1f}".lstrip("0"), ha="center",
                     va="center", fontsize=5.8,
                     color="white" if Mx[i, j] > 0.65 else INK)
axH.set_xticks(range(len(tasks)))
axH.set_xticklabels([short(t) for t in tasks], rotation=40, fontsize=7.0,
                    ha="right", rotation_mode="anchor")
axH.set_yticks(range(len(compounds)))
axH.set_yticklabels([" $\\circ$ ".join(short(p) for p in c.split("+"))
                     for c in compounds], fontsize=7.6)
prev = None
for j, t in enumerate(tasks):
    f = TASK_REGISTRY[t][1]
    if f != prev and prev is not None:
        axH.axvline(j - 0.5, color="white", lw=2)
    prev = f
col_mass = Mx.sum(axis=0)
axT.bar(range(len(tasks)), col_mass,
        color=[DPINK if t in truth else BLUE for t in tasks],
        width=0.7, zorder=2)
axT.set_ylabel("reuse", fontsize=6.2)
axT.set_yticks([])
axT.tick_params(labelbottom=False, length=0)
for sp in ["top", "right", "left"]:
    axT.spines[sp].set_visible(False)
axT.set_title("(a)  skill identification (top: total coefficient mass "
              "per skill)", fontsize=8, loc="left", color=INK,
              fontweight="bold")
handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=DPINK, lw=1.4)]
axT.legend(handles, ["ground-truth constituent"], loc="upper right",
           fontsize=6.5, frameon=False, borderaxespad=0.0)

# B: identification quality = coefficient mass on true constituents
y = np.arange(len(compounds))
mass = []
for c in compounds:
    comps = compound_components(c)
    tot = sum(cm[c].values()) + 1e-12
    mass.append(sum(cm[c].get(x, 0) for x in comps) / tot)
top1 = [max(cm[c], key=cm[c].get) in compound_components(c)
        for c in compounds]
axI.barh(y, mass, height=0.62, color=BLUE, edgecolor="none", zorder=2)
for yi, (m, t1) in enumerate(zip(mass, top1)):
    if t1:
        axI.plot(m + 0.07, yi, "o", color=DPINK, markersize=4, zorder=3)
axI.set_xlim(0, 1.05)
axI.set_xticks([0, 0.5, 1])
axI.tick_params(labelleft=False, labelsize=6.5)
axI.grid(axis="x", zorder=0)
axI.set_xlabel("coeff.\ mass on\ntrue constituents", fontsize=7.2)
axI.set_title("(b)", fontsize=8, loc="left", color=INK, fontweight="bold")

# C: CASS as bars (panel-B language); other methods as reference marks
axE.barh(y, agg.loc[compounds, "cass"], height=0.6, color=BLUE,
         edgecolor=DBLUE, linewidth=0.8, label="CASS", zorder=3)
axE.scatter(agg.loc[compounds, "naive"], y, marker="x", s=24, color=MUTED,
            label="naive", zorder=4)
axE.scatter(agg.loc[compounds, "retr"], y, s=26, color=PINK,
            edgecolor="white", linewidth=0.8, label="retrieval", zorder=4)
hc = pd.read_csv(out / "hendel_compound.csv").groupby("compound")["acc"] \
    .mean()
axE.scatter([hc.get(c, np.nan) for c in compounds], y, marker="s", s=26,
            color="#b8a1d9", edgecolor="white", linewidth=0.8,
            label="replace", zorder=4)
axE.scatter(agg.loc[compounds, "icl"], y, marker="|", s=110, color=INK,
            lw=1.4, label="ICL ceiling", zorder=4)
axE.tick_params(labelleft=False)
axE.set_xlim(-0.04, 1.05)
axE.set_xlabel("compound accuracy (case-sensitive)", fontsize=7.6)
axE.grid(axis="x", zorder=0)
axE.legend(ncol=3, frameon=False, fontsize=6.2, loc="lower left",
           mode="expand", bbox_to_anchor=(0.0, 1.005, 1.0, 0.12),
           handletextpad=0.25, columnspacing=0.6, borderaxespad=0.0)
axE.set_title("(c)  execution", fontsize=8, loc="left",
              color=INK, fontweight="bold", pad=25)
plot_scale(axS)
axS.set_title("(d)  dictionary scaling", fontsize=8, loc="left",
              color=INK, fontweight="bold", pad=25)
plot_eps(axP)
axP.set_title("(e)  $\\varepsilon$ vs.\\ recovery", fontsize=8,
              loc="left", color=INK, fontweight="bold", pad=25)
save("e2_heatmap")

# ---------------- Fig ablation sweeps (Table-2 companion) ----------------
d4 = pd.read_csv(out / "e4_ablations.csv")
rr = pd.read_csv(out / "review_response.csv")
rnd = rr[rr.exp == "control"]["acc"].mean()

def ax_vals(axis, vals):
    return [d4[(d4.axis == axis) & (d4.value == str(v))]["acc"].mean()
            for v in vals]

panels = [("r0", [0, 1, 2, 4, 8, 16], "shared rank $r_0$"),
          ("k", [1, 2, 4, 8], "examples $k$"),
          ("smax", [3, 5, 8, 31], "support cap $s_{\\max}$"),
          ("n", [25, 50, 100], "samples per skill $n$")]
fig, axs = plt.subplots(2, 2, figsize=(2.05, 1.82), sharey=True)
for ax, (axis, vals, title) in zip(axs.ravel(), panels):
    ys = ax_vals(axis, vals)
    xs = range(len(vals))
    ax.grid(axis="y", zorder=0, lw=0.5)
    ax.plot(xs, ys, color=DBLUE, lw=1.7, zorder=3, solid_capstyle="round")
    ax.plot(xs, ys, "o", color=DBLUE, markersize=3.4,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    if axis == "r0":
        ax.plot(1, ys[1], "o", markersize=7, markerfacecolor="none",
                markeredgecolor=DPINK, markeredgewidth=1.2, zorder=5)
        ax.axhline(rnd, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(len(vals) - 1, rnd - 0.035, "random dir.", ha="right",
                va="top", fontsize=5.4, color=MUTED)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(v) for v in vals], fontsize=5.8)
    ax.tick_params(axis="y", labelsize=5.8, length=2)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.set_title(title, fontsize=6.4, pad=2, color=INK)
    ax.set_ylim(0, 0.64)
    ax.set_yticks([0, 0.3, 0.6])
for ax in axs[:, 0]:
    ax.set_ylabel("acc.", fontsize=6)
fig.subplots_adjust(hspace=0.62, wspace=0.12)
save("e4_sweeps")

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
