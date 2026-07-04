"""Fig 2: three-module pipeline schematic (matplotlib, blue/pink palette)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from cass.config import FIGURES_DIR

BLUE, PINK, DBLUE, DPINK = "#5aa7e0", "#f2aac9", "#2f6fb3", "#cf5a92"
INK, MUTED = "#3b3b3b", "#8a8a8a"
PANEL = "#f6f9fc"

fig, ax = plt.subplots(figsize=(7.4, 2.7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")


def panel(x, w, title):
    ax.add_patch(FancyBboxPatch((x, 2), w, 30,
                 boxstyle="round,pad=0.6,rounding_size=1.6",
                 facecolor=PANEL, edgecolor="#dde6ee", lw=1))
    ax.text(x + 1.5, 29.4, title, ha="left", fontsize=7.5,
            color=INK, fontweight="bold")


def arrow(x0, x1, y=17):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=13, color=MUTED, lw=1.4))


# ---- Module 1: mining ----
panel(1, 30, "1  Mine skill subspaces (once)")
rng = np.random.default_rng(3)
for i, c in enumerate([BLUE, DPINK, "#8fc78f", "#c9a35a"]):
    y = 23.5 - i * 5.0
    ax.text(2.5, y, f"task {i+1} ±prompts", fontsize=5.9, color=INK,
            va="center")
    ax.annotate("", xy=(15.6, y), xytext=(14.1, y),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(16.6, y, "Δh", fontsize=6.5, color=c, va="center",
            style="italic")
ax.add_patch(FancyBboxPatch((21.5, 5), 8.2, 21.5,
             boxstyle="round,pad=0.4,rounding_size=1",
             facecolor="white", edgecolor=MUTED, lw=0.8))
ax.text(25.6, 23.6, "dictionary", fontsize=6.5, ha="center", color=MUTED)
for i, c in enumerate([BLUE, DPINK, "#8fc78f", "#c9a35a"]):
    y0 = 19.4 - i * 4.3
    ax.add_patch(plt.Rectangle((22.6, y0), 6, 2.6, facecolor=c, alpha=0.5,
                               edgecolor="none"))
    ax.text(25.6, y0 + 1.3, f"$U_{{{i+1}}},\\ \\mu_{{{i+1}}}$", fontsize=6,
            ha="center", va="center", color=INK)
ax.text(16, 3.6, "shared component $U_0$ removed", fontsize=6.5,
        color=DPINK, ha="center", style="italic")

arrow(31.6, 36.4)

# ---- Module 2: coding ----
panel(37, 28, "2  Code a novel task")
ax.text(38.5, 25.2, "$k$ examples → denoised $z$", fontsize=6.8,
        color=DBLUE)
ax.text(38.5, 20.5, "group LASSO:\nsupport $S$, code $c$,\nresidual"
        " $\\varepsilon$", fontsize=6.6, color=INK, va="top")
# coefficient stem plot
bx, bw = 52.5, 11
ax.plot([bx, bx + bw], [8.5, 8.5], color=MUTED, lw=0.8)
coefs = [0, 0.9, 0, 0, 0.5, 0, 0.25, 0]
cols = [BLUE if c > 0 else "#cccccc" for c in coefs]
for i, (c, col) in enumerate(zip(coefs, cols)):
    x = bx + 0.8 + i * 1.32
    if c > 0:
        ax.plot([x, x], [8.5, 8.5 + c * 9], color=DBLUE, lw=2,
                solid_capstyle="round")
        ax.plot(x, 8.5 + c * 9, "o", color=DBLUE, markersize=3.5)
    else:
        ax.plot(x, 8.5, "o", color="#cccccc", markersize=2)
ax.text(bx + bw / 2, 4.5, "sparse skill code", fontsize=6.5, ha="center",
        color=MUTED)
ax.text(51, 22.2, "$z \\approx \\sum_{t\\in S} U_t c_t + $ res.",
        fontsize=6.8, color=INK)

arrow(65.6, 70.4)

# ---- Module 3: serving policy + injection ----
panel(71, 28, "3  Serve queries (routing + steering)")
# signal branch
ax.text(72.3, 24.6, "$\\|z\\|$\nstrong?", fontsize=6.2, color=INK, va="center")
ax.annotate("", xy=(83.5, 22.2), xytext=(79, 24.2),
            arrowprops=dict(arrowstyle="-|>", color=DBLUE, lw=1.2))
ax.annotate("", xy=(83.5, 9.2), xytext=(79, 23.6),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
ax.text(81.6, 22.6, "yes", fontsize=5.6, color=DBLUE)
ax.text(79.4, 15.6, "no", fontsize=5.6, color=MUTED)
# compose box
ax.add_patch(FancyBboxPatch((84, 17.5), 14, 8.6,
             boxstyle="round,pad=0.4,rounding_size=1",
             facecolor="white", edgecolor=DBLUE, lw=1.0))
ax.text(91, 24.2, "compose (layers 12, 16)", fontsize=5.8, ha="center",
        color=DBLUE, fontweight="bold")
ax.text(91, 21.2, "$h \\leftarrow h + \\tilde\\alpha\\gamma\\Delta"
        " + g\\alpha P_S(\\mu_S{-}h)$", fontsize=5.6, ha="center",
        color=INK)
ax.text(91, 18.9, "trust gate $g$", fontsize=5.4, ha="center",
        color=DPINK, style="italic")
# replace box
ax.add_patch(FancyBboxPatch((84, 6.5), 14, 5.4,
             boxstyle="round,pad=0.4,rounding_size=1",
             facecolor="white", edgecolor=MUTED, lw=0.8))
ax.text(91, 10.4, "compress", fontsize=5.8, ha="center", color=INK,
        fontweight="bold")
ax.text(91, 8.2, "$h \\leftarrow \\bar v_{\\mathrm{prompt}}$",
        fontsize=5.8, ha="center", color=INK)
ax.text(84.5, 3.6, "$\\varepsilon$ large → escalate to full ICL",
        fontsize=6.0, color=MUTED, ha="left", style="italic")

for d in [FIGURES_DIR, FIGURES_DIR.parent / "paper" / "figures"]:
    plt.savefig(d / "pipeline.png", dpi=220, bbox_inches="tight")
plt.close()
print("pipeline figure saved")
