"""Build the skill dictionary at l* and produce H1 diagnostics:
(a) pairwise cosine of task mean vectors before/after common-component removal,
(b) per-task spectral decay, (c) block coherence mu_B vs r0.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cass.config import results_dir, FIGURES_DIR
from cass.dictionary import (build_dictionary, pairwise_cosine,
                             subcoherence_matrix)
from cass.extract import load_G
from cass.tasks import ALL_TASKS, TASK_REGISTRY

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else None
R0_SCAN = [0, 1, 2, 4, 8, 16]
R0_DEFAULT = 2

def main():
    out = results_dir(MODEL)
    FIGURES_DIR.mkdir(exist_ok=True)
    layer = LAYER or json.load(open(out / "layer_scan_summary.json"))["best_layer"]
    # order tasks by family for readable heatmaps
    tasks = sorted(ALL_TASKS, key=lambda t: (TASK_REGISTRY[t][1], t))
    G_by_task = {t: load_G(MODEL, t, layer).numpy() for t in tasks}

    diag = {"layer": layer}
    for r0 in R0_SCAN:
        D = build_dictionary(G_by_task, r0=r0)
        C = pairwise_cosine(D.anchors if r0 > 0 else D.raw_means)
        off = C[~np.eye(len(tasks), dtype=bool)]
        MB = subcoherence_matrix(D)
        mb_off = MB[~np.eye(len(tasks), dtype=bool)]
        ranks = [D.bases[t].shape[1] for t in tasks]
        diag[f"r0={r0}"] = dict(
            median_cos=float(np.median(np.abs(off))),
            mean_cos=float(np.mean(np.abs(off))),
            mu_B=float(mb_off.max()), median_mu=float(np.median(mb_off)),
            mean_rank=float(np.mean(ranks)), sum_rank=int(np.sum(ranks)),
        )
        print(f"r0={r0}: median|cos|={diag[f'r0={r0}']['median_cos']:.3f} "
              f"mu_B={diag[f'r0={r0}']['mu_B']:.3f} "
              f"median_muB={diag[f'r0={r0}']['median_mu']:.3f} "
              f"mean_rank={np.mean(ranks):.1f}")
        if r0 == R0_DEFAULT:
            pickle.dump(D, open(out / f"dictionary_l{layer}_r0{r0}.pkl", "wb"))
            # figures
            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
            D0 = build_dictionary(G_by_task, r0=0)
            for ax, DD, title in [
                    (axes[0], D0, "before common removal (r0=0)"),
                    (axes[1], D, f"after common removal (r0={r0})")]:
                CC = pairwise_cosine({t: DD.anchors[t] for t in tasks})
                sns.heatmap(CC, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                            xticklabels=tasks, yticklabels=tasks, square=True)
                ax.set_title(title)
                ax.tick_params(labelsize=5)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"cosine_matrix_{MODEL}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(7, 5))
            for t in tasks:
                s = D.spectra[t]
                plt.plot(np.arange(1, 21), (s[:20] ** 2) / (s ** 2).sum(),
                         alpha=0.5, lw=1)
            plt.yscale("log")
            plt.xlabel("component")
            plt.ylabel("fraction of spectral energy")
            plt.title(f"per-task spectral decay ({MODEL}, layer {layer})")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"spectra_{MODEL}.png", dpi=150)
            plt.close()

    json.dump(diag, open(out / "dictionary_diagnostics.json", "w"), indent=2)
    print(f"saved dictionary + diagnostics (layer {layer})")

if __name__ == "__main__":
    main()
