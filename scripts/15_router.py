"""CASS-R: signal-aware routing between composition (CASS) and prompt-state
compression (task-vector replacement).

Rule: use replacement when ||z|| < 0.9 x median dictionary anchor norm
(threshold selected by nested leave-one-task-out on LOTO, then frozen for
the novel and compound suites). Reproduces the numbers in Table 1:
LOTO 0.459 / Novel 0.499 / Compound 0.292.
Inputs: e1_loto.csv, baselines_lit.csv, hendel_compound.csv, e2_compound.csv,
zcache/. Pure post-processing; no GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from cass.dictionary import build_multilayer_dictionary
from cass.extract import load_G
from cass.pipeline import z_list_from_Z
from cass.tasks import ALL_TASKS, synthetic_tasks

LAYERS = [12, 16]
OUT = Path("results/llama31-8b")

G = {l: {t: load_G("llama31-8b", t, l).numpy() for t in ALL_TASKS}
     for l in LAYERS}
D = build_multilayer_dictionary(G, r0=1)
anc_med = np.median([np.linalg.norm(D.anchors[t]) for t in D.task_names])


def znorm(name):
    ns = []
    for seed in [0, 1, 2]:
        Z = torch.load(OUT / "zcache" / f"{name}_k4_s{seed}.pt",
                       map_location="cpu", weights_only=False).float()
        ns.append(np.linalg.norm(np.mean(z_list_from_Z(D, Z), axis=0)))
    return float(np.mean(ns))


e1 = pd.read_csv(OUT / "e1_loto.csv")
lit = pd.read_csv(OUT / "baselines_lit.csv")
c = e1[(e1["mode"] == "cass") & (e1["k"] == 4)].groupby("task").agg(
    cass=("acc", "mean"), znorm=("delta_norm", "mean"))
h = lit[(lit.suite == "loto") & (lit.method == "hendel_replace")] \
    .groupby("task")["acc"].mean().rename("hendel")
df = c.join(h)

ths = np.linspace(1.0, 8.0, 57)
routed = []
for t in df.index:                     # nested threshold selection
    rest = df.drop(t)
    accs = [np.where(rest["znorm"] < th, rest["hendel"],
                     rest["cass"]).mean() for th in ths]
    th_star = ths[int(np.argmax(accs))]
    r = df.loc[t]
    routed.append(r["hendel"] if r["znorm"] < th_star else r["cass"])
th_global = ths[int(np.argmax(
    [np.where(df["znorm"] < th, df["hendel"], df["cass"]).mean()
     for th in ths]))]
print(f"LOTO nested-routed: {np.mean(routed):.3f} "
      f"(cass {df['cass'].mean():.3f}, replace {df['hendel'].mean():.3f}, "
      f"oracle {df[['cass','hendel']].max(axis=1).mean():.3f})")
print(f"frozen threshold {th_global:.2f} = "
      f"{th_global/anc_med:.2f} x median anchor norm")

e7 = pd.read_csv(OUT / "e7_novel.csv")
c7 = e7[e7["mode"] == "cass"].groupby("task")["acc"].mean()
h7 = lit[(lit.suite == "novel") & (lit.method == "hendel_replace")] \
    .groupby("task")["acc"].mean()
picks = [h7[t] if znorm(t) < th_global else c7[t] for t in c7.index]
print(f"Novel routed (frozen th): {np.mean(picks):.3f} "
      f"(cass {c7.mean():.3f}, replace {h7.mean():.3f})")

e2 = pd.read_csv(OUT / "e2_compound.csv")
hc = pd.read_csv(OUT / "hendel_compound.csv")
c2 = e2.groupby("compound")["acc_cass"].mean()
h2 = hc.groupby("compound")["acc"].mean()
picks = [h2[n] if znorm(n) < th_global else c2[n] for n in c2.index]
print(f"Compound routed (frozen th): {np.mean(picks):.3f} "
      f"(cass {c2.mean():.3f}, replace {h2.mean():.3f})")
