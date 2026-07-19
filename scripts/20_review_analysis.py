"""Offline analyses for review-response (no GPU).

(1) z-only + routing baseline: the same ||z|| routing rule applied to the
    z-only steering arm, isolating the dictionary correction's net
    contribution from the contribution of knowing when to fall back.
(2) Dictionary gain conditional on the compose route (paired, bootstrap CI).
(3) Bootstrap CI for median oracle recovery rho (task-level resampling).
(4) Novel-15 breakdown by relation to the dictionary
    (inverse / same-family / no-counterpart).
-> results/llama31-8b/review2_analysis.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from cass.dictionary import build_multilayer_dictionary
from cass.extract import load_G
from cass.pipeline import z_list_from_Z
from cass.tasks import ALL_TASKS

LAYERS = [12, 16]
OUT = Path("results/llama31-8b")
rng = np.random.default_rng(0)

G = {l: {t: load_G("llama31-8b", t, l).numpy() for t in ALL_TASKS}
     for l in LAYERS}
D = build_multilayer_dictionary(G, r0=1)


def znorm(name):
    ns = []
    for seed in [0, 1, 2]:
        Z = torch.load(OUT / "zcache" / f"{name}_k4_s{seed}.pt",
                       map_location="cpu", weights_only=False).float()
        ns.append(np.linalg.norm(np.mean(z_list_from_Z(D, Z), axis=0)))
    return float(np.mean(ns))


e1 = pd.read_csv(OUT / "e1_loto.csv")
lit = pd.read_csv(OUT / "baselines_lit.csv")
base = e1[e1["k"] == 4]
df = base[base["mode"] == "cass"].groupby("task").agg(
    cass=("acc", "mean"), znorm=("delta_norm", "mean"))
df = df.join(base[base["mode"] == "zvec"].groupby("task")["acc"]
             .mean().rename("zvec"))
df = df.join(lit[(lit.suite == "loto") & (lit.method == "hendel_replace")]
             .groupby("task")["acc"].mean().rename("hendel"))

ths = np.linspace(1.0, 8.0, 57)


def nested_route(arm):
    routed = []
    for t in df.index:
        rest = df.drop(t)
        accs = [np.where(rest["znorm"] < th, rest["hendel"],
                         rest[arm]).mean() for th in ths]
        th_star = ths[int(np.argmax(accs))]
        r = df.loc[t]
        routed.append(r["hendel"] if r["znorm"] < th_star else r[arm])
    return float(np.mean(routed))


def global_th(arm):
    return float(ths[int(np.argmax(
        [np.where(df["znorm"] < th, df["hendel"], df[arm]).mean()
         for th in ths]))])


res = {}
res["loto"] = {
    "cass_pure": round(df["cass"].mean(), 3),
    "zvec_pure": round(df["zvec"].mean(), 3),
    "replace_pure": round(df["hendel"].mean(), 3),
    "cass_routed_nested": round(nested_route("cass"), 3),
    "zvec_routed_nested": round(nested_route("zvec"), 3),
}
thc, thz = global_th("cass"), global_th("zvec")
res["loto"]["frozen_th_cass"] = thc
res["loto"]["frozen_th_zvec"] = thz

# (2) paired dictionary gain on the compose-routed subset (frozen th)
comp = df[df["znorm"] >= thc]
diff = (comp["cass"] - comp["zvec"]).values
boots = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
res["compose_subset"] = {
    "n_tasks": len(comp),
    "cass": round(comp["cass"].mean(), 3),
    "zvec": round(comp["zvec"].mean(), 3),
    "paired_diff": round(diff.mean(), 3),
    "ci95": [round(np.percentile(boots, 2.5), 3),
             round(np.percentile(boots, 97.5), 3)],
}

# novel suite, frozen thresholds from LOTO (each arm its own)
e7 = pd.read_csv(OUT / "e7_novel.csv")
c7 = e7[e7["mode"] == "cass"].groupby("task")["acc"].mean()
z7 = e7[e7["mode"] == "zvec"].groupby("task")["acc"].mean()
h7 = lit[(lit.suite == "novel") & (lit.method == "hendel_replace")] \
    .groupby("task")["acc"].mean()
zn7 = {t: znorm(t) for t in c7.index}
res["novel"] = {
    "cass_pure": round(c7.mean(), 3),
    "zvec_pure": round(z7.mean(), 3),
    "replace_pure": round(h7.mean(), 3),
    "cass_routed": round(np.mean([h7[t] if zn7[t] < thc else c7[t]
                                  for t in c7.index]), 3),
    "zvec_routed": round(np.mean([h7[t] if zn7[t] < thz else z7[t]
                                  for t in c7.index]), 3),
}

# (3) bootstrap CI for median rho (k=4, steerable tasks)
R = pd.read_csv(OUT / "e1_summary.csv")
rho = R[(R["k"] == 4)].dropna(subset=["rho"])["rho"].values
med = [np.median(rng.choice(rho, len(rho))) for _ in range(10000)]
res["rho_median_k4"] = {
    "median": round(float(np.median(rho)), 3), "n": len(rho),
    "ci95": [round(np.percentile(med, 2.5), 3),
             round(np.percentile(med, 97.5), 3)],
}

# (4) novel breakdown by relation to the dictionary
CAT = {"capital-country": "inverse", "past-present": "inverse",
       "french-english": "inverse",
       "country-continent": "same-family", "element-symbol": "same-family",
       "animal-baby": "same-family", "fruit-color": "same-family",
       "profession-workplace": "same-family",
       "english-italian": "same-family", "english-portuguese": "same-family",
       "verb-gerund": "same-family", "adjective-comparative": "same-family",
       "adjective-adverb": "same-family",
       "word-last-letter": "no-counterpart",
       "number-plus-one": "no-counterpart"}
bk = {}
for cat in ["inverse", "same-family", "no-counterpart"]:
    ts = [t for t in c7.index if CAT[t] == cat]
    bk[cat] = {"n": len(ts), "cass": round(c7[ts].mean(), 3),
               "zvec": round(z7[ts].mean(), 3),
               "replace": round(h7[ts].mean(), 3)}
res["novel_breakdown"] = bk

json.dump(res, open(OUT / "review2_analysis.json", "w"), indent=2)
print(json.dumps(res, indent=2))
