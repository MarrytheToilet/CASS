"""Anchor-blend (conceptor-style) baseline on all 32 LOTO tasks."""
import csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cass.config import results_dir
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z, _support_weights
from cass.tasks import ALL_TASKS, load_task, zs_prompt
from cass.zcache import get_z

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}
rows_path = out / "improvements.csv"
done = set()
with open(rows_path) as f:
    done = {(r["exp"], r["cond"], r["task"], r["seed"])
            for r in csv.DictReader(f)}
fout = open(rows_path, "a", newline="")
w = csv.DictWriter(fout, fieldnames=["exp", "cond", "task", "seed", "acc"])
for tstar in ALL_TASKS:
    task = load_task(tstar)
    D = build_multilayer_dictionary(
        {l: {t: G[l][t] for t in ALL_TASKS if t != tstar} for l in LAYERS},
        r0=1)
    for seed in [0, 1, 2]:
        if ("baseline", "blend", tstar, str(seed)) in done:
            continue
        Z = get_z(hlm, task, K, seed)
        z_list = z_list_from_Z(D, Z)
        code = code_for(D, z_list)
        if code.support:
            ws = _support_weights(code)
            blend = sum(wi * D.anchors[n] for wi, n in zip(ws, code.support))
        else:
            blend = np.mean(z_list, axis=0)
        ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=blend)
        q = task.eval_queries[:50]
        acc = accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                    batch_size=25, op=ops, layer=lys),
                       [y for _, y in q])
        w.writerow(dict(exp="baseline", cond="blend", task=tstar, seed=seed,
                        acc=acc))
        fout.flush()
    print(tstar, "done", flush=True)
fout.close()
print("BLEND DONE")
