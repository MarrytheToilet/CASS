"""Long-form probe: one mechanically-verifiable multi-token task, testing
whether composed steering survives longer generation (reviewer scope
question). Task ``cap-words``: capitalize every word of a 5--7-word phrase
(output ~6-9 tokens, exact match on the full string).

Inputs are built locally from the antonym/synonym word pools (no API).
Modes: zero-shot, 4-shot ICL, task-vector replacement, CASS (frozen
32-skill dictionary, standard k=4 pipeline). -> longform_probe.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import results_dir
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G, extract_fewshot_z
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z
from cass.steer import _to_torch
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
out = results_dir(MODEL)

# ---- build the task locally ----
words = sorted({x.lower() for t in ["antonym", "synonym"]
                for x, y in load_task(t).dict_pool + load_task(t).fewshot_pool
                if x.isalpha()})
rng = np.random.default_rng(0)
def make_pair(n_words):
    ws = list(rng.choice(words, n_words, replace=False))
    return " ".join(ws), " ".join(w.capitalize() for w in ws)
pool = [make_pair(int(n)) for n in rng.integers(5, 8, size=90)]
fewshot_pool, eval_queries = pool[:40], pool[40:]

hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}
D = build_multilayer_dictionary(G, r0=1)

res = {"task": "cap-words", "n_eval": len(eval_queries)}
targets = [y for _, y in eval_queries]
zs_prompts = [zs_prompt(x) for x, _ in eval_queries]
MNT = 24

# zero-shot
res["zs"] = accuracy(hlm.generate(zs_prompts, batch_size=16,
                                  max_new_tokens=MNT), targets,
                     case_sensitive=True)

accs = {"icl4": [], "cass": [], "replace": []}
supports = []
for seed in [0, 1, 2]:
    r2 = np.random.default_rng(100 * seed + K)
    idx = r2.choice(len(fewshot_pool), K, replace=False)
    ex = [fewshot_pool[i] for i in idx]
    # 4-shot ICL
    accs["icl4"].append(accuracy(
        hlm.generate([icl_prompt(ex, x) for x, _ in eval_queries],
                     batch_size=8, max_new_tokens=MNT),
        targets, case_sensitive=True))
    # replacement (mean k-shot prompt state overwrite)
    kp = [icl_prompt([e for i, e in enumerate(ex) if i != j], x)
          for j, (x, _) in enumerate(ex)]
    H = hlm.last_token_hiddens(kp, batch_size=8)
    def rep_op(vec):
        v = _to_torch(vec, "cuda")
        def op(h):
            return v.unsqueeze(0).expand_as(h).clone()
        return op
    ops = [rep_op(H[:, l].mean(0).numpy()) for l in LAYERS]
    accs["replace"].append(accuracy(
        hlm.generate(zs_prompts, batch_size=16, max_new_tokens=MNT,
                     op=ops, layer=list(LAYERS)), targets,
        case_sensitive=True))
    # CASS (frozen dictionary, standard pipeline)
    Z = extract_fewshot_z(hlm, ex, seed=seed, n_reps=6)
    zl = z_list_from_Z(D, Z)
    code = code_for(D, zl)
    supports.append(code.support[:5])
    ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=np.mean(zl, axis=0))
    accs["cass"].append(accuracy(
        hlm.generate(zs_prompts, batch_size=16, max_new_tokens=MNT,
                     op=ops, layer=lys), targets, case_sensitive=True))
    print(f"seed {seed}: icl4={accs['icl4'][-1]:.2f} "
          f"replace={accs['replace'][-1]:.2f} cass={accs['cass'][-1]:.2f} "
          f"support={supports[-1]}", flush=True)

for k, v in accs.items():
    res[k] = round(float(np.mean(v)), 3)
res["supports"] = supports
json.dump(res, open(out / "longform_probe.json", "w"), indent=2)
print("LONGFORM:", res)
