"""Latency micro-benchmark: steered vs plain generation throughput."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch
from cass.dictionary import build_multilayer_dictionary
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z
from cass.tasks import ALL_TASKS, load_task, zs_prompt, icl_prompt
from cass.zcache import get_z

LAYERS=[12,16]
hlm = HookedLM("llama31-8b")
G = {l: {t: load_G("llama31-8b", t, l).numpy() for t in ALL_TASKS} for l in LAYERS}
task = load_task("antonym")
D = build_multilayer_dictionary({l: {t: G[l][t] for t in ALL_TASKS if t != "antonym"} for l in LAYERS}, r0=1)
Z = get_z(hlm, task, 4, 0)
zl = z_list_from_Z(D, Z); zm = np.mean(zl, axis=0)
code = code_for(D, zl)
ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=zm)
q = [zs_prompt(x) for x,_ in task.eval_queries[:50]]
icl = [icl_prompt(task.dict_pool[:10], x) for x,_ in task.eval_queries[:50]]
def bench(prompts, op=None, layer=None, n=3):
    ts=[]
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.time()
        hlm.generate(prompts, batch_size=25, op=op, layer=layer)
        torch.cuda.synchronize(); ts.append(time.time()-t0)
    return min(ts)
t_plain = bench(q)
t_steer = bench(q, ops, lys)
t_icl = bench(icl)
print(f"zero-shot plain: {t_plain:.2f}s | steered: {t_steer:.2f}s (+{(t_steer/t_plain-1)*100:.0f}%) | 10-shot ICL: {t_icl:.2f}s ({t_icl/t_steer:.1f}x steered)")
