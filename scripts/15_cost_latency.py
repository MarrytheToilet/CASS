"""Serving-cost measurements, two stages:
  tokens  : full 10-shot ICL per query vs CASS one-time few-shot extraction
            + zero-context queries; pure tokenizer arithmetic, no GPU
            -> token_cost.json (Table 3)
  latency : GPU micro-benchmark, steered vs plain vs 10-shot ICL generation
            wall-clock (hook overhead and the ICL slowdown quoted in RQ4)
Usage: python 18_cost_latency.py [model] [tokens|latency|all]
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.config import MODEL_PATHS, results_dir
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt, \
    build_fewshot_pair_prompts

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"
K = 4


def tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATHS[MODEL]))
    n = lambda s: len(tok(s).input_ids)
    rng = random.Random(0)
    icl_tok, zs_tok, extract_tok = [], [], []
    for t in ALL_TASKS:
        task = load_task(t)
        pool = task.dict_pool
        for x, _ in task.eval_queries[:10]:
            icl_tok.append(n(icl_prompt(rng.sample(pool, 10), x)))
            zs_tok.append(n(zs_prompt(x)))
        ex = task.fewshot_pool[:K]
        clean, corr = build_fewshot_pair_prompts(ex, rng)
        extract_tok.append(sum(n(p) for p in clean + corr))
    icl_m, zs_m, ext_m = map(np.mean, (icl_tok, zs_tok, extract_tok))
    rows = {}
    for Q in (10, 100, 1000, 10000):
        rows[Q] = dict(
            icl_total=int(icl_m * Q),
            cass_total=int(ext_m + zs_m * Q),
            ratio=round(icl_m * Q / (ext_m + zs_m * Q), 1))
    out = dict(model=MODEL, k=K,
               mean_icl_prompt_tokens=round(float(icl_m), 1),
               mean_zs_prompt_tokens=round(float(zs_m), 1),
               one_time_extraction_tokens=round(float(ext_m), 1),
               per_query_costs=rows)
    json.dump(out, open(results_dir(MODEL) / "token_cost.json", "w"),
              indent=2)
    print(json.dumps(out, indent=2))


def latency():
    import torch
    from cass.dictionary import build_multilayer_dictionary
    from cass.extract import load_G
    from cass.models import HookedLM
    from cass.pipeline import code_for, ops_for, z_list_from_Z
    from cass.zcache import get_z

    LAYERS = [12, 16]
    hlm = HookedLM(MODEL)
    G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS}
         for l in LAYERS}
    task = load_task("antonym")
    D = build_multilayer_dictionary(
        {l: {t: G[l][t] for t in ALL_TASKS if t != "antonym"}
         for l in LAYERS}, r0=1)
    Z = get_z(hlm, task, K, 0)
    zl = z_list_from_Z(D, Z)
    code = code_for(D, zl)
    ops, lys = ops_for(D, code, 1.0, 2.0, 1.0,
                       delta_vec=np.mean(zl, axis=0))
    q = [zs_prompt(x) for x, _ in task.eval_queries[:50]]
    icl = [icl_prompt(task.dict_pool[:10], x)
           for x, _ in task.eval_queries[:50]]

    def bench(prompts, op=None, layer=None, n=3):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.time()
            hlm.generate(prompts, batch_size=25, op=op, layer=layer)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        return min(ts)

    t_plain = bench(q)
    t_steer = bench(q, ops, lys)
    t_icl = bench(icl)
    print(f"zero-shot plain: {t_plain:.2f}s | steered: {t_steer:.2f}s "
          f"(+{(t_steer/t_plain-1)*100:.0f}%) | 10-shot ICL: {t_icl:.2f}s "
          f"({t_icl/t_steer:.1f}x steered)")


if __name__ == "__main__":
    if STAGE in ("tokens", "all"):
        tokens()
    if STAGE in ("latency", "all"):
        latency()
