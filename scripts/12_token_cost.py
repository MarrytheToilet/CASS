"""Token cost comparison: full 10-shot ICL per query vs CASS one-time few-shot
extraction + zero-context queries. Pure tokenizer arithmetic, no GPU."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from transformers import AutoTokenizer

from cass.config import MODEL_PATHS, results_dir
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt, \
    build_fewshot_pair_prompts
import random

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
K = 4

def main():
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
    json.dump(out, open(results_dir(MODEL) / "token_cost.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
