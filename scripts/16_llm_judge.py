"""LLM-judge failure taxonomy over stored E1 generations.

For each failed prediction (strict exact match = 0) of mode=cass k=4, ask the
API to classify the failure:
  A format  : answer is present/derivable but wrapped in prose or wrong format
  B neighbor: output executes a DIFFERENT but related task (e.g. wrong
              language, occupation instead of sport)
  C semantic: on-task attempt with wrong answer
  D degenerate: repetition/garbage/off-task continuation
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cass.api import chat_json, MODELS
from cass.config import results_dir
from cass.evaluate import exact_match

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
PER_TASK = 20

JUDGE = """Task: given input "{inp}", the correct answer is "{tgt}".
A steered language model generated: "{pred}"
Classify the failure with ONE letter:
A = the correct answer (or a correct variant) appears in the generation, but
    wrapped in prose/extra text instead of being given directly
B = the generation executes a DIFFERENT but related task (e.g. translates to
    the wrong language, gives occupation instead of sport, capital instead of
    currency)
C = a direct on-task attempt with an incorrect answer
D = degenerate output (repetition, garbage, mere continuation of the input)
Reply with a JSON object: {{"label": "A|B|C|D"}}"""


def main():
    out = results_dir(MODEL)
    fails = defaultdict(list)
    for line in open(out / "e1_gens.jsonl"):
        d = json.loads(line)
        parts = d["tag"].split("|")
        if len(parts) == 3 and parts[1] == "cass" and \
                parts[2].startswith("k4") and \
                not exact_match(d["pred"], d["target"]):
            fails[parts[0]].append(d)

    rng = random.Random(0)
    rows = []
    for task, items in sorted(fails.items()):
        sample = rng.sample(items, min(PER_TASK, len(items)))
        labels = []
        for d in sample:
            try:
                r = chat_json(JUDGE.format(inp=d["input"][:120],
                                           tgt=d["target"][:60],
                                           pred=d["pred"][:200]),
                              model=MODELS[1], max_tokens=300,
                              temperature=0.0)
                lab = str(r.get("label", "?")).strip().upper()[:1]
            except Exception:
                lab = "?"
            labels.append(lab)
        c = Counter(labels)
        n = len(labels)
        rows.append(dict(task=task, n_failures=len(items), judged=n,
                         format_A=c["A"] / n, neighbor_B=c["B"] / n,
                         semantic_C=c["C"] / n, degenerate_D=c["D"] / n))
        print(f"{task}: fails={len(items)} A={c['A']} B={c['B']} "
              f"C={c['C']} D={c['D']}", flush=True)
    json.dump(rows, open(out / "llm_judge_failures.json", "w"), indent=2)
    tot = Counter()
    for r in rows:
        for k in ("format_A", "neighbor_B", "semantic_C", "degenerate_D"):
            tot[k] += r[k] * r["judged"]
    N = sum(r["judged"] for r in rows)
    print("\nOVERALL:", {k: round(v / N, 3) for k, v in tot.items()},
          f"(n={N})")

if __name__ == "__main__":
    main()
