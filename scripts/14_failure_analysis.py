"""Failure analysis over stored E1 generations, two stages:
  judge    : LLM-judge taxonomy of failed cass k=4 predictions (strict exact
             match = 0) -> llm_judge_failures.json
      A format    : answer present/derivable but wrapped in prose/wrong format
      B neighbor  : output executes a DIFFERENT but related task
      C semantic  : on-task attempt with wrong answer
      D degenerate: repetition/garbage/off-task continuation
  contains : offline secondary metric -- does the prediction CONTAIN the
             target anywhere? Separates "wrong knowledge" from
             "right knowledge, wrong format" (person-sport biographies).
Usage: python 17_failure_analysis.py [model] [judge|contains|all] [gens.jsonl]
"""
import json
import random
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cass.api import chat_json, MODELS
from cass.config import results_dir
from cass.evaluate import exact_match

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"
GENS = sys.argv[3] if len(sys.argv) > 3 else "e1_gens.jsonl"
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


def judge():
    out = results_dir(MODEL)
    fails = defaultdict(list)
    for line in open(out / GENS):
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


def _contains(pred, target):
    p = " ".join(w.strip(string.punctuation) for w in pred.lower().split())
    t = " ".join(w.strip(string.punctuation) for w in target.lower().split())
    return bool(t) and t in p


def contains():
    path = results_dir(MODEL) / GENS
    em = defaultdict(list)
    ct = defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        em[d["tag"]].append(exact_match(d["pred"], d["target"]))
        ct[d["tag"]].append(_contains(d["pred"], d["target"]))
    # aggregate per task|mode (drop seed suffix)
    agg = defaultdict(lambda: [0, 0, 0])
    for tag in em:
        parts = tag.split("|")
        key = "|".join(parts[:2]) + ("|" + parts[2][:2] if len(parts) > 2
                                     else "")
        a = agg[key]
        a[0] += sum(em[tag])
        a[1] += sum(ct[tag])
        a[2] += len(em[tag])
    print(f"{'tag':46s} {'exact':>6s} {'contains':>8s}  (n)")
    for key in sorted(agg):
        e, c, n = agg[key]
        if c - e > 0.1 * n:  # only show where the gap is interesting
            print(f"{key:46s} {e/n:6.2f} {c/n:8.2f}  ({n})")


if __name__ == "__main__":
    if STAGE in ("judge", "all"):
        judge()
    if STAGE in ("contains", "all"):
        contains()
