"""Offline secondary metric from stored generations: does the prediction
CONTAIN the target answer anywhere in the generated text? Separates
"wrong knowledge" from "right knowledge, wrong format" (e.g. person-sport
biographies that mention the sport)."""
import json
import string
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cass.config import results_dir
from cass.evaluate import exact_match

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama31-8b"
GENS = sys.argv[2] if len(sys.argv) > 2 else "e1_gens.jsonl"


def contains(pred, target):
    p = " ".join(w.strip(string.punctuation)
                 for w in pred.lower().split())
    t = " ".join(w.strip(string.punctuation)
                 for w in target.lower().split())
    return bool(t) and t in p


def main():
    path = results_dir(MODEL) / GENS
    em = defaultdict(list)
    ct = defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        em[d["tag"]].append(exact_match(d["pred"], d["target"]))
        ct[d["tag"]].append(contains(d["pred"], d["target"]))
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
    main()
