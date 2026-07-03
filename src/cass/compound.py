"""Compound task construction: chain two (or three) base task operators.

A compound task feeds one task's data through string-level post-transforms
(capitalize / lowercase / first-letter / plural-style suffixing is avoided --
we only chain transforms that are exact and well-defined on the base outputs).
"""
import json
import random
from pathlib import Path

from .config import DATA_DIR
from .tasks import TASK_REGISTRY, TaskData

# string-level operators usable as second stage; semantics match the
# corresponding dictionary task exactly:
#   capitalize: 'word' -> 'Word'; capitalize-first-letter: 'word' -> 'W';
#   lowercase-first-letter: 'WORD'/'Word' -> 'w'; word-length: 'word' -> '4'
_STR_OPS = {
    "capitalize": lambda s: s[:1].upper() + s[1:],
    "capitalize-first-letter": lambda s: s[:1].upper(),
    "lowercase-first-letter": lambda s: s[:1].lower(),
    "word-length": lambda s: str(len(s)),
}

# mapping-level second stage: apply another dataset's mapping to the output
# (requires the intermediate output to be in the second dataset's input set)


def _load_pairs(name):
    rel, _ = TASK_REGISTRY[name]
    items = json.load(open(Path(DATA_DIR) / rel))
    return {str(it["input"]).strip(): str(it["output"]).strip() for it in items}

# compound registry: name -> (base_task, second_stage) where second_stage is
# a string-op name or ("map", task_name)
COMPOUND_REGISTRY = {
    "antonym+capitalize": ("antonym", "capitalize"),
    "synonym+capitalize": ("synonym", "capitalize"),
    "present-past+capitalize": ("present-past", "capitalize"),
    "singular-plural+capitalize": ("singular-plural", "capitalize"),
    "english-french+capitalize": ("english-french", "capitalize"),
    "next-item+capitalize": ("next-item", "capitalize"),
    "antonym+capitalize-first-letter": ("antonym", "capitalize-first-letter"),
    "present-past+capitalize-first-letter": ("present-past",
                                             "capitalize-first-letter"),
    "country-capital+word-length": ("country-capital", "word-length"),
    "country-capital+lowercase-first-letter": ("country-capital",
                                               "lowercase-first-letter"),
}


def compound_components(name):
    base, second = COMPOUND_REGISTRY[name]
    return [base, second]


def load_compound(name: str, seed: int = 0) -> TaskData:
    base, second = COMPOUND_REGISTRY[name]
    pairs = _load_pairs(base)
    op = _STR_OPS[second]
    out_pairs, seen = [], set()
    for x, y in pairs.items():
        y2 = op(y)
        if x and y2 and y2 != y and x not in seen:  # keep only cases where the
            seen.add(x)                              # second stage matters
            out_pairs.append((x, y2))
    rng = random.Random(1000 + seed)
    rng.shuffle(out_pairs)
    n = len(out_pairs)
    n_eval = min(50, n // 3)
    n_few = min(30, max(8, n // 6))
    return TaskData(name=name, family="compound",
                    eval_queries=out_pairs[:n_eval],
                    fewshot_pool=out_pairs[n_eval:n_eval + n_few],
                    dict_pool=out_pairs[n_eval + n_few:])
