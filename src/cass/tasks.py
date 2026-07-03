"""Task registry, data splits, and ICL prompt construction.

Datasets come from Todd et al. function_vectors repo (input/output string pairs).
Splits per task (fixed seed): eval queries (50) | dictionary pool (extraction
queries + shot sampling) | few-shot pool (k-shot examples for unseen-task tests).
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR

# name -> (relative path, family)
TASK_REGISTRY = {
    # knowledge mapping
    "country-capital": ("abstractive/country-capital.json", "knowledge"),
    "country-currency": ("abstractive/country-currency.json", "knowledge"),
    "landmark-country": ("abstractive/landmark-country.json", "knowledge"),
    "park-country": ("abstractive/park-country.json", "knowledge"),
    "person-occupation": ("abstractive/person-occupation.json", "knowledge"),
    "person-sport": ("abstractive/person-sport.json", "knowledge"),
    "person-instrument": ("abstractive/person-instrument.json", "knowledge"),
    "product-company": ("abstractive/product-company.json", "knowledge"),
    # linguistic transforms
    "antonym": ("abstractive/antonym.json", "linguistic"),
    "synonym": ("abstractive/synonym.json", "linguistic"),
    "present-past": ("abstractive/present-past.json", "linguistic"),
    "singular-plural": ("abstractive/singular-plural.json", "linguistic"),
    "capitalize": ("abstractive/capitalize.json", "linguistic"),
    "capitalize-first-letter": ("abstractive/capitalize_first_letter.json", "linguistic"),
    "capitalize-last-letter": ("abstractive/capitalize_last_letter.json", "linguistic"),
    "lowercase-first-letter": ("abstractive/lowercase_first_letter.json", "linguistic"),
    # translation
    "english-french": ("abstractive/english-french.json", "translation"),
    "english-german": ("abstractive/english-german.json", "translation"),
    "english-spanish": ("abstractive/english-spanish.json", "translation"),
    # algorithmic / symbolic
    "next-item": ("abstractive/next_item.json", "algorithmic"),
    "prev-item": ("abstractive/prev_item.json", "algorithmic"),
    "next-capital-letter": ("abstractive/next_capital_letter.json", "algorithmic"),
    "word-length": ("abstractive/word_length.json", "algorithmic"),
    "alphabetically-first": ("extractive/alphabetically_first_5.json", "algorithmic"),
    "alphabetically-last": ("extractive/alphabetically_last_5.json", "algorithmic"),
    "choose-first-of-list": ("extractive/choose_first_of_5.json", "algorithmic"),
    "choose-last-of-list": ("extractive/choose_last_of_5.json", "algorithmic"),
    "choose-middle-of-list": ("extractive/choose_middle_of_5.json", "algorithmic"),
    # extractive selection (list -> category member)
    "animal-from-list": ("extractive/animal_v_object_5.json", "selection"),
    "color-from-list": ("extractive/color_v_animal_5.json", "selection"),
    "fruit-from-list": ("extractive/fruit_v_animal_5.json", "selection"),
    "verb-from-list": ("extractive/verb_v_adjective_5.json", "selection"),
}

ALL_TASKS = list(TASK_REGISTRY)
FAMILIES = sorted({fam for _, fam in TASK_REGISTRY.values()})

N_EVAL = 50

# small fixed vocabulary used to corrupt labels when a derangement is
# impossible (single-shot prompts for k=1 unseen-task extraction)
_CORRUPT_WORDS = (
    "table window music garden river summer paper stone yellow doctor "
    "seven walking bright cold mountain letter engine forest silver market"
).split()


@dataclass
class TaskData:
    name: str
    family: str
    eval_queries: list = field(default_factory=list)   # [(x, y)] held-out evaluation
    dict_pool: list = field(default_factory=list)      # extraction queries + shots
    fewshot_pool: list = field(default_factory=list)   # k-shot examples for unseen-task tests


def load_task(name: str, seed: int = 0) -> TaskData:
    rel, family = TASK_REGISTRY[name]
    items = json.load(open(Path(DATA_DIR) / rel))
    # dedupe by input, keep string pairs only
    seen, pairs = set(), []
    for it in items:
        x, y = str(it["input"]).strip(), str(it["output"]).strip()
        if x and y and x not in seen:
            seen.add(x)
            pairs.append((x, y))
    rng = random.Random(1000 + seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_eval = min(N_EVAL, n // 4)
    n_few = min(30, max(8, (n - n_eval) // 6))
    return TaskData(
        name=name, family=family,
        eval_queries=pairs[:n_eval],
        fewshot_pool=pairs[n_eval:n_eval + n_few],
        dict_pool=pairs[n_eval + n_few:],
    )


def icl_prompt(shots, query_x):
    parts = [f"Q: {x}\nA: {y}" for x, y in shots]
    parts.append(f"Q: {query_x}\nA:")
    return "\n\n".join(parts)


def zs_prompt(query_x):
    return f"Q: {query_x}\nA:"


def corrupt_shots(shots, rng: random.Random):
    """Same inputs, labels permuted with no fixed point (derangement).
    For a single shot, substitute a random unrelated word."""
    if len(shots) == 1:
        x, y = shots[0]
        w = rng.choice([w for w in _CORRUPT_WORDS if w.lower() != y.lower()])
        return [(x, w)]
    labels = [y for _, y in shots]
    for _ in range(100):
        perm = labels[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(perm, labels)):
            break
    return [(x, p) for (x, _), p in zip(shots, perm)]


def build_pair_prompts(pool, n_pairs, n_shots, rng: random.Random):
    """n_pairs of (clean, corrupted) ICL prompts from a task pool.
    Query is held out of its own shots."""
    clean, corrupted = [], []
    idx = list(range(len(pool)))
    for i in range(n_pairs):
        qi = idx[i % len(idx)]
        qx = pool[qi][0]
        others = [j for j in idx if j != qi]
        shots = [pool[j] for j in rng.sample(others, min(n_shots, len(others)))]
        clean.append(icl_prompt(shots, qx))
        corrupted.append(icl_prompt(corrupt_shots(shots, rng), qx))
    return clean, corrupted


def build_fewshot_pair_prompts(examples, rng: random.Random, n_reps=6,
                               leave_self_out=True):
    """Unseen-task extraction from k examples only. For each example j we build
    n_reps (clean, corrupted) pairs with resampled shot orders and corruption
    draws to denoise the per-example diff vector. Shots exclude the query
    example when k>=2 (avoids answer leakage); k=1 falls back to self-shot
    with random-word corruption."""
    clean, corrupted = [], []
    k = len(examples)
    for j, (x, _) in enumerate(examples):
        shots_base = ([e for i, e in enumerate(examples) if i != j]
                      if (leave_self_out and k >= 2) else list(examples))
        for _ in range(n_reps):
            shots = shots_base[:]
            rng.shuffle(shots)
            clean.append(icl_prompt(shots, x))
            corrupted.append(icl_prompt(corrupt_shots(shots, rng), x))
    return clean, corrupted
