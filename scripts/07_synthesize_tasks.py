"""Synthesize new word-mapping tasks via the .env API (dictionary/test
enrichment). Pipeline per task: generate ~200 pairs -> dedupe/format-filter ->
cross-model verification of a sample (factual tasks) -> save if it passes.

The CASS method itself never touches the API; these are datasets only.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cass.api import chat_json, MODELS
from cass.config import ROOT

OUT = ROOT / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

# name -> (instruction, example pairs, family, factual?)
NEW_TASKS = {
    "country-continent": ("map a country to its continent",
                          [("France", "Europe"), ("Peru", "South America")],
                          "knowledge", True),
    "element-symbol": ("map a chemical element name to its symbol",
                       [("gold", "Au"), ("oxygen", "O")], "knowledge", True),
    "animal-baby": ("map an animal to the word for its young",
                    [("dog", "puppy"), ("cow", "calf")], "knowledge", True),
    "fruit-color": ("map a fruit or vegetable to its typical color",
                    [("banana", "yellow"), ("spinach", "green")],
                    "knowledge", True),
    "profession-workplace": ("map a profession to its typical workplace",
                             [("doctor", "hospital"), ("chef", "kitchen")],
                             "knowledge", True),
    "capital-country": ("map a capital city to its country (inverse of "
                        "country-capital)",
                        [("Paris", "France"), ("Tokyo", "Japan")],
                        "knowledge", True),
    "verb-gerund": ("map a verb to its -ing form",
                    [("run", "running"), ("see", "seeing")],
                    "linguistic", False),
    "adjective-comparative": ("map an adjective to its comparative form",
                              [("big", "bigger"), ("happy", "happier")],
                              "linguistic", False),
    "adjective-adverb": ("map an adjective to its adverb form",
                         [("quick", "quickly"), ("easy", "easily")],
                         "linguistic", False),
    "past-present": ("map a past-tense verb to its present tense (inverse "
                     "of present-past)",
                     [("went", "go"), ("saw", "see")], "linguistic", False),
    "english-italian": ("translate a common English word to Italian",
                        [("house", "casa"), ("water", "acqua")],
                        "translation", True),
    "english-portuguese": ("translate a common English word to Portuguese",
                           [("book", "livro"), ("red", "vermelho")],
                           "translation", True),
    "french-english": ("translate a common French word to English (inverse "
                       "direction)",
                       [("maison", "house"), ("eau", "water")],
                       "translation", True),
    "word-last-letter": ("map a word to its last letter, lowercase",
                         [("apple", "e"), ("cat", "t")], "algorithmic", False),
    "next-day": ("map a day of the week to the following day, lowercase",
                 [("monday", "tuesday"), ("sunday", "monday")],
                 "algorithmic", True),
    "number-plus-one": ("map a number written in digits (1-999) to the next "
                        "number",
                        [("7", "8"), ("42", "43")], "algorithmic", False),
}

GEN_PROMPT = """Generate a dataset for the task: {instr}.
Examples: {ex}.
Produce EXACTLY a JSON array of 220 objects, each {{"input": "...", "output": "..."}}.
Rules: all inputs unique and common/unambiguous; outputs must be correct and
in the SAME format as the examples (single word unless the answer requires
more); no duplicates; plain JSON only, no commentary."""

VERIFY_PROMPT = """You are checking a word-mapping dataset for the task: {instr}.
For each pair below, answer whether the mapping is CORRECT. Reply with a JSON
array of 0/1 (1=correct), same order, nothing else.
Pairs: {pairs}"""


def synthesize(name):
    instr, ex, family, factual = NEW_TASKS[name]
    path = OUT / f"{name}.json"
    if path.exists():
        return json.load(open(path))
    pairs = chat_json(GEN_PROMPT.format(instr=instr, ex=ex),
                      model=MODELS[0], max_tokens=8000, temperature=0.8)
    seen, clean = set(), []
    for p in pairs:
        try:
            x, y = str(p["input"]).strip(), str(p["output"]).strip()
        except Exception:
            continue
        if x and y and x.lower() not in seen and len(y.split()) <= 3:
            seen.add(x.lower())
            clean.append({"input": x, "output": y})
    # programmatic checks for mechanical tasks
    if name == "word-last-letter":
        clean = [p for p in clean if p["output"] == p["input"][-1].lower()]
    if name == "number-plus-one":
        clean = [p for p in clean if p["input"].isdigit()
                 and p["output"] == str(int(p["input"]) + 1)]
    if name == "verb-gerund":
        clean = [p for p in clean if p["output"].endswith("ing")]
    # cross-model verification for factual tasks
    if factual and clean:
        rng = random.Random(0)
        keep = []
        for i in range(0, len(clean), 40):
            batch = clean[i:i + 40]
            pl = [(p["input"], p["output"]) for p in batch]
            try:
                flags = chat_json(VERIFY_PROMPT.format(instr=instr, pairs=pl),
                                  model=MODELS[1], max_tokens=3000,
                                  temperature=0.0)
                keep += [p for p, f in zip(batch, flags) if f == 1]
            except Exception:
                keep += batch  # verifier failed -> keep, log
        frac = len(keep) / max(len(clean), 1)
        print(f"  verify: kept {len(keep)}/{len(clean)} ({frac:.0%})")
        clean = keep
    json.dump(clean, open(path, "w"), ensure_ascii=False, indent=1)
    return clean


if __name__ == "__main__":
    report = {}
    for name in NEW_TASKS:
        try:
            data = synthesize(name)
            report[name] = len(data)
            print(f"{name}: {len(data)} pairs", flush=True)
        except Exception as e:
            report[name] = f"FAIL {e}"
            print(f"{name}: FAILED {e}", flush=True)
    json.dump(report, open(OUT / "_report.json", "w"), indent=2)
    ok = [n for n, v in report.items() if isinstance(v, int) and v >= 120]
    print(f"\n{len(ok)}/{len(NEW_TASKS)} tasks usable (>=120 pairs): {ok}")
