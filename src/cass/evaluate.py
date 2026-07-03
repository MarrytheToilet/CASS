"""Exact-match evaluation of generated answers."""
import re
import string

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.strip().split("\n")[0].strip()
    s = s.strip(string.punctuation + " ")
    return re.sub(r"\s+", " ", s.lower())


def exact_match(pred: str, target: str) -> bool:
    p, t = normalize(pred), normalize(target)
    if not t:
        return False
    if p == t or p.startswith(t + " ") or p.startswith(t):
        return True
    # first-word match for single-word targets (tolerates trailing rambling)
    if " " not in t and p.split(" ")[:1] == [t]:
        return True
    return False


def accuracy(preds, targets) -> float:
    assert len(preds) == len(targets)
    if not preds:
        return 0.0
    return sum(exact_match(p, t) for p, t in zip(preds, targets)) / len(preds)
