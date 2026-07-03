"""Exact-match evaluation of generated answers.

Match rule: the first len(target_words) words of the prediction's first line
must equal the target's words (punctuation-stripped per word). This is strict
for single-letter/short targets ('Walked' does NOT match target 'W') while
tolerating trailing rambling after the answer.

case_sensitive=True is required for tasks whose targets differ only by case
(e.g. compound '...+capitalize' tasks).
"""
import json
import string


def _words(s: str, case_sensitive: bool):
    s = s.strip().split("\n")[0]
    if not case_sensitive:
        s = s.lower()
    return [w for w in (t.strip(string.punctuation) for t in s.split())
            if w]


def exact_match(pred: str, target: str, case_sensitive: bool = False) -> bool:
    t = _words(target, case_sensitive)
    if not t:
        return False
    p = _words(pred, case_sensitive)
    return p[:len(t)] == t


def accuracy(preds, targets, case_sensitive: bool = False) -> float:
    assert len(preds) == len(targets)
    if not preds:
        return 0.0
    return sum(exact_match(p, t, case_sensitive)
               for p, t in zip(preds, targets)) / len(preds)


def dump_preds(path, tag, prompts_or_inputs, preds, targets):
    """Append raw generations for offline re-scoring."""
    with open(path, "a") as f:
        for x, p, t in zip(prompts_or_inputs, preds, targets):
            f.write(json.dumps(dict(tag=tag, input=x, pred=p, target=t),
                               ensure_ascii=False) + "\n")
