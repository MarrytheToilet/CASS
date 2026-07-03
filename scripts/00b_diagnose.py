"""Diagnose injection quality: inspect generations, scan gamma x layer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.evaluate import accuracy, normalize
from cass.extract import extract_task
from cass.models import HookedLM
from cass.steer import make_additive_op
from cass.tasks import load_task, zs_prompt

MODEL = "llama31-8b"
TASKS = ["antonym", "english-french", "country-capital"]
N_EVAL = 25

def main():
    hlm = HookedLM(MODEL)
    for tname in TASKS:
        task = load_task(tname)
        queries = task.eval_queries[:N_EVAL]
        targets = [y for _, y in queries]
        zs_prompts = [zs_prompt(x) for x, _ in queries]

        zs_preds = hlm.generate(zs_prompts, batch_size=16)
        print(f"\n=== {tname} (zs acc {accuracy(zs_preds, targets):.2f}) ===")
        for p, t in list(zip(zs_preds, targets))[:4]:
            print(f"  ZS pred={p!r} target={t!r}")

        G = extract_task(hlm, task, n_pairs=50, batch_size=6)
        best = (0, None, None)
        for l in [10, 12, 14, 16, 18]:
            mu = G[:, l, :].mean(0).numpy()
            hnorm_ratio = np.linalg.norm(mu)
            for gamma in [1.0, 2.0, 3.0, 5.0]:
                op = make_additive_op(mu, gamma=gamma)
                preds = hlm.generate(zs_prompts, batch_size=16, op=op, layer=l)
                acc = accuracy(preds, targets)
                if acc > best[0]:
                    best = (acc, l, gamma, preds)
                print(f"  l={l} gamma={gamma} |mu|={hnorm_ratio:.1f} acc={acc:.2f}")
        acc, l, gamma, preds = best
        print(f"  BEST l={l} gamma={gamma} acc={acc:.2f}; sample:")
        for p, t in list(zip(preds, targets))[:6]:
            print(f"    pred={p!r} target={t!r} match={normalize(p)[:20]!r}")

if __name__ == "__main__":
    main()
