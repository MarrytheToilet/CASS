"""4-shot ICL baseline: the SAME k=4 demonstrations, used in-context."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cass.config import results_dir
from cass.evaluate import accuracy
from cass.models import HookedLM
from cass.tasks import ALL_TASKS, load_task, icl_prompt, synthetic_tasks

MODEL, K = "llama31-8b", 4
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
path = out / "icl4.csv"
done = set()
if path.exists():
    with open(path) as f:
        done = {(r["task"], r["seed"]) for r in csv.DictReader(f)}
fout = open(path, "a", newline="")
w = csv.DictWriter(fout, fieldnames=["suite", "task", "seed", "acc"])
if not done:
    w.writeheader()
for suite, names in [("loto", ALL_TASKS), ("novel", list(synthetic_tasks()))]:
    for tname in names:
        task = load_task(tname)
        q = task.eval_queries[:50]
        for seed in [0, 1, 2]:
            if (tname, str(seed)) in done:
                continue
            rng = np.random.default_rng(100 * seed + K)
            idx = rng.choice(len(task.fewshot_pool), K, replace=False)
            shots = [task.fewshot_pool[i] for i in idx]
            prompts = [icl_prompt(shots, x) for x, _ in q]
            acc = accuracy(hlm.generate(prompts, batch_size=16),
                           [y for _, y in q])
            w.writerow(dict(suite=suite, task=tname, seed=seed, acc=acc))
            fout.flush()
        print(tname, "done", flush=True)
fout.close()
print("ICL4 DONE")
