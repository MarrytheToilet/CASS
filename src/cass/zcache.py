"""Cache few-shot z extractions (task, k, seed) -> [k, L+1, d] tensor."""
import numpy as np
import torch

from .config import results_dir
from .extract import extract_fewshot_z


def get_z(hlm, task, k, seed, compound=False):
    """Returns [k, L+1, d] float32 tensor of per-example diff vectors."""
    cache = results_dir(hlm.key) / "zcache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{task.name}_k{k}_s{seed}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False).float()
    rng = np.random.default_rng(100 * seed + k)
    idx = rng.choice(len(task.fewshot_pool), k, replace=False)
    examples = [task.fewshot_pool[i] for i in idx]
    Z = extract_fewshot_z(hlm, examples, seed=seed)
    torch.save(Z.to(torch.float16), path)
    return Z
