"""Contrastive activation extraction: g = h(clean 10-shot) - h(label-shuffled 10-shot).

All layers are captured in one forward pass and stored, so layer selection and
layer ablations never require re-extraction.
"""
import json
import random

import torch

from .config import results_dir
from .models import HookedLM
from .tasks import TaskData, build_pair_prompts, build_fewshot_pair_prompts


def extract_task(hlm: HookedLM, task: TaskData, n_pairs=100, n_shots=10,
                 seed=0, batch_size=8):
    """Returns diff activations G [n, L+1, d] (float32) for one task."""
    rng = random.Random(7000 + seed)
    clean, corrupted = build_pair_prompts(task.dict_pool, n_pairs, n_shots, rng)
    h_pos = hlm.last_token_hiddens(clean, batch_size)
    h_neg = hlm.last_token_hiddens(corrupted, batch_size)
    return h_pos - h_neg


def extract_and_save(hlm: HookedLM, task: TaskData, n_pairs=100, n_shots=10,
                     seed=0, batch_size=8):
    out = results_dir(hlm.key) / "activations"
    out.mkdir(exist_ok=True)
    path = out / f"{task.name}.pt"
    if path.exists():
        return path
    G = extract_task(hlm, task, n_pairs, n_shots, seed, batch_size)
    torch.save({"G": G.to(torch.float16), "task": task.name, "family": task.family,
                "n_pairs": n_pairs, "n_shots": n_shots, "seed": seed}, path)
    return path


def load_G(model_key: str, task_name: str, layer: int = None) -> torch.Tensor:
    """Loads [n, L+1, d] float32, or [n, d] at one layer."""
    blob = torch.load(results_dir(model_key) / "activations" / f"{task_name}.pt",
                      map_location="cpu", weights_only=False)
    G = blob["G"].float()
    return G if layer is None else G[:, layer, :]


def extract_fewshot_z(hlm: HookedLM, examples, seed=0, batch_size=8):
    """Unseen-task query representations from k examples.
    Returns Z [k, L+1, d] (one diff vector per example, all layers)."""
    rng = random.Random(9000 + seed)
    clean, corrupted = build_fewshot_pair_prompts(examples, rng)
    h_pos = hlm.last_token_hiddens(clean, batch_size)
    h_neg = hlm.last_token_hiddens(corrupted, batch_size)
    return h_pos - h_neg
