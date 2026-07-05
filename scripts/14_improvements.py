"""Reviewer-driven experiments:
(i)   learned-Delta baseline: optimize a per-task steering delta on the SAME
      k=4 examples by gradient descent (minimal-training comparison point).
(ii)  signed per-skill gate: drop support members whose anchor opposes z
      from the correction (finer than the global gate).
(iii) prefill-only vs every-step injection.
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cass.config import results_dir
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import load_G
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z, _support_weights
from cass.steer import make_affine_op, _to_torch
from cass.tasks import ALL_TASKS, load_task, zs_prompt
from cass.zcache import get_z

MODEL = "llama31-8b"
LAYERS = [12, 16]
K = 4
N_EVAL = 50

out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}

rows_path = out / "improvements.csv"
done = set()
if rows_path.exists():
    with open(rows_path) as f:
        done = {(r["exp"], r["cond"], r["task"], r["seed"])
                for r in csv.DictReader(f)}
fout = open(rows_path, "a", newline="")
writer = csv.DictWriter(fout, fieldnames=["exp", "cond", "task", "seed",
                                          "acc"])
if not done:
    writer.writeheader()


def emit(exp, cond, task, seed, acc):
    writer.writerow(dict(exp=exp, cond=cond, task=task, seed=seed, acc=acc))
    fout.flush()


def eval_ops(task, ops, lys, n=N_EVAL):
    q = task.eval_queries[:n]
    return accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                 batch_size=25, op=ops, layer=lys),
                    [y for _, y in q])


def signed_ops(D, code, z_mean):
    """Per-skill signed gate: correction from members aligned with z only."""
    delta = z_mean.copy()
    w = _support_weights(code)
    target = float(sum(wi * np.linalg.norm(D.anchors[n])
                       for wi, n in zip(w, code.support)))
    dn = np.linalg.norm(delta)
    if dn > 1e-8:
        delta *= target / dn
    aligned, gs = [], []
    for wi, n in zip(w, code.support):
        gt = float(delta @ D.anchors[n] /
                   (np.linalg.norm(delta) * np.linalg.norm(D.anchors[n])
                    + 1e-12))
        if gt > 0.05:
            aligned.append((wi, n, gt))
    if not aligned:
        from cass.steer import make_additive_op
        return ([make_additive_op(d, gamma=1.0)
                 for d in np.split(delta, len(LAYERS))], list(LAYERS))
    wsum = sum(wi for wi, _, _ in aligned)
    gate = sum(wi * gt for wi, _, gt in aligned) / wsum
    ops, lys = [], []
    db = D.split(delta)
    for l in D.layers:
        Dl = D.per_layer[l]
        mu_l = sum(wi / wsum * Dl.anchors[n] for wi, n, _ in aligned)
        B_l = np.concatenate([Dl.bases[n] for _, n, _ in aligned], axis=1)
        from cass.pipeline import _gated_op
        ops.append(_gated_op(db[l], B_l, mu_l, gate, 1.0, 2.0, 1.0))
        lys.append(l)
    return ops, lys


def eval_prefill(task, ops, lys, n=N_EVAL):
    """Injection only during prefill (seq_len>1), not decode steps."""
    q = task.eval_queries[:n]
    prompts = [zs_prompt(x) for x, _ in q]
    handles = []
    for o, l in zip(ops, lys):
        def hook(module, inputs, output, _o=o):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] > 1:          # prefill only
                h[:, -1, :] = _o(h[:, -1, :]).to(h.dtype)
        handles.append(hlm.layers[l - 1].register_forward_hook(hook))
    try:
        texts = []
        for i in range(0, len(prompts), 25):
            enc = hlm.tok(prompts[i:i + 25], return_tensors="pt",
                          padding=True).to(hlm.device)
            gen = hlm.model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=hlm.tok.pad_token_id)
            texts.extend(hlm.tok.batch_decode(
                gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    finally:
        for h in handles:
            h.remove()
    return accuracy(texts, [y for _, y in q])



# ---------- (ii)+(iii) on all 32 tasks, 3 seeds ----------
for tstar in ALL_TASKS:
    task = load_task(tstar)
    D = build_multilayer_dictionary(
        {l: {t: G[l][t] for t in ALL_TASKS if t != tstar} for l in LAYERS},
        r0=1)
    for seed in [0, 1, 2]:
        need = [c for c in ["signed", "prefill_only"]
                if ("variant", c, tstar, str(seed)) not in done]
        if not need:
            continue
        Z = get_z(hlm, task, K, seed)
        z_list = z_list_from_Z(D, Z)
        z_mean = np.mean(z_list, axis=0)
        code = code_for(D, z_list)
        if "signed" in need:
            ops, lys = signed_ops(D, code, z_mean)
            emit("variant", "signed", tstar, seed, eval_ops(task, ops, lys))
        if "prefill_only" in need:
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=z_mean)
            emit("variant", "prefill_only", tstar, seed,
                 eval_prefill(task, ops, lys))
    print(f"variants {tstar} done", flush=True)


# ---------- (i) learned-Delta baseline ----------
def learn_delta(task, seed, steps=40, lr=0.02):
    rng = np.random.default_rng(100 * seed + K)
    idx = rng.choice(len(task.fewshot_pool), K, replace=False)
    examples = [task.fewshot_pool[i] for i in idx]
    deltas = [torch.zeros(hlm.d, device=hlm.device, dtype=torch.float32,
                          requires_grad=True) for _ in LAYERS]
    opt = torch.optim.Adam(deltas, lr=lr)
    texts = [f"Q: {x}\nA: {y}" for x, y in examples]
    enc = hlm.tok(texts, return_tensors="pt", padding=True).to(hlm.device)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    # supervise only the answer tokens (after "A:")
    for bi, (x, y) in enumerate(examples):
        plen = len(hlm.tok(f"Q: {x}\nA:").input_ids)
        pad = (enc.attention_mask[bi] == 0).sum().item()
        labels[bi, :pad + plen] = -100
    handles = []
    for d, l in zip(deltas, LAYERS):
        def hook(module, inputs, output, _d=d):
            h = output[0] if isinstance(output, tuple) else output
            return (h + _d.to(h.dtype),) + tuple(output[1:]) \
                if isinstance(output, tuple) else h + _d.to(h.dtype)
        handles.append(hlm.layers[l - 1].register_forward_hook(hook))
    for p in hlm.model.parameters():
        p.requires_grad_(False)
    hlm.model.config.use_cache = False
    try:
        for _ in range(steps):
            opt.zero_grad()
            for bi in range(enc.input_ids.shape[0]):   # per-example accum
                outp = hlm.model(
                    input_ids=enc.input_ids[bi:bi + 1],
                    attention_mask=enc.attention_mask[bi:bi + 1],
                    labels=labels[bi:bi + 1])
                (outp.loss / enc.input_ids.shape[0]).backward()
            opt.step()
    finally:
        for h in handles:
            h.remove()
        hlm.model.config.use_cache = True
        torch.cuda.empty_cache()
    return [d.detach() for d in deltas]


def eval_learned(task, deltas, n=N_EVAL):
    q = task.eval_queries[:n]
    prompts = [zs_prompt(x) for x, _ in q]
    handles = []
    for d, l in zip(deltas, LAYERS):
        def hook(module, inputs, output, _d=d):
            h = output[0] if isinstance(output, tuple) else output
            h += _d.to(h.dtype)          # all positions, matching training
        handles.append(hlm.layers[l - 1].register_forward_hook(hook))
    try:
        texts = []
        for i in range(0, len(prompts), 25):
            enc = hlm.tok(prompts[i:i + 25], return_tensors="pt",
                          padding=True).to(hlm.device)
            gen = hlm.model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=hlm.tok.pad_token_id)
            texts.extend(hlm.tok.batch_decode(
                gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    finally:
        for h in handles:
            h.remove()
    return accuracy(texts, [y for _, y in q])


t0 = time.time()
for tstar in ALL_TASKS:
    task = load_task(tstar)
    for seed in [0, 1, 2]:
        if ("learned", "delta40", tstar, str(seed)) in done:
            continue
        deltas = learn_delta(task, seed)
        emit("learned", "delta40", tstar, seed, eval_learned(task, deltas))
    print(f"learned {tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)

fout.close()
print("IMPROVEMENTS DONE")
