"""Fill remaining Table-1 gaps: run each missing (method, suite) combo.
  novel   : blend, recon, learned-delta
  compound: icv_pc1, blend, zvec, learned-delta, oracle(own extraction),
            4-shot ICL
Checkpointed in results/llama31-8b/fill_gaps.csv.
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cass.config import results_dir
from cass.compound import COMPOUND_REGISTRY, load_compound
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy
from cass.extract import extract_and_save, load_G
from cass.models import HookedLM
from cass.pipeline import (code_for, ops_for, oracle_ops, z_list_from_Z,
                           _support_weights)
from cass.tasks import ALL_TASKS, load_task, icl_prompt, synthetic_tasks, \
    zs_prompt
from cass.zcache import get_z

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}
D = build_multilayer_dictionary(G, r0=1)

path = out / "fill_gaps.csv"
done = set()
if path.exists():
    with open(path) as f:
        done = {(r["suite"], r["method"], r["task"], r["seed"])
                for r in csv.DictReader(f)}
fout = open(path, "a", newline="")
w = csv.DictWriter(fout, fieldnames=["suite", "method", "task", "seed",
                                     "acc"])
if not done:
    w.writeheader()


def emit(suite, method, task, seed, acc):
    w.writerow(dict(suite=suite, method=method, task=task, seed=seed,
                    acc=acc))
    fout.flush()


def eval_op(task, ops, lys, cs=False):
    q = task.eval_queries[:50]
    return accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                 batch_size=25, op=ops, layer=lys),
                    [y for _, y in q], case_sensitive=cs)


def blend_ops(code, z_mean):
    if code.support:
        ws = _support_weights(code)
        blend = sum(wi * D.anchors[n] for wi, n in zip(ws, code.support))
    else:
        blend = z_mean
    return ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=blend)


def icv_ops(Z):
    from cass.steer import make_additive_op
    ops = []
    for l in LAYERS:
        Zl = Z[:, l].numpy().astype(np.float64)
        Zl = np.stack([D.per_layer[l].project_out_shared(z) for z in Zl])
        pc = np.linalg.svd(Zl, full_matrices=False)[2][0]
        if pc @ Zl.mean(0) < 0:
            pc = -pc
        norm = np.median([np.linalg.norm(D.per_layer[l].anchors[t])
                          for t in D.task_names])
        ops.append(make_additive_op(pc * norm, gamma=1.0))
    return ops, list(LAYERS)


def learned_ops(task, seed):
    rng = np.random.default_rng(100 * seed + K)
    idx = rng.choice(len(task.fewshot_pool), K, replace=False)
    ex = [task.fewshot_pool[i] for i in idx]
    deltas = [torch.zeros(hlm.d, device=hlm.device, dtype=torch.float32,
                          requires_grad=True) for _ in LAYERS]
    opt = torch.optim.Adam(deltas, lr=0.02)
    texts = [f"Q: {x}\nA: {y}" for x, y in ex]
    enc = hlm.tok(texts, return_tensors="pt", padding=True).to(hlm.device)
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    for bi, (x, y) in enumerate(ex):
        plen = len(hlm.tok(f"Q: {x}\nA:").input_ids)
        pad = (enc.attention_mask[bi] == 0).sum().item()
        labels[bi, :pad + plen] = -100
    handles = []
    for dd, l in zip(deltas, LAYERS):
        def hook(module, inputs, output, _d=dd):
            h = output[0] if isinstance(output, tuple) else output
            return (h + _d.to(h.dtype),) + tuple(output[1:]) \
                if isinstance(output, tuple) else h + _d.to(h.dtype)
        handles.append(hlm.layers[l - 1].register_forward_hook(hook))
    for p in hlm.model.parameters():
        p.requires_grad_(False)
    hlm.model.config.use_cache = False
    try:
        for _ in range(40):
            opt.zero_grad()
            for bi in range(enc.input_ids.shape[0]):
                o = hlm.model(input_ids=enc.input_ids[bi:bi + 1],
                              attention_mask=enc.attention_mask[bi:bi + 1],
                              labels=labels[bi:bi + 1])
                (o.loss / enc.input_ids.shape[0]).backward()
            opt.step()
    finally:
        for h in handles:
            h.remove()
        hlm.model.config.use_cache = True
        torch.cuda.empty_cache()
    from cass.steer import make_additive_op

    def all_pos_op(vec):
        v = torch.as_tensor(vec, device="cuda", dtype=torch.float32)

        def op(h):
            return h.float() + v
        return op
    return [all_pos_op(d.detach().cpu().numpy()) for d in deltas], \
        list(LAYERS)


def eval_learned_allpos(task, ops, lys, cs=False):
    # learned delta applies at ALL positions (training-consistent)
    q = task.eval_queries[:50]
    prompts = [zs_prompt(x) for x, _ in q]
    handles = []
    for o, l in zip(ops, lys):
        def hook(module, inputs, output, _o=o):
            h = output[0] if isinstance(output, tuple) else output
            delta = (_o(h[:, 0, :]) - h[:, 0, :].float()).to(h.dtype)
            h += delta.unsqueeze(1)
        handles.append(hlm.layers[l - 1].register_forward_hook(hook))
    try:
        texts = []
        for i in range(0, len(prompts), 25):
            enc = hlm.tok(prompts[i:i + 25], return_tensors="pt",
                          padding=True).to(hlm.device)
            gen = hlm.model.generate(**enc, max_new_tokens=8,
                                     do_sample=False,
                                     pad_token_id=hlm.tok.pad_token_id)
            texts.extend(hlm.tok.batch_decode(
                gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    finally:
        for h in handles:
            h.remove()
    return accuracy(texts, [y for _, y in q], case_sensitive=cs)


t0 = time.time()

# ---------- novel suite ----------
for tname in list(synthetic_tasks()):
    task = load_task(tname)
    for seed in [0, 1, 2]:
        need = [m for m in ["blend", "recon", "learned"]
                if ("novel", m, tname, str(seed)) not in done]
        if not need:
            continue
        Z = get_z(hlm, task, K, seed)
        z_list = z_list_from_Z(D, Z)
        z_mean = np.mean(z_list, axis=0)
        code = code_for(D, z_list)
        if "blend" in need:
            ops, lys = blend_ops(code, z_mean)
            emit("novel", "blend", tname, seed, eval_op(task, ops, lys))
        if "recon" in need:
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0)
            emit("novel", "recon", tname, seed, eval_op(task, ops, lys))
        if "learned" in need:
            ops, lys = learned_ops(task, seed)
            emit("novel", "learned", tname, seed,
                 eval_learned_allpos(task, ops, lys))
    print(f"novel/{tname} done ({(time.time()-t0)/60:.1f} min)", flush=True)

# ---------- compound suite ----------
for cname in COMPOUND_REGISTRY:
    comp = load_compound(cname)
    # oracle: extract compound's own subspace once
    extract_and_save(hlm, comp, n_pairs=100, batch_size=6)
    if ("compound", "oracle", cname, "0") not in done:
        Gc = {l: load_G(MODEL, cname, l).numpy() for l in LAYERS}
        Dc = build_multilayer_dictionary(
            {l: {**G[l], cname: Gc[l]} for l in LAYERS}, r0=1)
        ops, lys = oracle_ops(Dc, cname, 1.0, 2.0, 1.0)
        emit("compound", "oracle", cname, 0,
             eval_op(comp, ops, lys, cs=True))
    for seed in [0, 1, 2]:
        need = [m for m in ["icv", "blend", "zvec", "learned", "icl4"]
                if ("compound", m, cname, str(seed)) not in done]
        if not need:
            continue
        Z = get_z(hlm, comp, K, seed)
        z_list = z_list_from_Z(D, Z)
        z_mean = np.mean(z_list, axis=0)
        code = code_for(D, z_list)
        if "icv" in need:
            ops, lys = icv_ops(Z)
            emit("compound", "icv", cname, seed,
                 eval_op(comp, ops, lys, cs=True))
        if "blend" in need:
            ops, lys = blend_ops(code, z_mean)
            emit("compound", "blend", cname, seed,
                 eval_op(comp, ops, lys, cs=True))
        if "zvec" in need:
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0,
                               injection="additive", delta_vec=z_mean)
            emit("compound", "zvec", cname, seed,
                 eval_op(comp, ops, lys, cs=True))
        if "learned" in need:
            ops, lys = learned_ops(comp, seed)
            emit("compound", "learned", cname, seed,
                 eval_learned_allpos(comp, ops, lys, cs=True))
        if "icl4" in need:
            rng = np.random.default_rng(100 * seed + K)
            idx = rng.choice(len(comp.fewshot_pool), K, replace=False)
            shots = [comp.fewshot_pool[i] for i in idx]
            q = comp.eval_queries[:50]
            acc = accuracy(hlm.generate(
                [icl_prompt(shots, x) for x, _ in q], batch_size=16),
                [y for _, y in q], case_sensitive=True)
            emit("compound", "icl4", cname, seed, acc)
    print(f"compound/{cname} done ({(time.time()-t0)/60:.1f} min)",
          flush=True)

fout.close()
print("FILL GAPS DONE")
