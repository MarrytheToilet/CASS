"""Attribution studies for RQ2 (what the gain comes from), five stages:
  denoise  : z single-pair (m=1) vs denoised (m=6), additive vs full CASS
             -> review_response.csv  (exp=denoise)
  control  : shared-component specificity -- remove a RANDOM rank-1 direction
             instead of the identified U0 (5 draws)
             -> review_response.csv  (exp=control)
  hparam   : robustness to (gamma, beta, alpha_max) perturbed around (1,2,1)
             -> review_response.csv  (exp=hparam)
  variants : signed per-skill gate; prefill-only vs every-step injection
             -> improvements.csv     (exp=variant)
  learned  : learned-Delta reference (40 grad steps on the same k=4, frozen
             model params) on all 32 LOTO tasks
             -> improvements.csv     (exp=learned)
  soft     : soft interpolation toward the replacement state vs hard routing
             -> soft_hybrid.csv
  pc23     : specificity control -- project out ONLY the 2nd or 3rd shared
             PC instead of the 1st -> review_response.csv (exp=pc_control)
  alphaiso : isolate the adaptive scale -- keep gate g, fix alpha(h)=alpha_max
             (complements the existing 'ungated' variant, which keeps alpha
             and removes g) -> improvements.csv (exp=variant, cond=alpha_const)
  osupp    : compounds with ORACLE support (true constituents fed to the
             operator, z unchanged) -> fill_gaps.csv (method=oracle_support)
Usage: python 12_attribution.py [denoise|control|hparam|variants|learned|soft|all]
"""
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from cass.config import results_dir
from cass.dictionary import build_dictionary, build_multilayer_dictionary, \
    MultiLayerDictionary, rank_by_energy
from cass.evaluate import accuracy
from cass.extract import load_G, extract_fewshot_z
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z, _support_weights, \
    _gated_op
from cass.steer import make_additive_op, _to_torch
from cass.tasks import ALL_TASKS, load_task, icl_prompt, zs_prompt
from cass.zcache import get_z

MODEL, LAYERS, K = "llama31-8b", [12, 16], 4
STAGES = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["all"])
REP_TASKS = ["antonym", "present-past", "country-capital", "person-sport",
             "english-french", "next-item", "choose-first-of-list",
             "animal-from-list"]
SEEDS = [0, 1, 2]

out = results_dir(MODEL)
hlm = HookedLM(MODEL)
G = {l: {t: load_G(MODEL, t, l).numpy() for t in ALL_TASKS} for l in LAYERS}


def open_csv(path, fieldnames, done_key):
    done = set()
    if path.exists():
        with open(path) as f:
            done = {done_key(r) for r in csv.DictReader(f)}
    fout = open(path, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if not done or os.path.getsize(path) == 0:
        w.writeheader()
    return done, fout, w


def loto_dictionary(tstar):
    return build_multilayer_dictionary(
        {l: {t: G[l][t] for t in ALL_TASKS if t != tstar} for l in LAYERS},
        r0=1)


def eval_ops(task, ops, lys, n=50):
    q = task.eval_queries[:n]
    return accuracy(hlm.generate([zs_prompt(x) for x, _ in q],
                                 batch_size=25, op=ops, layer=lys),
                    [y for _, y in q])


def z_for(task, seed, n_reps):
    rng = np.random.default_rng(100 * seed + K)
    idx = rng.choice(len(task.fewshot_pool), K, replace=False)
    examples = [task.fewshot_pool[i] for i in idx]
    return extract_fewshot_z(hlm, examples, seed=seed, n_reps=n_reps)


# ---------- review_response.csv stages ----------
def _rr_writer():
    return open_csv(out / "review_response.csv",
                    ["exp", "cond", "task", "seed", "acc"],
                    lambda r: (r["exp"], r["cond"], r["task"], r["seed"]))


def denoise():
    done, fout, w = _rr_writer()
    t0 = time.time()
    for tstar in REP_TASKS:
        task = load_task(tstar)
        D = loto_dictionary(tstar)
        for seed in SEEDS:
            conds = [c for c in ["z_m1_add", "z_m6_add", "cass_m1", "cass_m6"]
                     if ("denoise", c, tstar, str(seed)) not in done]
            if not conds:
                continue
            Z1, Z6 = z_for(task, seed, 1), z_for(task, seed, 6)
            for cond, Z in [("z_m1_add", Z1), ("z_m6_add", Z6),
                            ("cass_m1", Z1), ("cass_m6", Z6)]:
                if cond not in conds:
                    continue
                zl = z_list_from_Z(D, Z)
                zm = np.mean(zl, axis=0)
                code = code_for(D, zl)
                if cond.endswith("_add"):
                    ops, lys = ops_for(D, code, 1.0, 2.0, 1.0,
                                       injection="additive", delta_vec=zm)
                else:
                    ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=zm)
                w.writerow(dict(exp="denoise", cond=cond, task=tstar,
                                seed=seed, acc=eval_ops(task, ops, lys, 25)))
                fout.flush()
        print(f"denoise {tstar} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    fout.close()


def control():
    done, fout, w = _rr_writer()

    def build_with_projection(exclude, U0_by_layer):
        per = {}
        for l in LAYERS:
            Dl = build_dictionary({t: G[l][t] for t in ALL_TASKS
                                   if t != exclude}, r0=0)
            U0 = U0_by_layer[l]
            for n in Dl.task_names:
                Gt = (G[l][n].astype(np.float64).T
                      - U0 @ (U0.T @ G[l][n].astype(np.float64).T))
                U, s, _ = np.linalg.svd(Gt, full_matrices=False)
                r = rank_by_energy(s, 0.90, 16)
                Dl.bases[n] = U[:, :r]
                Dl.anchors[n] = Gt.mean(1)
                Dl.U0 = U0
            per[l] = Dl
        return MultiLayerDictionary(per)

    rng0 = np.random.default_rng(7)
    t0 = time.time()
    for tstar in REP_TASKS:
        task = load_task(tstar)
        for draw in range(3):
            cond = f"random_dir_{draw}"
            if ("control", cond, tstar, "0") in done:
                continue
            U0r = {l: np.linalg.qr(rng0.standard_normal((hlm.d, 1)))[0]
                   for l in LAYERS}
            D = build_with_projection(tstar, U0r)
            Z = z_for(task, 0, 6)
            zl = z_list_from_Z(D, Z)
            code = code_for(D, zl)
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0,
                               delta_vec=np.mean(zl, axis=0))
            w.writerow(dict(exp="control", cond=cond, task=tstar, seed=0,
                            acc=eval_ops(task, ops, lys, 25)))
            fout.flush()
        print(f"control {tstar} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    fout.close()


def hparam():
    done, fout, w = _rr_writer()
    HP = [(1.0, 2.0, 1.0), (0.75, 2.0, 1.0), (1.25, 2.0, 1.0),
          (1.0, 1.5, 1.0), (1.0, 3.0, 1.0), (1.0, 2.0, 0.75),
          (1.0, 2.0, 1.25)]
    t0 = time.time()
    for tstar in REP_TASKS:
        task = load_task(tstar)
        D = loto_dictionary(tstar)
        for seed in SEEDS[:2]:
            Z = z_for(task, seed, 6)
            zl = z_list_from_Z(D, Z)
            zm = np.mean(zl, axis=0)
            code = code_for(D, zl)
            for g, b, am in HP:
                cond = f"g{g}_b{b}_a{am}"
                if ("hparam", cond, tstar, str(seed)) in done:
                    continue
                ops, lys = ops_for(D, code, g, b, am, delta_vec=zm)
                w.writerow(dict(exp="hparam", cond=cond, task=tstar,
                                seed=seed, acc=eval_ops(task, ops, lys, 25)))
                fout.flush()
        print(f"hparam {tstar} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    fout.close()


# ---------- improvements.csv stages ----------
def _imp_writer():
    return open_csv(out / "improvements.csv",
                    ["exp", "cond", "task", "seed", "acc"],
                    lambda r: (r["exp"], r["cond"], r["task"], r["seed"]))


def _signed_ops(D, code, z_mean):
    """Per-skill signed gate: correction from members aligned with z only."""
    delta = z_mean.copy()
    w = _support_weights(code)
    target = float(sum(wi * np.linalg.norm(D.anchors[n])
                       for wi, n in zip(w, code.support)))
    dn = np.linalg.norm(delta)
    if dn > 1e-8:
        delta *= target / dn
    aligned = []
    for wi, n in zip(w, code.support):
        gt = float(delta @ D.anchors[n] /
                   (np.linalg.norm(delta) * np.linalg.norm(D.anchors[n])
                    + 1e-12))
        if gt > 0.05:
            aligned.append((wi, n, gt))
    if not aligned:
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
        ops.append(_gated_op(db[l], B_l, mu_l, gate, 1.0, 2.0, 1.0))
        lys.append(l)
    return ops, lys


def _eval_prefill(task, ops, lys, n=50):
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


def variants():
    done, fout, w = _imp_writer()
    for tstar in ALL_TASKS:
        task = load_task(tstar)
        D = loto_dictionary(tstar)
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
                ops, lys = _signed_ops(D, code, z_mean)
                w.writerow(dict(exp="variant", cond="signed", task=tstar,
                                seed=seed, acc=eval_ops(task, ops, lys)))
                fout.flush()
            if "prefill_only" in need:
                ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=z_mean)
                w.writerow(dict(exp="variant", cond="prefill_only", task=tstar,
                                seed=seed, acc=_eval_prefill(task, ops, lys)))
                fout.flush()
        print(f"variants {tstar} done", flush=True)
    fout.close()


def learned():
    done, fout, w = _imp_writer()

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
        for bi, (x, y) in enumerate(examples):   # supervise answer tokens only
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
                for bi in range(enc.input_ids.shape[0]):
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

    def eval_learned(task, deltas, n=50):
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
                gen = hlm.model.generate(**enc, max_new_tokens=8,
                                         do_sample=False,
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
            w.writerow(dict(exp="learned", cond="delta40", task=tstar,
                            seed=seed, acc=eval_learned(task, deltas)))
            fout.flush()
        print(f"learned {tstar} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    fout.close()


# ---------- soft_hybrid.csv stage ----------
def soft():
    TASKS = REP_TASKS + ["person-instrument", "capitalize-last-letter",
                         "word-length", "next-capital-letter",
                         "choose-last-of-list"]
    BETAS = [0.3, 0.5, 0.7, 1.0]
    _, fout, w = open_csv(out / "soft_hybrid.csv",
                          ["task", "seed", "beta", "acc"],
                          lambda r: (r["task"], r["seed"], r["beta"]))

    def soft_op(gop, v, beta):
        vt = _to_torch(v, "cuda")

        def op(h):
            return gop(h.float()) + beta * (vt.unsqueeze(0) - h.float())
        return op

    t0 = time.time()
    for tstar in TASKS:
        task = load_task(tstar)
        D = loto_dictionary(tstar)
        q = task.eval_queries[:25]
        targets = [y for _, y in q]
        prompts = [zs_prompt(x) for x, _ in q]
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(100 * seed + K)
            idx = rng.choice(len(task.fewshot_pool), K, replace=False)
            ex = [task.fewshot_pool[i] for i in idx]
            kp = [icl_prompt([e for i, e in enumerate(ex) if i != j], x)
                  for j, (x, _) in enumerate(ex)]
            H = hlm.last_token_hiddens(kp, batch_size=8)
            V = {l: H[:, l].mean(0).numpy() for l in LAYERS}
            Z = get_z(hlm, task, K, seed)
            z_list = z_list_from_Z(D, Z)
            z_mean = np.mean(z_list, axis=0)
            code = code_for(D, z_list)
            delta = z_mean.copy()
            if code.support:
                ws = _support_weights(code)
                target = float(sum(wi * np.linalg.norm(D.anchors[n])
                                   for wi, n in zip(ws, code.support)))
                dn = np.linalg.norm(delta)
                if dn > 1e-8:
                    delta *= target / dn
                mu_full = sum(wi * D.anchors[n]
                              for wi, n in zip(ws, code.support))
                gate = max(0.0, float(delta @ mu_full /
                           (np.linalg.norm(delta) * np.linalg.norm(mu_full)
                            + 1e-12)))
            db = D.split(delta)
            for beta in BETAS:
                ops = []
                for l in LAYERS:
                    if code.support:
                        Dl = D.per_layer[l]
                        mu_l = sum(wi * Dl.anchors[n]
                                   for wi, n in zip(ws, code.support))
                        B_l = np.concatenate([Dl.bases[n]
                                              for n in code.support], axis=1)
                        gop = _gated_op(db[l], B_l, mu_l, gate, 1.0, 2.0, 1.0)
                    else:
                        gop = (lambda h, _d=db[l]:
                               h + _to_torch(_d, "cuda").unsqueeze(0))
                    ops.append(soft_op(gop, V[l], beta))
                acc = accuracy(hlm.generate(prompts, batch_size=25, op=ops,
                                            layer=list(LAYERS)), targets)
                w.writerow(dict(task=tstar, seed=seed, beta=beta, acc=acc))
                fout.flush()
        print(f"soft {tstar} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    fout.close()



def pc23():
    done, fout, w = _rr_writer()

    def build_with_pc(exclude, comp_idx):
        per = {}
        for l in LAYERS:
            names = [t for t in ALL_TASKS if t != exclude]
            means = np.stack([G[l][t].astype(np.float64).mean(0)
                              for t in names], axis=1)
            U = np.linalg.svd(means, full_matrices=False)[0]
            U0 = U[:, comp_idx:comp_idx + 1]
            Dl = build_dictionary({t: G[l][t] for t in names}, r0=0)
            for n in Dl.task_names:
                Gt = (G[l][n].astype(np.float64).T
                      - U0 @ (U0.T @ G[l][n].astype(np.float64).T))
                Uu, sv, _ = np.linalg.svd(Gt, full_matrices=False)
                r = rank_by_energy(sv, 0.90, 16)
                Dl.bases[n] = Uu[:, :r]
                Dl.anchors[n] = Gt.mean(1)
                Dl.U0 = U0
            per[l] = Dl
        return MultiLayerDictionary(per)

    t0 = time.time()
    for tstar in REP_TASKS:
        task = load_task(tstar)
        for comp_idx, cond in [(1, "pc2"), (2, "pc3")]:
            D = build_with_pc(tstar, comp_idx)
            for seed in SEEDS:
                if ("pc_control", cond, tstar, str(seed)) in done:
                    continue
                Z = z_for(task, seed, 6)
                zl = z_list_from_Z(D, Z)
                code = code_for(D, zl)
                ops, lys = ops_for(D, code, 1.0, 2.0, 1.0,
                                   delta_vec=np.mean(zl, axis=0))
                w.writerow(dict(exp="pc_control", cond=cond, task=tstar,
                                seed=seed, acc=eval_ops(task, ops, lys, 25)))
                fout.flush()
        print(f"pc23 {tstar} done ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    fout.close()


def _alpha_const_op(dl, B_l, mu_l, gate, gamma, alpha_max):
    """_gated_op with the adaptive alpha(h) frozen at alpha_max."""
    import torch
    Q, _ = np.linalg.qr(B_l)
    Qt = _to_torch(Q, "cuda")
    dvec = _to_torch(dl, "cuda")
    mu = _to_torch(mu_l, "cuda")

    def op(h):
        h = h.float()
        diff = mu.unsqueeze(0) - h
        proj = (diff @ Qt) @ Qt.T
        eff = gate * alpha_max + (1.0 - gate)
        return h + eff * gamma * dvec.unsqueeze(0) + gate * alpha_max * proj
    return op


def alphaiso():
    done, fout, w = _imp_writer()
    for tstar in ALL_TASKS:
        task = load_task(tstar)
        D = loto_dictionary(tstar)
        for seed in [0, 1, 2]:
            if ("variant", "alpha_const", tstar, str(seed)) in done:
                continue
            Z = get_z(hlm, task, K, seed)
            z_list = z_list_from_Z(D, Z)
            z_mean = np.mean(z_list, axis=0)
            code = code_for(D, z_list)
            delta = z_mean.copy()
            if code.support:
                ws = _support_weights(code)
                target = float(sum(wi * np.linalg.norm(D.anchors[n])
                                   for wi, n in zip(ws, code.support)))
                dn = np.linalg.norm(delta)
                if dn > 1e-8:
                    delta *= target / dn
                mu_full = sum(wi * D.anchors[n]
                              for wi, n in zip(ws, code.support))
                gate = max(0.0, float(delta @ mu_full /
                           (np.linalg.norm(delta) * np.linalg.norm(mu_full)
                            + 1e-12)))
                db = D.split(delta)
                ops, lys = [], []
                for l in D.layers:
                    Dl = D.per_layer[l]
                    mu_l = sum(wi * Dl.anchors[n]
                               for wi, n in zip(ws, code.support))
                    B_l = np.concatenate([Dl.bases[n] for n in code.support],
                                         axis=1)
                    ops.append(_alpha_const_op(db[l], B_l, mu_l, gate,
                                               1.0, 1.0))
                    lys.append(l)
            else:
                ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=z_mean)
            w.writerow(dict(exp="variant", cond="alpha_const", task=tstar,
                            seed=seed, acc=eval_ops(task, ops, lys)))
            fout.flush()
        print(f"alphaiso {tstar} done", flush=True)
    fout.close()


def osupp():
    from cass.compound import COMPOUND_REGISTRY, load_compound, \
        compound_components
    from cass.solver import SparseCode
    D = build_multilayer_dictionary(G, r0=1)
    import csv as _csv
    path = out / "fill_gaps.csv"
    done = set()
    with open(path) as fh:
        done = {(r["suite"], r["method"], r["task"], r["seed"])
                for r in _csv.DictReader(fh)}
    fout = open(path, "a", newline="")
    w = _csv.DictWriter(fout, fieldnames=["suite", "method", "task",
                                          "seed", "acc"])
    from cass.evaluate import accuracy
    from cass.tasks import zs_prompt as _zsp
    for cname in COMPOUND_REGISTRY:
        comp = load_compound(cname)
        comps = [c for c in compound_components(cname) if c in D.task_names]
        for seed in [0, 1, 2]:
            if ("compound", "oracle_support", cname, str(seed)) in done:
                continue
            Z = get_z(hlm, comp, K, seed)
            z_list = z_list_from_Z(D, Z)
            z_mean = np.mean(z_list, axis=0)
            # oracle support: true constituents, ls coefficients (orthonormal
            # blocks -> projections), z direction unchanged
            coeffs = {t: D.bases[t].T @ z_mean for t in comps}
            delta = sum(D.bases[t] @ coeffs[t] for t in comps)
            res = float(np.linalg.norm(z_mean - delta)
                        / (np.linalg.norm(z_mean) + 1e-12))
            code = SparseCode(coeffs, comps, delta, res, 0.0)
            ops, lys = ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=z_mean)
            q = comp.eval_queries[:50]
            acc = accuracy(hlm.generate([_zsp(x) for x, _ in q],
                                        batch_size=25, op=ops, layer=lys),
                           [y for _, y in q], case_sensitive=True)
            w.writerow(dict(suite="compound", method="oracle_support",
                            task=cname, seed=seed, acc=acc))
            fout.flush()
        print(f"osupp {cname} done", flush=True)
    fout.close()


if __name__ == "__main__":
    stages = (["denoise", "control", "hparam", "variants", "learned", "soft"]
              if STAGES == ["all"] else STAGES)
    for s in stages:
        {"denoise": denoise, "control": control, "hparam": hparam,
         "variants": variants, "learned": learned, "soft": soft,
         "pc23": pc23, "alphaiso": alphaiso, "osupp": osupp}[s]()
