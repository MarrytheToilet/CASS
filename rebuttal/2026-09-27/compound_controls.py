"""Matched correction swaps on Compound-10, seeds 0/1/2.

All conditions retain the inferred operator's additive direction, norm, gate,
and state-dependent alpha. Only the corrective projector/anchor changes.
Wrong supports exclude both inferred and ground-truth skills, match each block's
rank, and minimize deviation in within-support mean and maximum coherence.
No predictions or evaluation labels enter control selection.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_evidence import dictionary
from cass.compound import COMPOUND_REGISTRY, load_compound, compound_components
from cass.dictionary import subcoherence_matrix
from cass.evaluate import accuracy
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z
from cass.solver import SparseCode
from cass.tasks import zs_prompt, icl_prompt


def weights(code):
    w = np.array([np.linalg.norm(code.coeffs[t]) for t in code.support])
    return w/w.sum()


def geometry(D, names, w):
    mu = sum(wi*D.anchors[t] for wi, t in zip(w, names))
    Q = {}
    for l in D.layers:
        Q[l] = np.linalg.qr(np.concatenate([D.per_layer[l].bases[t] for t in names], axis=1))[0]
    return mu, Q


def controlled_ops(D, code, z, names=None, new_weights=None, off=False):
    w = weights(code)
    ref_mu, ref_Q = geometry(D, code.support, w)
    target = sum(wi*np.linalg.norm(D.anchors[t]) for wi, t in zip(w, code.support))
    delta = z * (target/max(np.linalg.norm(z), 1e-12))
    gate = max(0., float(delta@ref_mu/(np.linalg.norm(delta)*np.linalg.norm(ref_mu)+1e-12)))
    alt_mu, alt_Q = geometry(D, names or code.support, new_weights if new_weights is not None else w)
    def tensor(v):
        return torch.as_tensor(v, device="cuda", dtype=torch.float32)
    out = []
    for l in D.layers:
        qref, qalt = tensor(ref_Q[l]), tensor(alt_Q[l])
        mref, malt = tensor(D.split(ref_mu)[l]), tensor(D.split(alt_mu)[l])
        dv = tensor(D.split(delta)[l])
        def op(h, qr=qref, qa=qalt, mr=mref, ma=malt, d=dv):
            h = h.float()
            diff = mr.unsqueeze(0)-h
            orth = diff - (diff@qr)@qr.T
            alpha = (2.*orth.norm(dim=1)/(h.norm(dim=1)+1e-8)).clamp(max=1.).unsqueeze(1)
            eff = gate*alpha + (1.-gate)
            correction = ((ma.unsqueeze(0)-h)@qa)@qa.T
            return h + eff*d.unsqueeze(0) + (0. if off else gate)*alpha*correction
        out.append(op)
    return out, gate


def wrong_supports(D, names, truths, coherence, rng):
    pos = {t: i for i,t in enumerate(D.task_names)}
    def stats(s):
        if len(s) < 2:
            return np.zeros(2)
        a = coherence[np.ix_([pos[t] for t in s], [pos[t] for t in s])]
        v = a[np.triu_indices(len(s), 1)]
        return np.array([v.mean(), v.max()])
    ref = stats(names)
    excluded = set(names)|set(truths)
    pools = [[t for t in D.task_names if t not in excluded and D.bases[t].shape[1] == D.bases[n].shape[1]] for n in names]
    if any(not x for x in pools):
        return [], {"reason": "No distinct rank-matched alternative for one block", "reference": ref.tolist()}
    candidates = {}
    for _ in range(4000):
        cand = tuple(rng.choice(p) for p in pools)
        if len(set(cand)) != len(cand):
            continue
        v = stats(cand)
        key = tuple(sorted(cand))
        if key not in candidates:
            candidates[key] = (float(np.linalg.norm((v-ref)/np.array([.02,.03]))), cand, v)
    selected = sorted(candidates.values(), key=lambda x:(x[0],x[1]))[:3]
    meta = {"reference": ref.tolist(), "candidate_count": len(candidates), "selected": []}
    for score, cand, v in selected:
        meta["selected"].append(dict(support=cand, coherence=v.tolist(),
                                     mean_gap=float(abs(v[0]-ref[0])), max_gap=float(abs(v[1]-ref[1])),
                                     tolerance_pass=bool(abs(v[0]-ref[0])<=.02 and abs(v[1]-ref[1])<=.03)))
    return [list(x[1]) for x in selected], meta


def main():
    torch.set_num_threads(1)
    _, D = dictionary()
    coherence = np.maximum.reduce([subcoherence_matrix(D.per_layer[l]) for l in D.layers])
    hlm = HookedLM("llama31-8b")
    result_path = HERE / "compound_controls.csv"
    fields = ["task", "seed", "condition", "acc", "gate", "support", "matched_coherence"]
    done = set()
    if result_path.exists():
        with result_path.open() as f:
            done = {(r["task"],int(r["seed"]),r["condition"]) for r in csv.DictReader(f)}
    meta_path = HERE / "compound_control_matching.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metadata["runtime"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    with result_path.open("a", newline="") as f, (HERE/"compound_control_generations.jsonl").open("a") as gens:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for ti, name in enumerate(COMPOUND_REGISTRY):
            task = load_compound(name)
            truths = list(compound_components(name))
            prompts = [zs_prompt(x) for x,_ in task.eval_queries]
            targets = [y for _,y in task.eval_queries]
            for seed in [0,1,2]:
                Z = torch.load(ROOT/"results/llama31-8b/zcache"/f"{name}_k4_s{seed}.pt", map_location="cpu", weights_only=True).float()
                zl = z_list_from_Z(D, Z)
                z = np.mean(zl, axis=0)
                code = code_for(D, zl)
                if not code.support:
                    raise ValueError(f"Empty inferred support: {name}/{seed}")
                wrong, meta = wrong_supports(D, code.support, truths, coherence, np.random.default_rng(27000+ti*10+seed))
                metadata[f"{name}|{seed}"] = meta
                tw = np.array([np.linalg.norm(D.bases[t].T@z) for t in truths])
                tw /= tw.sum()
                conditions = [("inferred",code.support,None,False), ("no_correction",code.support,None,True),
                              ("true_correction",truths,tw,False),
                              ("z_only",code.support,None,False),("true_support",truths,tw,False),
                              ("replace",code.support,None,False)]
                conditions += [(f"wrong_{i}",s,None,False) for i,s in enumerate(wrong)]
                # Verify our inferred operator matches the original implementation.
                original, _ = ops_for(D, code, 1., 2., 1., delta_vec=z)
                inferred, _ = controlled_ops(D, code, z)
                probe = torch.as_tensor(np.stack([z[:D.d], np.ones(D.d)]), dtype=torch.float32, device="cuda")
                for a,b in zip(original,inferred):
                    torch.testing.assert_close(a(probe),b(probe),atol=2e-5,rtol=2e-5)
                for cond, support, cw, off in conditions:
                    if (name,seed,cond) in done:
                        continue
                    ops, gate = controlled_ops(D,code,z,support,cw,off)
                    if cond == "z_only":
                        ops,_ = ops_for(D,code,1.,2.,1.,injection="additive",delta_vec=z)
                        gate = None
                    elif cond == "true_support":
                        coefficients = {t:D.bases[t].T@z for t in truths}
                        delta = sum(D.bases[t]@coefficients[t] for t in truths)
                        tc = SparseCode(coefficients,truths,delta,float(np.linalg.norm(z-delta)/np.linalg.norm(z)),0.)
                        ops,_ = ops_for(D,tc,1.,2.,1.,delta_vec=z)
                        cmu = sum(wi*D.anchors[t] for wi,t in zip(tw,truths))
                        gate = max(0.,float(z@cmu/(np.linalg.norm(z)*np.linalg.norm(cmu)+1e-12)))
                    elif cond == "replace":
                        rg = np.random.default_rng(100*seed+4)
                        ex = [task.fewshot_pool[i] for i in rg.choice(len(task.fewshot_pool),4,replace=False)]
                        pp = [icl_prompt([e for i,e in enumerate(ex) if i!=j],x) for j,(x,_) in enumerate(ex)]
                        H = hlm.last_token_hiddens(pp,batch_size=8)
                        ops = []
                        for l in D.layers:
                            v = H[:,l].mean(0).to(device="cuda",dtype=torch.float32)
                            def replace(h,vec=v):
                                return vec.unsqueeze(0).expand_as(h).clone()
                            ops.append(replace)
                        gate = None
                    preds = hlm.generate(prompts,batch_size=25,op=ops,layer=D.layers)
                    acc = accuracy(preds,targets,case_sensitive=True)
                    passed = meta["selected"][int(cond.split('_')[1])]["tolerance_pass"] if cond.startswith("wrong_") else ""
                    writer.writerow(dict(task=name,seed=seed,condition=cond,acc=acc,gate=gate,
                                         support="|".join(support),matched_coherence=passed))
                    for (x,y),pred in zip(task.eval_queries,preds):
                        gens.write(json.dumps(dict(task=name,seed=seed,condition=cond,input=x,target=y,pred=pred))+"\n")
                    f.flush(); gens.flush()
                    print(name,seed,cond,round(acc,4),"gate",round(gate,3) if gate is not None else None,flush=True)
                meta_path.write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
