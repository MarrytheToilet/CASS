"""Shared helpers for logged, split-aware night experiments."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cass.tasks import ALL_TASKS, load_task
from cass.dictionary import build_multilayer_dictionary
from cass.evaluate import accuracy

DEV_TASKS = ['country-capital', 'person-sport', 'antonym', 'present-past',
             'singular-plural', 'english-french', 'english-spanish', 'next-item',
             'word-length', 'choose-first-of-list', 'animal-from-list', 'color-from-list']


def dev_split(task, seed):
    nval = min(12, len(task.fewshot_pool) - 4)
    assert nval >= 4
    queries = task.fewshot_pool[:nval]
    pool = task.fewshot_pool[nval:]
    rng = np.random.default_rng(100 * seed + 4)
    demos = [pool[i] for i in rng.choice(len(pool), 4, replace=False)]
    assert not ({x for x, _ in queries} & {x for x, _ in demos})
    assert not ({x for x, _ in queries} & {x for x, _ in task.eval_queries})
    assert not ({x for x, _ in queries} & {x for x, _ in task.dict_pool})
    return demos, queries


def load_g(model, layers):
    out = {l: {} for l in layers}
    for name in ALL_TASKS:
        blob = torch.load(ROOT/'results'/model/'activations'/f'{name}.pt',
                          map_location='cpu', weights_only=True)
        for l in layers:
            out[l][name] = blob['G'][:, l].float().numpy()
    return out


def make_dict(G, layers, exclude=None):
    return build_multilayer_dictionary(
        {l: {n: a for n, a in G[l].items() if n != exclude} for l in layers}, r0=1)


@torch.no_grad()
def generate(hlm, prompts, ops=None, layers=None, schedule='all', positions=1,
             batch_size=25, max_new_tokens=8, stop_after_answer=False):
    texts = []
    for i in range(0, len(prompts), batch_size):
        enc = hlm.tok(prompts[i:i+batch_size], return_tensors='pt', padding=True).to(hlm.device)
        mask0 = enc.attention_mask.bool()
        handles = []
        for op, layer in zip(ops or [], layers or []):
            step = [0]
            def hook(module, inputs, output, operation=op, counter=step):
                h = output[0] if isinstance(output, tuple) else output
                index = counter[0]
                counter[0] += 1
                if schedule == 'prefill' and index > 0:
                    return
                if schedule == 'first3' and index >= 3:
                    return
                if schedule == 'decode' and index == 0:
                    return
                if index == 0 and positions != 1:
                    mask = mask0.clone()
                    if positions != 'all':
                        mask[:, :-int(positions)] = False
                    h[mask] = operation(h[mask]).to(h.dtype)
                else:
                    h[:, -1, :] = operation(h[:, -1, :]).to(h.dtype)
            handles.append(hlm.layers[layer-1].register_forward_hook(hook))
        try:
            extra = {}
            if stop_after_answer:
                from answer_stop import AnswerStop
                from transformers import StoppingCriteriaList
                extra['stopping_criteria'] = StoppingCriteriaList([AnswerStop(hlm.tok,enc.input_ids.shape[1])])
            out = hlm.model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=hlm.tok.pad_token_id, **extra)
            texts.extend(hlm.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
        finally:
            for handle in handles:
                handle.remove()
    return texts


class Ledger:
    def __init__(self, stem):
        self.path = HERE / f'{stem}.jsonl'
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.done.add((row['task'], row['seed'], row['config']))
        self.file = self.path.open('a')

    def has(self, task, seed, config):
        return (task, seed, config) in self.done

    def add(self, task, seed, config, queries, preds, **extra):
        row = dict(task=task, seed=seed, config=config, n=len(queries),
                   acc=accuracy(preds, [y for _, y in queries], case_sensitive='+' in task),
                   queries=queries, predictions=preds, **extra)
        self.file.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.file.flush()
        self.done.add((task, seed, config))
        return row['acc']


def correction_ops(D, code, zlist, gamma=1., correction=1., shrink=0., rescale=True,
                   selected_layers=None):
    """Original gated formula with explicit ablation knobs; no fitted weights."""
    z = np.mean(zlist, axis=0).copy()
    weights = np.array([np.linalg.norm(code.coeffs[n]) for n in code.support])
    weights = weights / weights.sum() if len(weights) else weights
    mu = sum((w * D.anchors[n] for w, n in zip(weights, code.support)), start=np.zeros_like(z))
    Qs = {}
    for l in D.layers:
        if code.support:
            Qs[l] = np.linalg.qr(np.concatenate([D.per_layer[l].bases[n] for n in code.support], axis=1))[0]
        else:
            Qs[l] = np.zeros((D.d, 0))
    if shrink:
        for j, l in enumerate(D.layers):
            sl = slice(j*D.d, (j+1)*D.d)
            Q = Qs[l]
            residuals = np.asarray(zlist)[:, sl]
            residuals = residuals - (residuals @ Q) @ Q.T
            mean = residuals.mean(0)
            noise = np.sum((residuals - mean)**2) / max(1, len(zlist)*(len(zlist)-1))
            fraction = min(1., noise / (np.sum(mean**2) + 1e-12))
            z[sl] -= shrink * fraction * mean
    if rescale and len(weights):
        target = sum(w*np.linalg.norm(D.anchors[n]) for w,n in zip(weights,code.support))
        z *= target / (np.linalg.norm(z)+1e-12)
    gate = max(0., float(z@mu/(np.linalg.norm(z)*np.linalg.norm(mu)+1e-12)))
    ops, layers = [], []
    for l in selected_layers or D.layers:
        delta = torch.tensor(D.split(z)[l], device='cuda', dtype=torch.float32)
        anchor = torch.tensor(D.split(mu)[l], device='cuda', dtype=torch.float32)
        Q = torch.tensor(Qs[l], device='cuda', dtype=torch.float32)
        def operation(h, d=delta, m=anchor, q=Q):
            h = h.float()
            if not code.support:
                return h + gamma*d
            diff = m-h
            proj = (diff@q)@q.T
            alpha = (2*(diff-proj).norm(dim=1)/(h.norm(dim=1)+1e-8)).clamp(max=1).unsqueeze(1)
            eff = gate*alpha+(1-gate)
            return h + eff*gamma*d + correction*gate*alpha*proj
        ops.append(operation)
        layers.append(l)
    return ops, layers
