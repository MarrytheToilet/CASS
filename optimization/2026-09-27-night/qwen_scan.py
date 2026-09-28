"""Development-only oracle layer/gain/schedule scan. Does not tune on eval."""
import json
import os
import time
import numpy as np
import torch
from common import HERE, DEV_TASKS, dev_split, load_g, make_dict, generate, Ledger
from cass.models import HookedLM
from cass.steer import make_additive_op, make_affine_op
from cass.tasks import load_task, zs_prompt, icl_prompt

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
hlm = HookedLM('qwen3-4b')
layers = [4, 8, 12, 16, 20, 24, 28, 32, 34]
G = load_g('qwen3-4b', layers)
D = {l: make_dict(G, [l]).per_layer[l] for l in layers}
ledger = Ledger('qwen_oracle_dev')
start = time.time()
for name in DEV_TASKS:
    task = load_task(name)
    demos, queries = dev_split(task, 10)
    prompts = [zs_prompt(x) for x,y in queries]
    for mode in ['zero', 'icl4']:
        if ledger.has(name, 10, mode): continue
        pp = prompts if mode == 'zero' else [icl_prompt(demos,x) for x,y in queries]
        ledger.add(name,10,mode,queries,generate(hlm,pp),demos=demos)
    for l in layers:
        dl = D[l]
        for kind in ['raw_add', 'clean_add', 'affine']:
            for gain in [.25, .5, 1., 2., 4.]:
                for schedule in ['all','prefill']:
                    label = f'{kind}_l{l}_g{gain}_{schedule}'
                    if ledger.has(name,10,label): continue
                    if kind == 'affine':
                        op = make_affine_op(dl.anchors[name],dl.bases[name],dl.anchors[name],gamma=gain,beta=2.,alpha_max=1.)
                    else:
                        vec = dl.raw_means[name] if kind == 'raw_add' else dl.anchors[name]
                        op = make_additive_op(vec,gamma=gain)
                    ledger.add(name,10,label,queries,generate(hlm,prompts,[op],[l],schedule),
                               scope='known-task oracle, development split')
    print(name, 'done', round(time.time()-start,1), flush=True)
(HERE/'qwen_oracle_dev_done.json').write_text(json.dumps(dict(elapsed=time.time()-start, configs=270, tasks=DEV_TASKS)))
