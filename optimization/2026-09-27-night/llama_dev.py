"""LOTO development scan; validation inputs and labels never enter extraction."""
import json
import os
import time
import numpy as np
import torch
from common import HERE, DEV_TASKS, dev_split, load_g, make_dict, generate, Ledger, correction_ops
from cass.models import HookedLM
from cass.tasks import load_task, zs_prompt, icl_prompt
from cass.extract import extract_fewshot_z
from cass.pipeline import code_for, z_list_from_Z, ops_for

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
hlm = HookedLM('llama31-8b')
layers = [12,16]
G = load_g('llama31-8b', layers)
ledger = Ledger('llama_dev')
configs = []
for gamma in [.5,.75,1.,1.5,2.]:
    for correction in [0.,.5,1.,2.]:
        for schedule in ['all','prefill']:
            configs.append(dict(gamma=gamma,correction=correction,schedule=schedule))
for shrink in [.25,.5,1.]:
    for gamma in [.75,1.,1.5]:
        configs.append(dict(gamma=gamma,shrink=shrink))
for gamma in [.5,1.,2.,4.]:
    configs.append(dict(gamma=gamma,rescale=False))
for l in layers:
    for gamma in [1.,2.,4.]:
        for schedule in ['all','prefill']:
            configs.append(dict(gamma=gamma,selected_layers=[l],schedule=schedule))
(HERE/'llama_dev_grid.json').write_text(json.dumps(configs,indent=2))
start = time.time()
for name in DEV_TASKS:
    D = make_dict(G,layers,exclude=name)
    task = load_task(name)
    for seed in [10,11]:
        demos,queries = dev_split(task,seed)
        prompts = [zs_prompt(x) for x,y in queries]
        cache = HERE/f'cache_llama_dev_{name}_{seed}.pt'
        if cache.exists():
            Z = torch.load(cache,weights_only=True)
        else:
            Z = extract_fewshot_z(hlm,demos,seed=seed,batch_size=24)
            torch.save(Z,cache)
        zl = z_list_from_Z(D,Z)
        code = code_for(D,zl)
        # Verify the configurable default exactly implements the original op.
        new, lys = correction_ops(D,code,zl)
        old, _ = ops_for(D,code,delta_vec=np.mean(zl,axis=0))
        probe = torch.randn(7,hlm.d,device='cuda')
        for a,b in zip(new,old):
            torch.testing.assert_close(a(probe),b(probe),rtol=2e-5,atol=2e-5)
        for mode in ['zero','icl4']:
            if ledger.has(name,seed,mode): continue
            pp = prompts if mode=='zero' else [icl_prompt(demos,x) for x,y in queries]
            ledger.add(name,seed,mode,queries,generate(hlm,pp),demos=demos)
        for kw in configs:
            label = json.dumps(kw,sort_keys=True,separators=(',',':'))
            if ledger.has(name,seed,label): continue
            args = kw.copy()
            schedule = args.pop('schedule','all')
            ops,lys = correction_ops(D,code,zl,**args)
            ledger.add(name,seed,label,queries,generate(hlm,prompts,ops,lys,schedule),
                       support=code.support,residual=code.residual,znorm=float(np.linalg.norm(np.mean(zl,axis=0))),demos=demos)
        print(name,seed,'done',round(time.time()-start,1),flush=True)
(HERE/'llama_dev_done.json').write_text(json.dumps(dict(elapsed=time.time()-start,configs=len(configs),tasks=DEV_TASKS)))
