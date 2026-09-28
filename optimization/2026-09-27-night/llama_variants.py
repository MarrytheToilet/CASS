"""Independent-query development scan of compressed-prompt and operator variants."""
import json
import os
import random
import time
import numpy as np
import torch
from common import HERE,DEV_TASKS,dev_split,load_g,make_dict,generate,Ledger,ALL_TASKS
from variant_ops import OperatorFactory
from efficient_ops import selected_hiddens
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt,icl_prompt,build_pair_prompts
from cass.extract import extract_fewshot_z
from cass.pipeline import z_list_from_Z

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
hlm=HookedLM('llama31-8b');layers=[12,16];G=load_g('llama31-8b',layers)
anchors=HERE/'llama_clean_anchors.pt'
if anchors.exists():clean=torch.load(anchors,weights_only=True)
else:
    clean={}
    for name in ALL_TASKS:
        pp,_=build_pair_prompts(load_task(name).dict_pool,100,10,random.Random(7000))
        H=selected_hiddens(hlm,pp,layers,batch_size=16)
        clean[name]={l:h.mean(0).numpy() for l,h in H.items()}
        print('clean anchor',name,flush=True)
    # Store plain tensors for safe reload.
    torch.save({n:{l:torch.tensor(a) for l,a in row.items()} for n,row in clean.items()},anchors)
clean={n:{l:np.asarray(a) for l,a in row.items()} for n,row in clean.items()}
configs=[]
for shots in [0,1]:
    gains=[.5,1.,1.5,2.] if shots==0 else [.25,.5,1.,1.5]
    for kind in ['original','rank16','rank32','context','context_residual']:
        for gain in gains:
            for schedule in ['all','prefill']:
                configs.append(dict(shots=shots,kind=kind,gamma=gain,correction=1.,schedule=schedule))
    for gain in gains:
        for schedule in ['all','prefill']:
            configs.append(dict(shots=shots,kind='original',gamma=gain,correction=0.,schedule=schedule))
(HERE/'llama_variants_grid.json').write_text(json.dumps(configs,indent=2))
ledger=Ledger('llama_variants_dev');start=time.time()
for name in DEV_TASKS:
    task=load_task(name);D=make_dict(G,layers,exclude=name);solver=GramSolver(D)
    for seed in [10,11]:
        demos,queries=dev_split(task,seed)
        cache=HERE/f'cache_llama_dev_{name}_{seed}.pt'
        Z=torch.load(cache,weights_only=True) if cache.exists() else extract_fewshot_z(hlm,demos,seed=seed,batch_size=24)
        zl=z_list_from_Z(D,Z);code=solver.solve(zl)
        factory=OperatorFactory(D,code,zl,clean)
        p0=[zs_prompt(x) for x,y in queries]
        p1=[icl_prompt(demos[:1],x) for x,y in queries]
        for mode,pp in [('zero',p0),('icl1',p1),('icl4',[icl_prompt(demos,x) for x,y in queries])]:
            if not ledger.has(name,seed,mode):ledger.add(name,seed,mode,queries,generate(hlm,pp),demos=demos)
        for cfg in configs:
            label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
            if ledger.has(name,seed,label):continue
            ops,lys=factory.ops(cfg['kind'],cfg['gamma'],cfg['correction'])
            pp=p1 if cfg['shots'] else p0
            ledger.add(name,seed,label,queries,generate(hlm,pp,ops,lys,cfg['schedule']),
                       support=code.support,residual=code.residual,demos=demos)
        print(name,seed,'done',round(time.time()-start,1),flush=True)
(HERE/'llama_variants_done.json').write_text(json.dumps(dict(elapsed=time.time()-start)))
