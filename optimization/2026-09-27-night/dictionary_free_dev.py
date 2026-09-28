"""Fair development tuning of dictionary-free directions and state interpolation."""
import json
import os
import numpy as np
import torch
from common import HERE,DEV_TASKS,dev_split,generate,Ledger
from efficient_ops import selected_hiddens
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
hlm=HookedLM('llama31-8b');layers=[12,16];ledger=Ledger('dictionary_free_dev')
grid=[]
for fraction in [0.,.25,.5,1.]:
    for mode in ['mean','pc1']:
        for lys in [[12],[16],[12,16]]:
            for gain in [.03125,.0625,.125,.25,.5,1.,2.]:
                for schedule in ['all','prefill']:
                    grid.append(dict(null_fraction=fraction,mode=mode,layers=lys,gamma=gain,schedule=schedule))
for lys in [[12],[16],[12,16]]:
    for gain in [.25,.5,.75,1.]:
        for schedule in ['all','prefill']:
            grid.append(dict(mode='state_interpolation',layers=lys,gamma=gain,schedule=schedule))
(HERE/'dictionary_free_grid.json').write_text(json.dumps(grid,indent=2))
for name in DEV_TASKS:
    task=load_task(name)
    for seed in [10,11]:
        ex,queries=dev_split(task,seed)
        raw=torch.load(HERE/f'cache_llama_dev_{name}_{seed}.pt',weights_only=True)
        null=torch.load(HERE/f'cache_null_dev_{name}_{seed}.pt',weights_only=True)
        baseline=selected_hiddens(hlm,[zs_prompt(x) for x,y in ex],layers,batch_size=4)
        positive={l:(null[l]+baseline[l]).mean(0).numpy() for l in layers}
        directions={}
        for fraction in [0.,.25,.5,1.]:
            for lys in [[12],[16],[12,16]]:
                V=np.concatenate([((1-fraction)*raw[:,l]+fraction*null[l]).numpy() for l in lys],axis=1)
                mean=V.mean(0)
                pc=np.linalg.svd(V,full_matrices=False)[2][0]
                if pc@mean<0:pc=-pc
                pc*=np.linalg.norm(mean)
                for mode,d in [('mean',mean),('pc1',pc)]:
                    directions[(fraction,tuple(lys),mode)]=[torch.tensor(d[i*hlm.d:(i+1)*hlm.d],device='cuda') for i in range(len(lys))]
        for cfg in grid:
            label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
            if ledger.has(name,seed,label):continue
            lys=cfg['layers'];gain=cfg['gamma'];ops=[]
            if cfg['mode']=='state_interpolation':
                for l in lys:
                    target=torch.as_tensor(positive[l],device='cuda')
                    def op(h,v=target,g=gain):return h.float()+g*(v-h.float())
                    ops.append(op)
            else:
                for d in directions[(cfg['null_fraction'],tuple(lys),cfg['mode'])]:
                    def op(h,v=d,g=gain):return h.float()+g*v
                    ops.append(op)
            pp=[zs_prompt(x) for x,y in queries]
            preds=generate(hlm,pp,ops,lys,cfg['schedule'])
            ledger.add(name,seed,label,queries,preds,demos=ex,dictionary_used=False)
        print(name,seed,'done',flush=True)
(HERE/'dictionary_free_dev_done.json').write_text('{"complete":true}\n')
