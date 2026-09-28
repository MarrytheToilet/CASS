"""Development-only interaction of improved contrast with safer correction geometry."""
import json
import os
import numpy as np
import torch
from common import HERE,DEV_TASKS,ALL_TASKS,dev_split,load_g,make_dict,generate,Ledger
from variant_ops import OperatorFactory
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt
from cass.pipeline import z_list_from_Z

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
hlm=HookedLM('llama31-8b');layers=[12,16]
raw=load_g('llama31-8b',layers);null={l:{} for l in layers};clean={}
for name in ALL_TASKS:
    b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
    for l in layers:null[l][name]=b['G_by_layer'][l].float().numpy()
    clean[name]={l:b['clean_mean'][l].numpy() for l in layers}
ledger=Ledger('combined_dev')
grid=[]
for fraction in [.25,1.]:
    for kind in ['original','rank16','context_residual']:
        for gain in [1.,1.5]:
            for correction in [.25,.5,1.,'coverage','uncertainty']:
                for schedule in ['all','prefill']:
                    grid.append(dict(null_fraction=fraction,kind=kind,gamma=gain,correction=correction,schedule=schedule))
(HERE/'combined_grid.json').write_text(json.dumps(grid,indent=2))
for name in DEV_TASKS:
    task=load_task(name)
    for fraction in [.25,1.]:
        G={l:{n:(1-fraction)*raw[l][n]+fraction*null[l][n] for n in ALL_TASKS} for l in layers}
        D=make_dict(G,layers,exclude=name);solver=GramSolver(D)
        for seed in [10,11]:
            ex,queries=dev_split(task,seed)
            Z=torch.load(HERE/f'cache_llama_dev_{name}_{seed}.pt',weights_only=True).clone()
            zn=torch.load(HERE/f'cache_null_dev_{name}_{seed}.pt',weights_only=True)
            for l in layers:Z[:,l]=(1-fraction)*Z[:,l]+fraction*zn[l]
            zl=z_list_from_Z(D,Z);code=solver.solve(zl);factory=OperatorFactory(D,code,zl,clean)
            V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
            agreement=float((V@V.T)[np.triu_indices(4,1)].mean())
            coverage=float(max(0.,1-code.residual**2));uncertainty=float(np.clip(1-agreement,0,1))
            pp=[zs_prompt(x) for x,y in queries]
            for cfg in grid:
                if cfg['null_fraction']!=fraction:continue
                label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
                if ledger.has(name,seed,label):continue
                strength={'coverage':coverage,'uncertainty':uncertainty}.get(cfg['correction'],cfg['correction'])
                ops,lys=factory.ops(cfg['kind'],cfg['gamma'],strength)
                preds=generate(hlm,pp,ops,lys,cfg['schedule'])
                acc=ledger.add(name,seed,label,queries,preds,support=code.support,residual=code.residual,
                               applied_correction=strength,agreement=agreement,demos=ex)
            print(name,fraction,seed,'done',flush=True)
(HERE/'combined_dev_done.json').write_text('{"complete":true}\n')
