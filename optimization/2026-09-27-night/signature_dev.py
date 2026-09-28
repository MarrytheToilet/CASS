"""Compare label-shuffle and no-demonstration contrast, with matching dictionaries."""
import json
import os
import random
import time
import numpy as np
import torch
from common import HERE,DEV_TASKS,ALL_TASKS,dev_split,load_g,make_dict,generate,Ledger,correction_ops
from efficient_ops import selected_hiddens
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,build_pair_prompts,build_fewshot_pair_prompts,zs_prompt,icl_prompt
from cass.extract import extract_fewshot_z
from cass.pipeline import z_list_from_Z


def main():
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    hlm=HookedLM('llama31-8b');layers=[12,16]
    raw=load_g('llama31-8b',layers);null={l:{} for l in layers}
    folder=HERE/'llama_null_activations';folder.mkdir(exist_ok=True)
    clean_anchors={};start=time.time()
    for name in ALL_TASKS:
        path=folder/f'{name}.pt'
        if path.exists():blob=torch.load(path,weights_only=True)
        else:
            task=load_task(name);pp,_=build_pair_prompts(task.dict_pool,100,10,random.Random(7000))
            p0=[zs_prompt(task.dict_pool[i%len(task.dict_pool)][0]) for i in range(100)]
            hp=selected_hiddens(hlm,pp,layers,batch_size=16)
            hz=selected_hiddens(hlm,p0,layers,batch_size=25)
            blob=dict(G_by_layer={l:(hp[l]-hz[l]).half() for l in layers},
                      clean_mean={l:hp[l].mean(0) for l in layers},n_pairs=100,n_shots=10)
            torch.save(blob,path)
        for l in layers:null[l][name]=blob['G_by_layer'][l].float().numpy()
        clean_anchors[name]=blob['clean_mean']
        print('null dictionary',name,round(time.time()-start,1),flush=True)
    if not (HERE/'llama_clean_anchors.pt').exists():torch.save(clean_anchors,HERE/'llama_clean_anchors.pt')
    ledger=Ledger('signature_dev')
    for name in DEV_TASKS:
        task=load_task(name)
        zcache={}
        for seed in [10,11]:
            demos,queries=dev_split(task,seed)
            path=HERE/f'cache_llama_dev_{name}_{seed}.pt'
            original=torch.load(path,weights_only=True) if path.exists() else extract_fewshot_z(hlm,demos,seed=seed,batch_size=24)
            path=HERE/f'cache_null_dev_{name}_{seed}.pt'
            if path.exists():zn=torch.load(path,weights_only=True)
            else:
                pp,_=build_fewshot_pair_prompts(demos,random.Random(9000+seed),n_reps=6)
                hp=selected_hiddens(hlm,pp,layers,batch_size=24)
                hz=selected_hiddens(hlm,[zs_prompt(x) for x,y in demos],layers,batch_size=4)
                zn={l:hp[l].reshape(4,6,-1).mean(1)-hz[l] for l in layers}
                torch.save(zn,path)
            zcache[seed]=(original,zn,demos,queries)
        for fraction in [0.,.25,.5,1.]:
            mixed={l:{t:(1-fraction)*raw[l][t]+fraction*null[l][t] for t in ALL_TASKS} for l in layers}
            D=make_dict(mixed,layers,exclude=name);solver=GramSolver(D)
            for seed,(original,zn,demos,queries) in zcache.items():
                Z=original.clone()
                for l in layers:Z[:,l]=(1-fraction)*original[:,l]+fraction*zn[l]
                zl=z_list_from_Z(D,Z);code=solver.solve(zl)
                prompts=[zs_prompt(x) for x,y in queries]
                for gain in [.25,.5,1.,1.5,2.]:
                    for correction in [0.,1.]:
                        ops,lys=correction_ops(D,code,zl,gamma=gain,correction=correction)
                        for schedule in ['all','prefill']:
                            cfg=dict(null_fraction=fraction,gamma=gain,correction=correction,schedule=schedule)
                            label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
                            if ledger.has(name,seed,label):continue
                            preds=generate(hlm,prompts,ops,lys,schedule)
                            ledger.add(name,seed,label,queries,preds,support=code.support,residual=code.residual,
                                       znorm=float(np.linalg.norm(np.mean(zl,axis=0))),demos=demos)
            print(name,fraction,'done',round(time.time()-start,1),flush=True)
    (HERE/'signature_dev_done.json').write_text(json.dumps(dict(elapsed=time.time()-start)))


if __name__=='__main__':main()
