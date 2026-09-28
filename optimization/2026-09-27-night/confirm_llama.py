"""Frozen development winners on original held-out queries, seeds 20/21/22."""
import json
import os
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,load_g,make_dict,generate,Ledger,correction_ops
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound
from cass.extract import extract_fewshot_z
from cass.pipeline import z_list_from_Z,ops_for

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
selected=json.loads((HERE/'llama_selected.json').read_text())
hlm=HookedLM('llama31-8b');layers=[12,16]
G=load_g('llama31-8b',layers);Dfull=make_dict(G,layers);Sfull=GramSolver(Dfull)
ledger=Ledger('llama_confirm')
for suite,names in [('loto',ALL_TASKS),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
    for name in names:
        task=load_compound(name) if suite=='compound' else load_task(name)
        D=make_dict(G,layers,exclude=name) if suite=='loto' else Dfull
        solver=GramSolver(D) if suite=='loto' else Sfull
        for seed in [20,21,22]:
            rng=np.random.default_rng(100*seed+4)
            ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
            queries=task.eval_queries;pp=[zs_prompt(x) for x,y in queries]
            cache=HERE/f'cache_confirm_{name}_{seed}.pt'
            if cache.exists():Z=torch.load(cache,weights_only=True)
            else:
                Z=extract_fewshot_z(hlm,ex,seed=seed,batch_size=24);torch.save(Z,cache)
            zl=z_list_from_Z(D,Z);code=solver.solve(zl);z=np.mean(zl,axis=0)
            H=hlm.last_token_hiddens([icl_prompt([e for i,e in enumerate(ex) if i!=j],x) for j,(x,y) in enumerate(ex)],batch_size=4)
            rep=[]
            for l in layers:
                v=H[:,l].mean(0).to('cuda')
                def op(h,vec=v):return vec.unsqueeze(0).expand_as(h).clone()
                rep.append(op)
            configs={'original':dict(gamma=1.,correction=1.,schedule='all'),
                     'best_correction':selected['best_nonzero_correction'],
                     'best_overall':selected['config']}
            fixed=selected['best_nonzero_correction'].copy();fixed['correction']=0.
            configs['matched_no_correction']=fixed
            methods=list(configs)+['original_z','matched_z','replace','icl4']
            for method in methods:
                if ledger.has(name,seed,method):continue
                schedule='all';prompts=pp
                if method in configs:
                    cfg=configs[method].copy();schedule=cfg.pop('schedule','all')
                    ops,lys=correction_ops(D,code,zl,**cfg)
                elif method in ['original_z','matched_z']:
                    cfg=selected['best_nonzero_correction'] if method=='matched_z' else {}
                    schedule=cfg.get('schedule','all')
                    ops,lys=ops_for(D,code,gamma=cfg.get('gamma',1.),injection='additive',delta_vec=z)
                elif method=='replace':ops,lys=rep,layers
                else:ops,lys=None,None;prompts=[icl_prompt(ex,x) for x,y in queries]
                preds=generate(hlm,prompts,ops,lys,schedule)
                acc=ledger.add(name,seed,method,queries,preds,suite=suite,development_task=name in DEV_TASKS,
                               demos=ex,support=code.support,residual=code.residual,znorm=float(np.linalg.norm(z)))
                print(suite,name,seed,method,round(acc,4),flush=True)
(HERE/'llama_confirm_done.json').write_text('{"complete":true}\n')
