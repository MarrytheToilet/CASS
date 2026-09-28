"""Four-demo-only calibration applied to the frozen combined geometry."""
import json
import os
import random
import time
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,load_g,make_dict,generate,Ledger
from efficient_ops import selected_hiddens,selected_zlist
from freeze_extensions import freeze
from variant_ops import OperatorFactory
from fast_solver import GramSolver
from adaptive_calibration import teacher_loss,replacement_ops
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt,build_fewshot_pair_prompts
from cass.compound import COMPOUND_REGISTRY,load_compound


def extract(hlm,ex,layers,seed,fraction):
    pp,pc=build_fewshot_pair_prompts(ex,random.Random(9000+seed),n_reps=6);n=len(ex);m=n*6
    prompts=pp+(pc if fraction<1 else [])+([zs_prompt(x) for x,y in ex] if fraction>0 else [])
    H=selected_hiddens(hlm,prompts,layers,batch_size=24);Z={};positive={}
    for l,h in H.items():
        positive[l]=h[:m].reshape(n,6,-1).mean(1)
        raw=(h[:m]-h[m:2*m]).reshape(n,6,-1).mean(1) if fraction<1 else 0.
        null=positive[l]-h[-n:] if fraction>0 else 0.
        Z[l]=(1-fraction)*raw+fraction*null
    return dict(Z=Z,H=positive,n_sequences=len(prompts))


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    frozen=freeze()['combined']['config'];fraction=frozen['null_fraction'];layers=[12,16]
    raw=load_g('llama31-8b',layers);clean={}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:raw[l][name]=(1-fraction)*raw[l][name]+fraction*b['G_by_layer'][l].float().numpy()
        clean[name]={l:b['clean_mean'][l].numpy() for l in layers}
    full=make_dict(raw,layers);fullsolver=GramSolver(full);hlm=HookedLM('llama31-8b')
    grid=[dict(mode=mode,gamma=gamma,schedule=schedule) for mode in ['cass','z_only']
          for gamma in [.5,1.,1.5,2.,3.] for schedule in ['all','prefill']]
    grid += [dict(mode='replace',schedule=s) for s in ['all','prefill']]
    ledger=Ledger('adaptive_combined');selections=(HERE/'adaptive_combined_selection.jsonl').open('a')
    def prepare(data,D,solver):
        zl=selected_zlist(D,data['Z']);code=solver.solve(zl);factory=OperatorFactory(D,code,zl,clean)
        V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
        uncertainty=float(np.clip(1-(V@V.T)[np.triu_indices(len(V),1)].mean(),0,1))
        strength={'uncertainty':uncertainty,'coverage':max(0.,1-code.residual**2)}.get(frozen['correction'],frozen['correction'])
        return factory,strength,replacement_ops(data['H'],layers),code,zl
    for suite,names in [('loto',ALL_TASKS),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            D=make_dict(raw,layers,exclude=name) if suite=='loto' else full;solver=GramSolver(D) if suite=='loto' else fullsolver
            queries=task.eval_queries;pp=[zs_prompt(x) for x,y in queries]
            for seed in [20,21,22]:
                methods=['adaptive_cass','fixed_policy_no_correction','adaptive_z','frozen_cass','frozen_no_correction','icl4']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                start=time.perf_counter();path=HERE/f'cache_adaptive_combined_{name}_{seed}.pt';reused=path.exists()
                if reused:data=torch.load(path,weights_only=True)
                else:
                    data={'full':extract(hlm,ex,layers,seed,fraction),'folds':[]}
                    for j in range(4):
                        subset=[e for i,e in enumerate(ex) if i!=j];assert ex[j][0] not in {x for x,y in subset}
                        data['folds'].append(extract(hlm,subset,layers,seed,fraction))
                    torch.save(data,path)
                losses=np.zeros((len(grid),4))
                for j,(x,y) in enumerate(ex):
                    factory,strength,rep,_,_=prepare(data['folds'][j],D,solver)
                    for ci,cfg in enumerate(grid):
                        ops=rep if cfg['mode']=='replace' else factory.ops(frozen['kind'],cfg['gamma'],strength if cfg['mode']=='cass' else 0.)[0]
                        losses[ci,j]=teacher_loss(hlm,zs_prompt(x),y,ops,layers,cfg['schedule'])
                means=losses.mean(1)
                ci=min([i for i,c in enumerate(grid) if c['mode']!='z_only'],key=lambda i:means[i])
                zi=min([i for i,c in enumerate(grid) if c['mode']!='cass'],key=lambda i:means[i])
                factory,strength,rep,code,zl=prepare(data['full'],D,solver);elapsed=time.perf_counter()-start
                selections.write(json.dumps(dict(task=name,seed=seed,candidates=grid,losses=losses.tolist(),cass_index=ci,z_index=zi,
                    frozen_config=frozen,examples=ex,seconds=elapsed,cache_reused=reused,
                    extraction_sequences=sum(d['n_sequences'] for d in [data['full']]+data['folds']),teacher_forwards=88))+'\n');selections.flush()
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    prompts=pp;schedule=frozen['schedule'];selected=None
                    if method in ['adaptive_cass','fixed_policy_no_correction','adaptive_z']:
                        selected=grid[zi if method=='adaptive_z' else ci];schedule=selected['schedule']
                        ops=rep if selected['mode']=='replace' else factory.ops(frozen['kind'],selected['gamma'],strength if method=='adaptive_cass' else 0.)[0]
                    elif method=='frozen_cass':ops=factory.ops(frozen['kind'],frozen['gamma'],strength)[0]
                    elif method=='frozen_no_correction':ops=factory.ops(frozen['kind'],frozen['gamma'],0.)[0]
                    else:ops=None;prompts=[icl_prompt(ex,x) for x,y in queries]
                    preds=generate(hlm,prompts,ops,layers if ops else None,schedule)
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,development_task=name in DEV_TASKS,
                         selected=selected,common_policy=grid[ci],z_policy=grid[zi],demos=ex,calibration_seconds=elapsed,
                         support=code.support,residual=code.residual,znorm=float(np.linalg.norm(np.mean(zl,axis=0))))
                    print(name,seed,method,round(acc,4),flush=True)
    (HERE/'adaptive_combined_done.json').write_text('{"complete":true}\n')

if __name__=='__main__':main()
