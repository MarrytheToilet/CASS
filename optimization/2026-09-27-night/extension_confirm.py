"""Frozen improved operators and fully independent dictionary-free baselines."""
import argparse
import json
import os
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,load_g,make_dict,generate,Ledger
from freeze_extensions import freeze
from signature_confirm import extract
from efficient_ops import selected_hiddens,selected_zlist
from variant_ops import OperatorFactory
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound


def dictionary_free_ops(cfg,data,hlm,ex):
    lys=cfg['layers'];gain=cfg['gamma'];ops=[]
    if cfg['mode']=='state_interpolation':
        baseline=selected_hiddens(hlm,[zs_prompt(x) for x,y in ex],lys,batch_size=4)
        for l in lys:
            target=(data['null'][l]+baseline[l]).mean(0).to('cuda')
            def op(h,v=target,g=gain):return h.float()+g*(v-h.float())
            ops.append(op)
    else:
        a=cfg['null_fraction']
        V=np.concatenate([((1-a)*data['raw'][l]+a*data['null'][l]).numpy() for l in lys],axis=1)
        mean=V.mean(0);direction=mean
        if cfg['mode']=='pc1':
            direction=np.linalg.svd(V,full_matrices=False)[2][0]
            if direction@mean<0:direction=-direction
            direction*=np.linalg.norm(mean)
        for j,l in enumerate(lys):
            d=torch.tensor(direction[j*hlm.d:(j+1)*hlm.d],device='cuda')
            def op(h,v=d,g=gain):return h.float()+g*v
            ops.append(op)
    return ops,lys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    selected=freeze();layers=[12,16]
    raw=load_g('llama31-8b',layers);null={l:{} for l in layers};clean={}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:null[l][name]=b['G_by_layer'][l].numpy()
        clean[name]={l:b['clean_mean'][l].numpy() for l in layers}
    a=selected['combined']['config']['null_fraction']
    mixtures={f:{l:{n:(1-f)*raw[l][n]+f*null[l][n] for n in ALL_TASKS} for l in layers} for f in set([0.,a])}
    full={f:make_dict(G,layers) for f,G in mixtures.items()};solvers={f:GramSolver(D) for f,D in full.items()}
    hlm=HookedLM('llama31-8b');stem='extension_fresh' if args.fresh else 'extension_confirm';ledger=Ledger(stem)
    methods=['combined','combined_no_correction','oneshot','oneshot_no_correction','dictionary_free','icl1','icl4','zero']
    if args.fresh:methods+=['original','signature_mixed','null_no_correction']
    for suite,names in [('loto',ALL_TASKS),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            # Additional validation is on target examples excluded from all its own mining.
            queries=(task.dict_pool[-50:] if args.fresh else task.eval_queries)
            if args.fresh:
                forbidden={x for x,y in task.eval_queries+task.fewshot_pool}
                assert not ({x for x,y in queries}&forbidden),(name,'fresh split overlap')
            ds={f:make_dict(G,layers,exclude=name) for f,G in mixtures.items()} if suite=='loto' else full
            ss={f:GramSolver(D) for f,D in ds.items()} if suite=='loto' else solvers
            if args.fresh:
                for f in [.25,1.]:
                    if f not in ds:
                        G={l:{n:(1-f)*raw[l][n]+f*null[l][n] for n in ALL_TASKS} for l in layers}
                        ds[f]=make_dict(G,layers,exclude=name if suite=='loto' else None);ss[f]=GramSolver(ds[f])
            for seed in [20,21,22]:
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                path=HERE/f'cache_signature_confirm_{name}_{seed}.pt'
                if path.exists():data=torch.load(path,weights_only=True)
                else:data=extract(hlm,ex,layers,seed);torch.save(data,path)
                factories={};codes={};zs={}
                for f,D in ds.items():
                    zl=selected_zlist(D,{l:(1-f)*data['raw'][l]+f*data['null'][l] for l in layers})
                    code=ss[f].solve(zl);factories[f]=OperatorFactory(D,code,zl,clean);codes[f]=code;zs[f]=zl
                p0=[zs_prompt(x) for x,y in queries];p1=[icl_prompt(ex[:1],x) for x,y in queries]
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    cfg={};prompts=p0;ops=None;lys=None;schedule='all';strength=0.;f=0.
                    if method.startswith('combined') or method.startswith('oneshot'):
                        base='combined' if method.startswith('combined') else 'oneshot'
                        cfg=selected[base]['config'].copy();f=cfg.get('null_fraction',0.)
                        V=np.asarray(zs[f]);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
                        agreement=float((V@V.T)[np.triu_indices(4,1)].mean())
                        strength={'coverage':max(0.,1-codes[f].residual**2),'uncertainty':float(np.clip(1-agreement,0,1))}.get(cfg['correction'],cfg['correction'])
                        if method.endswith('no_correction'):strength=0.
                        ops,lys=factories[f].ops(cfg['kind'],cfg['gamma'],strength)
                        schedule=cfg['schedule'];prompts=p1 if base=='oneshot' else p0
                    elif method=='dictionary_free':
                        cfg=selected['dictionary_free']['config'];ops,lys=dictionary_free_ops(cfg,data,hlm,ex);schedule=cfg['schedule']
                    elif method in ['original','signature_mixed','null_no_correction']:
                        f={'original':0.,'signature_mixed':.25,'null_no_correction':1.}[method]
                        gamma=1.5 if method=='null_no_correction' else 1.
                        strength=0. if method=='null_no_correction' else 1.
                        schedule='prefill' if method=='null_no_correction' else 'all'
                        ops,lys=factories[f].ops('original',gamma,strength)
                    elif method=='icl1':prompts=p1
                    elif method=='icl4':prompts=[icl_prompt(ex,x) for x,y in queries]
                    preds=generate(hlm,prompts,ops,lys,schedule)
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,development_task=name in DEV_TASKS,
                                   config_parameters=cfg,schedule=schedule,demos=ex,applied_correction=strength,
                                   support=codes[f].support,residual=codes[f].residual,
                                   znorm=float(np.linalg.norm(np.mean(zs[f],axis=0))),
                                   original_znorm=float(np.linalg.norm(np.mean(zs[0.],axis=0))),fresh_queries=args.fresh)
                    print(suite,name,seed,method,round(acc,4),flush=True)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')

if __name__=='__main__':main()
