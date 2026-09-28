"""Model-specific native-prompt contrast development and frozen task transfer."""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,dev_split,make_dict,generate,Ledger,load_g
from chat_probe import render
from efficient_ops import selected_hiddens,selected_zlist
from fast_solver import GramSolver
from variant_ops import OperatorFactory
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt,build_pair_prompts,build_fewshot_pair_prompts
from cass.compound import COMPOUND_REGISTRY,load_compound
from cass.pipeline import ops_for

LAYERS=[20,24,28]


def mine(hlm,model):
    folder=HERE/(model+'_contrast_activations');folder.mkdir(exist_ok=True)
    raw={l:{} for l in LAYERS};null={l:{} for l in LAYERS};clean={}
    for name in ALL_TASKS:
        path=folder/(name+'.pt')
        if path.exists():b=torch.load(path,weights_only=True)
        else:
            task=load_task(name);pp,pc=build_pair_prompts(task.dict_pool,100,10,random.Random(7000))
            hp=selected_hiddens(hlm,[render(hlm,p) for p in pp+pc],LAYERS,batch_size=24)
            hz=selected_hiddens(hlm,[render(hlm,zs_prompt(task.dict_pool[i%len(task.dict_pool)][0])) for i in range(100)],LAYERS,batch_size=25)
            b=dict(raw={l:(hp[l][:100]-hp[l][100:]).half() for l in LAYERS},
                   null={l:(hp[l][:100]-hz[l]).half() for l in LAYERS},
                   clean={l:hp[l][:100].mean(0) for l in LAYERS})
            torch.save(b,path)
        for l in LAYERS:raw[l][name]=b['raw'][l].float().numpy();null[l][name]=b['null'][l].float().numpy()
        clean[name]={l:b['clean'][l].numpy() for l in LAYERS}
        print('mined',model,name,flush=True)
    return raw,null,clean


def signature(hlm,model,name,seed,ex):
    path=HERE/f'cache_{model}_contrast_{name}_{seed}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    pp,pc=build_fewshot_pair_prompts(ex,random.Random(9000+seed),n_reps=6)
    hp=selected_hiddens(hlm,[render(hlm,p) for p in pp+pc],LAYERS,batch_size=24)
    hz=selected_hiddens(hlm,[render(hlm,zs_prompt(x)) for x,y in ex],LAYERS,batch_size=4)
    b=dict(raw={l:(h[:24]-h[24:]).reshape(4,6,-1).mean(1) for l,h in hp.items()},
           null={l:h[:24].reshape(4,6,-1).mean(1)-hz[l] for l,h in hp.items()},
           positive={l:h[:24].reshape(4,6,-1).mean(1) for l,h in hp.items()})
    torch.save(b,path);return b


def dictionaries(raw,null,exclude=None,settings=None):
    settings=settings or [(f,lys) for f in [0.,.25,1.] for lys in [[20],[24],[28],[20,24],[24,28]]]
    ds={};ss={}
    for f,lys in settings:
        key=(f,tuple(lys))
        if key in ds:continue
        G={l:{n:(1-f)*raw[l][n]+f*null[l][n] for n in ALL_TASKS} for l in lys}
        ds[key]=make_dict(G,lys,exclude);ss[key]=GramSolver(ds[key])
    return ds,ss


def setup(ds,ss,data,clean):
    out={}
    for (f,lys),D in ds.items():
        zl=selected_zlist(D,{l:(1-f)*data['raw'][l]+f*data['null'][l] for l in lys});code=ss[(f,lys)].solve(zl)
        V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
        uncertainty=float(np.clip(1-(V@V.T)[np.triu_indices(4,1)].mean(),0,1))
        out[(f,lys)]=(OperatorFactory(D,code,zl,clean),code,zl,uncertainty)
    return out


def get_ops(cfg,pre,off=False):
    factory,code,zl,uncertainty=pre[(cfg['null_fraction'],tuple(cfg['layers']))]
    strength=uncertainty if cfg['correction']=='uncertainty' else cfg['correction']
    if off:strength=0.
    ops,lys=factory.ops(cfg['kind'],cfg['gamma'],strength)
    return ops,lys,code,zl,strength


def development(hlm,model,raw,null,clean):
    grid=[]
    for f in [0.,.25,1.]:
        for lys in [[20],[24],[28],[20,24],[24,28]]:
            for gamma in [.5,1.,2.]:
                for correction in [0.,1.]:
                    for schedule in ['all','prefill']:
                        grid.append(dict(null_fraction=f,layers=lys,gamma=gamma,correction=correction,schedule=schedule,kind='original'))
    for lys in [[20],[24],[28],[20,24],[24,28]]:
        for gamma in [1.,2.]:
            for schedule in ['all','prefill']:
                grid.append(dict(null_fraction=1.,layers=lys,gamma=gamma,correction='uncertainty',schedule=schedule,kind='context_residual'))
    (HERE/(model+'_contrast_grid.json')).write_text(json.dumps(grid,indent=2)+'\n')
    ledger=Ledger(model+'_contrast_dev')
    for name in DEV_TASKS:
        task=load_task(name);ds,ss=dictionaries(raw,null,exclude=name)
        for seed in [10,11]:
            ex,queries=dev_split(task,seed);data=signature(hlm,model,name,seed,ex);pre=setup(ds,ss,data,clean)
            pp=[render(hlm,zs_prompt(x)) for x,y in queries]
            for mode in ['zero','icl4']:
                if ledger.has(name,seed,mode):continue
                prompts=pp if mode=='zero' else [render(hlm,icl_prompt(ex,x)) for x,y in queries]
                ledger.add(name,seed,mode,queries,generate(hlm,prompts),demos=ex)
            for cfg in grid:
                label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
                if ledger.has(name,seed,label):continue
                ops,lys,code,zl,strength=get_ops(cfg,pre)
                preds=generate(hlm,pp,ops,lys,cfg['schedule'])
                ledger.add(name,seed,label,queries,preds,demos=ex,support=code.support,residual=code.residual,
                           znorm=float(np.linalg.norm(np.mean(zl,axis=0))),applied_correction=strength)
            print(model,name,seed,'development complete',flush=True)
    (HERE/(model+'_contrast_dev_done.json')).write_text('{"complete":true}\n')


def freeze(model):
    path=HERE/(model+'_contrast_dev.jsonl');grouped=defaultdict(list)
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['config'].startswith('{'):grouped[r['config']].append(r['acc'])
    assert len(grouped)==200 and all(len(v)==24 for v in grouped.values())
    means={k:float(np.mean(v)) for k,v in grouped.items()}
    best=max(means,key=means.get);nonzero=max([k for k in means if json.loads(k)['correction']!=0],key=means.get)
    record=dict(best_overall=json.loads(best),best_correction=json.loads(nonzero),
                development_overall=means[best],development_correction=means[nonzero],all_development_means=means,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dest=HERE/(model+'_contrast_selected.json')
    if dest.exists():assert json.loads(dest.read_text())==record
    else:dest.write_text(json.dumps(record,indent=2)+'\n')
    return record


def confirmation(hlm,model,raw,null,clean,fresh=False):
    selected=freeze(model)
    settings=[(c['null_fraction'],c['layers']) for c in [selected['best_overall'],selected['best_correction']]]
    settings.append((0.,[24]));full,solvers=dictionaries(raw,null,settings=settings)
    # Original plain-prompt dictionary, separately mined for Qwen2.5 if needed.
    if model=='qwen3-4b':plain=load_g(model,[14,20])
    else:
        plain={l:{} for l in [14,20]};folder=HERE/(model+'_plain_activations');folder.mkdir(exist_ok=True)
        for name in ALL_TASKS:
            p=folder/(name+'.pt')
            if p.exists():b=torch.load(p,weights_only=True)
            else:
                pp,pc=build_pair_prompts(load_task(name).dict_pool,100,10,random.Random(7000))
                H=selected_hiddens(hlm,pp+pc,[14,20],batch_size=24)
                b={l:(H[l][:100]-H[l][100:]).half() for l in [14,20]};torch.save(b,p)
            for l in [14,20]:plain[l][name]=b[l].float().numpy()
    stem=model+('_contrast_fresh' if fresh else '_contrast_confirm')
    pd=make_dict(plain,[14,20]);ps=GramSolver(pd);ledger=Ledger(stem)
    for suite,names in [('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            ds,ss=dictionaries(raw,null,exclude=name,settings=settings) if suite=='heldout_known' else (full,solvers)
            dp=make_dict(plain,[14,20],exclude=name) if suite=='heldout_known' else pd
            sp=GramSolver(dp) if suite=='heldout_known' else ps
            queries=task.dict_pool[-50:] if fresh else task.eval_queries
            if fresh:assert not ({x for x,y in queries}&{x for x,y in task.eval_queries+task.fewshot_pool})
            p0=[zs_prompt(x) for x,y in queries];pp=[render(hlm,p) for p in p0]
            for seed in [20,21,22]:
                methods=['best_correction','best_overall','matched_no_correction','native_shuffle24','plain_original','icl4','zero']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                data=signature(hlm,model,name,seed,ex);pre=setup(ds,ss,data,clean)
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    prompts=pp;ops=None;lys=None;cfg={};code=None;strength=0.;schedule='all';znorm=None
                    if method in ['best_correction','best_overall','matched_no_correction','native_shuffle24']:
                        cfg=(dict(null_fraction=0.,layers=[24],kind='original',gamma=2.,correction=1.,schedule='all') if method=='native_shuffle24' else selected['best_overall' if method=='best_overall' else 'best_correction'])
                        ops,lys,code,zl,strength=get_ops(cfg,pre,method=='matched_no_correction');schedule=cfg['schedule'];znorm=float(np.linalg.norm(np.mean(zl,axis=0)))
                    elif method=='plain_original':
                        cfg=dict(null_fraction=0.,layers=[14,20],kind='original',gamma=1.,correction=1.,schedule='all',prompt='plain')
                        strength=1.
                        cp,cn=build_fewshot_pair_prompts(ex,random.Random(9000+seed),n_reps=6)
                        H=selected_hiddens(hlm,cp+cn,[14,20],batch_size=24)
                        Z={l:(h[:24]-h[24:]).reshape(4,6,-1).mean(1) for l,h in H.items()}
                        zl=selected_zlist(dp,Z);code=sp.solve(zl);ops,lys=ops_for(dp,code,delta_vec=np.mean(zl,axis=0));prompts=p0
                    elif method=='icl4':prompts=[render(hlm,icl_prompt(ex,x)) for x,y in queries]
                    preds=generate(hlm,prompts,ops,lys,schedule)
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,config_parameters=cfg,demos=ex,
                                   support=code.support if code else [],residual=code.residual if code else None,
                                   applied_correction=strength,znorm=znorm,
                                   dictionary_used=method not in ['icl4','zero'],
                                   fresh_queries=fresh,
                                   development_source_sha256=selected['source_sha256'])
                    print(model,suite,name,seed,method,round(acc,4),flush=True)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['qwen3-4b','qwen25-3b']);ap.add_argument('stage',choices=['dev','confirm','fresh']);args=ap.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    if args.model=='qwen25-3b':assert (HERE/'qwen25_verified.json').exists()
    hlm=HookedLM(args.model);raw,null,clean=mine(hlm,args.model)
    if args.stage=='dev':development(hlm,args.model,raw,null,clean)
    else:confirmation(hlm,args.model,raw,null,clean,fresh=args.stage=='fresh')

if __name__=='__main__':main()
