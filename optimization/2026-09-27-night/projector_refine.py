"""Coefficient-weighted corrective geometry, selected only on development tasks."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
import numpy as np
import torch
import position_refine as position
from common import HERE, ALL_TASKS, DEV_TASKS, dev_split, make_dict, generate, Ledger
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task, synthetic_tasks, zs_prompt, icl_prompt
from cass.compound import COMPOUND_REGISTRY, load_compound

LAYERS=position.LAYERS
assets=position.assets


class WeightedFactory:
    def __init__(self, base, device='cuda'):
        self.base=base
        self.device=device
        self.cache={}

    def geometry(self, mode):
        if mode in self.cache:return self.cache[mode]
        base=self.base;D=base.D;support=base.code.support;weights=base.w
        rows=[]
        for layer in D.layers:
            d=D.split(base.delta)[layer]
            if support:
                B=np.concatenate([np.sqrt(w)*D.per_layer[layer].bases[n]
                                  for w,n in zip(weights,support)],axis=1)
                if mode.startswith('rank'):
                    Q=np.linalg.svd(B,full_matrices=False)[0][:,:int(mode[4:])]
                    mu=sum(w*base.clean[n][layer] for w,n in zip(weights,support))
                    b=Q@(Q.T@mu);B=Q
                else:
                    assert mode=='weighted'
                    b=sum(w*(D.per_layer[layer].bases[n]@(D.per_layer[layer].bases[n].T@base.clean[n][layer]))
                          for w,n in zip(weights,support))
            else:B=np.zeros((D.d,0));b=np.zeros(D.d)
            rows.append(tuple(torch.as_tensor(v,device=self.device,dtype=torch.float32) for v in [B,d,b]))
        self.cache[mode]=rows
        return rows

    def ops(self, mode, gamma, strength):
        if mode=='union':return self.base.ops('context_residual',gamma,strength)
        c=strength*self.base.gate;operators=[]
        for B,d,b in self.geometry(mode):
            def operation(h,basis=B,direction=d,field=b):
                shifted=h.float()+gamma*direction
                return shifted+c*(field-(shifted@basis)@basis.T)
            operators.append(operation)
        return operators,list(self.base.D.layers)


def prepare(D,solver,data,clean):
    factory,strength,code=position.prepare(D,solver,data,clean)
    return WeightedFactory(factory),strength,code


def operators(cfg,prepared,data,hlm,examples,off=False):
    factory,uncertainty,code=prepared
    if cfg['pipeline']!='projector':
        return position.operators(cfg,(factory.base,uncertainty,code),data,hlm,examples,off)
    strength=0. if off else float(np.clip(cfg['correction_multiplier']*uncertainty,0.,1.))
    return factory.ops(cfg['projection'],cfg['gamma'],strength)


def grid():
    prior=position.freeze()['best_correction'];assert prior['null_fraction']==1.
    configs=[]
    for projection in ['union','rank8','rank16','rank32','weighted']:
        for gamma in [1.,1.5,2.]:
            for schedule in ['all','prefill']:
                for multiplier in [0.,.5,1.,1.5]:
                    configs.append(dict(prior,pipeline='projector',projection=projection,gamma=gamma,
                                        schedule=schedule,correction_multiplier=multiplier))
    assert len(configs)==120
    return configs


def freeze():
    path=HERE/'projector_dev.jsonl';grouped=defaultdict(list)
    for line in path.read_text().splitlines():
        row=json.loads(line);grouped[row['config']].append(row['acc'])
    assert len(grouped)==120 and all(len(v)==24 for v in grouped.values())
    means={k:float(np.mean(v)) for k,v in grouped.items()}
    keys=list(means);nonzero=[k for k in keys if json.loads(k)['correction_multiplier']>0.]
    prior=position.freeze()
    selected=dict(best_correction=json.loads(max(nonzero,key=means.get)),
                  best_overall=json.loads(max(keys,key=means.get)),dictionary_free=prior['dictionary_free'],
                  position_cass=prior['best_correction'],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  position_development_source_sha256=prior['source_sha256'],all_development_means=means)
    selected['development_means']={k:means[json.dumps(selected[k],sort_keys=True,separators=(',',':'))]
                                   for k in ['best_correction','best_overall']}
    dest=HERE/'projector_selected.json'
    if dest.exists():assert json.loads(dest.read_text())==selected
    else:dest.write_text(json.dumps(selected,indent=2)+'\n')
    return selected


def references(stems):
    result={}
    for stem in stems:
        for line in (HERE/(stem+'.jsonl')).read_text().splitlines():
            row=json.loads(line);result[stem,row['task'],row['seed'],row['config']]=row
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','confirm','fresh']);args=ap.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    G,clean=assets();hlm=HookedLM('llama31-8b')
    if args.stage=='dev':
        configs=grid();ledger=Ledger('projector_dev')
        (HERE/'projector_grid.json').write_text(json.dumps(configs,indent=2)+'\n')
        for name in DEV_TASKS:
            task=load_task(name);D=make_dict(G,LAYERS,exclude=name);solver=GramSolver(D)
            for seed in [10,11]:
                ex,queries=dev_split(task,seed)
                raw=torch.load(HERE/f'cache_llama_dev_{name}_{seed}.pt',weights_only=True)
                data=dict(raw={l:raw[:,l] for l in LAYERS},
                          null=torch.load(HERE/f'cache_null_dev_{name}_{seed}.pt',weights_only=True))
                prepared=prepare(D,solver,data,clean);prompts=[zs_prompt(x) for x,y in queries]
                for cfg in configs:
                    label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
                    if ledger.has(name,seed,label):continue
                    ops,layers=operators(cfg,prepared,data,hlm,ex)
                    predictions=generate(hlm,prompts,ops,layers,cfg['schedule'],positions=cfg['positions'])
                    ledger.add(name,seed,label,queries,predictions,demos=ex)
                print(name,seed,'projector development complete',flush=True)
        freeze();(HERE/'projector_dev_done.json').write_text('{"complete":true}\n');return
    selected=freeze();fresh=args.stage=='fresh';split='fresh' if fresh else 'confirm'
    stem='projector_'+split;ledger=Ledger(stem);pos_stem='position_'+split;ext_stem='extension_'+split
    saved=references([pos_stem,ext_stem])
    baseline={'dictionary_free':(pos_stem,'dictionary_free'),'position_cass':(pos_stem,'best_correction'),
              'frozen_cass':(ext_stem,'combined'),'icl4':(pos_stem,'icl4')}
    full=make_dict(G,LAYERS);fullsolver=GramSolver(full)
    for suite,names in [('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),
                        ('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            queries=task.dict_pool[-50:] if fresh else task.eval_queries
            if fresh:assert not ({x for x,y in queries}&{x for x,y in task.eval_queries+task.fewshot_pool})
            D=make_dict(G,LAYERS,exclude=name) if suite=='heldout_known' else full
            solver=GramSolver(D) if suite=='heldout_known' else fullsolver
            for seed in [20,21,22]:
                methods=['best_correction','matched_no_correction','best_overall',*baseline]
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4)
                ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                data=torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True)
                prepared=prepare(D,solver,data,clean);prompts=[zs_prompt(x) for x,y in queries]
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    reused=None
                    if method in baseline:
                        source,source_method=baseline[method];row=saved[source,name,seed,source_method]
                        assert row['queries']==[list(q) for q in queries] and row['demos']==[list(e) for e in ex]
                        predictions=row['predictions'];cfg=row.get('config_parameters',{});reused=[source,source_method]
                    else:
                        cfg=selected['best_correction' if method=='matched_no_correction' else method]
                        ops,layers=operators(cfg,prepared,data,hlm,ex,method=='matched_no_correction')
                        predictions=generate(hlm,prompts,ops,layers,cfg['schedule'],positions=cfg['positions'])
                    value=ledger.add(name,seed,method,queries,predictions,suite=suite,demos=ex,
                                     config_parameters=cfg,fresh_queries=fresh,reused_baseline=reused,
                                     dictionary_used=method not in ['dictionary_free','icl4'],
                                     development_source_sha256=selected['source_sha256'],
                                     support=prepared[2].support,residual=prepared[2].residual)
                    print(stem,name,seed,method,round(value,4),flush=True)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


if __name__=='__main__':main()
