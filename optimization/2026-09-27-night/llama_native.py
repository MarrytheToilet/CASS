"""Native-template Llama follow-up with separately tuned dictionary-free control."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
import numpy as np
import torch
from common import HERE, ALL_TASKS, DEV_TASKS, dev_split, generate, Ledger
import qwen_contrast as shared
from chat_probe import render
from cass.models import HookedLM
from cass.tasks import load_task, synthetic_tasks, zs_prompt, icl_prompt
from cass.compound import COMPOUND_REGISTRY, load_compound

MODEL='llama31-8b'
PAIRS=[[12,16],[16,20],[20,24]]
shared.LAYERS=[12,16,20,24]


def grid():
    out=[]
    for a in [0.,.25,1.]:
        for layers in PAIRS:
            for gamma in [.5,1.,1.5,2.]:
                for schedule in ['all','prefill']:
                    for correction in [0.,'uncertainty']:
                        out.append(dict(mode='context',null_fraction=a,layers=layers,gamma=gamma,
                                        correction=correction,kind='context_residual',schedule=schedule))
                    for mode in ['mean','pc1']:
                        out.append(dict(mode=mode,null_fraction=a,layers=layers,gamma=gamma,schedule=schedule))
    assert len(out)==288
    return out


def ops(cfg,pre,data,off=False):
    if cfg['mode']=='context':
        return shared.get_ops(cfg,pre,off)[:2]
    layers=cfg['layers'];a=cfg['null_fraction'];gamma=cfg['gamma']
    V=np.concatenate([((1-a)*data['raw'][l]+a*data['null'][l]).numpy() for l in layers],axis=1)
    mean=V.mean(0);direction=mean
    if cfg['mode']=='pc1':
        direction=np.linalg.svd(V,full_matrices=False)[2][0]
        if direction@mean<0:direction=-direction
        direction*=np.linalg.norm(mean)
    d=V.shape[1]//len(layers);out=[]
    for i,l in enumerate(layers):
        delta=torch.tensor(direction[i*d:(i+1)*d],device='cuda')
        def op(h,v=delta,g=gamma):return h.float()+g*v
        out.append(op)
    return out,layers


def development(hlm,raw,null,clean):
    configs=grid();ledger=Ledger('llama_native_dev')
    (HERE/'llama_native_grid.json').write_text(json.dumps(configs,indent=2)+'\n')
    settings=[(a,layers) for a in [0.,.25,1.] for layers in PAIRS]
    for name in DEV_TASKS:
        task=load_task(name);ds,ss=shared.dictionaries(raw,null,exclude=name,settings=settings)
        for seed in [10,11]:
            ex,queries=dev_split(task,seed)
            data=shared.signature(hlm,MODEL,name,seed,ex);pre=shared.setup(ds,ss,data,clean)
            pp=[render(hlm,zs_prompt(x)) for x,y in queries]
            for cfg in configs:
                label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
                if ledger.has(name,seed,label):continue
                oo,ll=ops(cfg,pre,data);pred=generate(hlm,pp,oo,ll,cfg['schedule'])
                ledger.add(name,seed,label,queries,pred,demos=ex,dictionary_used=cfg['mode']=='context')
            for method in ['icl4','zero']:
                if ledger.has(name,seed,method):continue
                prompts=pp if method=='zero' else [render(hlm,icl_prompt(ex,x)) for x,y in queries]
                ledger.add(name,seed,method,queries,generate(hlm,prompts),demos=ex)
            print(name,seed,'native development complete',flush=True)
    (HERE/'llama_native_dev_done.json').write_text('{"complete":true}\n')


def freeze():
    path=HERE/'llama_native_dev.jsonl';grouped=defaultdict(list)
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['config'].startswith('{'):grouped[r['config']].append(r['acc'])
    assert len(grouped)==288 and all(len(v)==24 for v in grouped.values())
    means={k:float(np.mean(v)) for k,v in grouped.items()}
    families={'best_correction':[k for k in means if json.loads(k)['mode']=='context' and json.loads(k)['correction']!=0],
              'best_overall':[k for k in means if json.loads(k)['mode']=='context'],
              'dictionary_free':[k for k in means if json.loads(k)['mode']!='context']}
    record={};record['development_means']={}
    for label,keys in families.items():
        selected=max(keys,key=means.get);record[label]=json.loads(selected);record['development_means'][label]=means[selected]
    record.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_development_means=means)
    dest=HERE/'llama_native_selected.json'
    if dest.exists():assert json.loads(dest.read_text())==record
    else:dest.write_text(json.dumps(record,indent=2)+'\n')
    return record


def confirmation(hlm,raw,null,clean,fresh=False):
    selected=freeze()
    original=dict(mode='context',null_fraction=0.,layers=[12,16],gamma=1.,correction=1.,kind='original',schedule='all')
    settings=[(c['null_fraction'],c['layers']) for c in [selected['best_correction'],selected['best_overall'],original]]
    full,solvers=shared.dictionaries(raw,null,settings=settings)
    stem='llama_native_fresh' if fresh else 'llama_native_confirm';ledger=Ledger(stem)
    methods=['best_correction','matched_no_correction','best_overall','dictionary_free','native_original','icl1','icl4','zero']
    for suite,names in [('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),
                        ('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            queries=task.dict_pool[-50:] if fresh else task.eval_queries
            if fresh:assert not ({x for x,y in queries}&{x for x,y in task.eval_queries+task.fewshot_pool})
            ds,ss=shared.dictionaries(raw,null,exclude=name,settings=settings) if suite=='heldout_known' else (full,solvers)
            pp=[render(hlm,zs_prompt(x)) for x,y in queries]
            for seed in [20,21,22]:
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                data=shared.signature(hlm,MODEL,name,seed,ex);pre=shared.setup(ds,ss,data,clean)
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    oo=None;ll=None;cfg={};schedule='all';prompts=pp;code=None
                    if method in ['best_correction','best_overall','matched_no_correction','dictionary_free','native_original']:
                        cfg=original if method=='native_original' else selected['best_correction' if method=='matched_no_correction' else method]
                        oo,ll=ops(cfg,pre,data,method=='matched_no_correction');schedule=cfg['schedule']
                        if cfg['mode']=='context':code=pre[(cfg['null_fraction'],tuple(cfg['layers']))][1]
                    elif method.startswith('icl'):
                        n=1 if method=='icl1' else 4
                        prompts=[render(hlm,icl_prompt(ex[:n],x)) for x,y in queries]
                    predictions=generate(hlm,prompts,oo,ll,schedule)
                    value=ledger.add(name,seed,method,queries,predictions,suite=suite,demos=ex,config_parameters=cfg,
                                     fresh_queries=fresh,support=code.support if code else [],residual=code.residual if code else None,
                                     mean_input_tokens=float(np.mean([len(hlm.tok.encode(p)) for p in prompts])))
                    print(stem,suite,name,seed,method,round(value,4),flush=True)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','confirm','fresh']);args=ap.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    hlm=HookedLM(MODEL);raw,null,clean=shared.mine(hlm,MODEL)
    if args.stage=='dev':development(hlm,raw,null,clean)
    else:confirmation(hlm,raw,null,clean,args.stage=='fresh')
