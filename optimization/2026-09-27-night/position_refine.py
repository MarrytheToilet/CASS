"""Development-only multi-position refinement with frozen confirmation."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,dev_split,make_dict,generate,Ledger
from freeze_extensions import freeze as base_selection
from extension_confirm import dictionary_free_ops
from efficient_ops import selected_zlist
from fast_solver import GramSolver
from variant_ops import OperatorFactory
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound

LAYERS=[12,16]


def assets():
    G={l:{} for l in LAYERS};clean={}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in LAYERS:G[l][name]=b['G_by_layer'][l].float().numpy()
        clean[name]={l:b['clean_mean'][l].numpy() for l in LAYERS}
    return G,clean


def grid():
    original=base_selection()['combined']['config'];assert original['null_fraction']==1.
    free=json.loads((HERE/'dictionary_free_refined_selected.json').read_text())['config']
    configs=[]
    for positions in [1,4,8,'all']:
        for schedule in ['all','prefill']:
            for gamma in [1.,1.5,2.]:
                for correction in [0.,'uncertainty']:
                    configs.append(dict(original,pipeline='context',positions=positions,schedule=schedule,gamma=gamma,correction=correction))
            for gamma in [.5,1.,1.5,2.]:
                configs.append(dict(free,pipeline='dictionary_free',positions=positions,schedule=schedule,gamma=gamma))
    assert len(configs)==80
    return configs


def prepare(D,solver,data,clean):
    zl=selected_zlist(D,data['null']);code=solver.solve(zl)
    V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
    strength=float(np.clip(1-(V@V.T)[np.triu_indices(4,1)].mean(),0,1))
    return OperatorFactory(D,code,zl,clean),strength,code


def operators(cfg,prepared,data,hlm,examples,off=False):
    if cfg['pipeline']=='dictionary_free':return dictionary_free_ops(cfg,data,hlm,examples)
    factory,strength,code=prepared
    c=strength if cfg['correction']=='uncertainty' else cfg['correction']
    return factory.ops(cfg['kind'],cfg['gamma'],0. if off else c)


def freeze():
    path=HERE/'position_dev.jsonl';grouped=defaultdict(list)
    for line in path.read_text().splitlines():
        row=json.loads(line);grouped[row['config']].append(row['acc'])
    assert len(grouped)==80 and all(len(v)==24 for v in grouped.values())
    means={k:float(np.mean(v)) for k,v in grouped.items()};selected={}
    context=[k for k in means if json.loads(k)['pipeline']=='context']
    choices={'best_correction':[k for k in context if json.loads(k)['correction']!=0.],
             'best_overall':context,'dictionary_free':[k for k in means if json.loads(k)['pipeline']=='dictionary_free']}
    for method,keys in choices.items():selected[method]=json.loads(max(keys,key=means.get))
    selected.update(development_means={k:means[json.dumps(v,sort_keys=True,separators=(',',':'))] for k,v in selected.items()},
                    all_development_means=means,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dest=HERE/'position_selected.json'
    if dest.exists():assert json.loads(dest.read_text())==selected
    else:dest.write_text(json.dumps(selected,indent=2)+'\n')
    return selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['dev','confirm','fresh']);args=parser.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    G,clean=assets();hlm=HookedLM('llama31-8b')
    if args.stage=='dev':
        configs=grid();ledger=Ledger('position_dev')
        (HERE/'position_grid.json').write_text(json.dumps(configs,indent=2)+'\n')
        for name in DEV_TASKS:
            task=load_task(name);D=make_dict(G,LAYERS,exclude=name);solver=GramSolver(D)
            for seed in [10,11]:
                ex,queries=dev_split(task,seed)
                raw=torch.load(HERE/f'cache_llama_dev_{name}_{seed}.pt',weights_only=True)
                data=dict(raw={l:raw[:,l] for l in LAYERS},null=torch.load(HERE/f'cache_null_dev_{name}_{seed}.pt',weights_only=True))
                prepared=prepare(D,solver,data,clean);pp=[zs_prompt(x) for x,y in queries]
                for cfg in configs:
                    label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
                    if ledger.has(name,seed,label):continue
                    ops,layers=operators(cfg,prepared,data,hlm,ex)
                    preds=generate(hlm,pp,ops,layers,cfg['schedule'],positions=cfg['positions'])
                    ledger.add(name,seed,label,queries,preds,demos=ex)
                print(name,seed,'position development complete',flush=True)
        freeze();(HERE/'position_dev_done.json').write_text('{"complete":true}\n');return
    selected=freeze();fresh=args.stage=='fresh';stem='position_fresh' if fresh else 'position_confirm';ledger=Ledger(stem)
    source='extension_fresh' if fresh else 'extension_confirm'
    contents=(HERE/(source+'.jsonl')).read_bytes();source_hash=hashlib.sha256(contents).hexdigest()
    reference={(r['task'],r['seed'],r['config']):r for r in map(json.loads,contents.decode().splitlines())}
    checked_reuse=set()
    base_cfg=base_selection()['combined']['config']
    frozen=dict(base_cfg,pipeline='context',positions=1)
    full=make_dict(G,LAYERS);fullsolver=GramSolver(full)
    for suite,names in [('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            queries=task.dict_pool[-50:] if fresh else task.eval_queries
            if fresh:assert not ({x for x,y in queries}&{x for x,y in task.eval_queries+task.fewshot_pool})
            D=make_dict(G,LAYERS,exclude=name) if suite=='heldout_known' else full
            solver=GramSolver(D) if suite=='heldout_known' else fullsolver
            for seed in [20,21,22]:
                methods=['best_correction','matched_no_correction','best_overall','dictionary_free','frozen_cass','icl4']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                data=torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True)
                prepared=prepare(D,solver,data,clean);pp=[zs_prompt(x) for x,y in queries]
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    cfg={};ops=None;layers=None;schedule='all';positions=1;prompts=pp
                    if method=='icl4':prompts=[icl_prompt(ex,x) for x,y in queries]
                    else:
                        cfg=frozen if method=='frozen_cass' else selected['best_correction' if method=='matched_no_correction' else method]
                        ops,layers=operators(cfg,prepared,data,hlm,ex,method=='matched_no_correction')
                        schedule=cfg['schedule'];positions=cfg['positions']
                    reuse_method=None
                    if method=='icl4':reuse_method='icl4'
                    elif cfg==frozen:reuse_method='combined_no_correction' if method=='matched_no_correction' else 'combined'
                    if reuse_method is not None:
                        ref=reference[name,seed,reuse_method]
                        assert ref['queries']==[list(q) for q in queries] and ref['demos']==[list(e) for e in ex]
                        if reuse_method!='icl4':assert ref['config_parameters']==base_cfg
                        predictions=ref['predictions']
                        if (suite,reuse_method) not in checked_reuse:
                            assert generate(hlm,prompts,ops,layers,schedule,positions=positions)==predictions
                            checked_reuse.add((suite,reuse_method))
                    else:predictions=generate(hlm,prompts,ops,layers,schedule,positions=positions)
                    value=ledger.add(name,seed,method,queries,predictions,suite=suite,demos=ex,config_parameters=cfg,
                                     fresh_queries=fresh,development_source_sha256=selected['source_sha256'],
                                     reused_source=source if reuse_method else None,reused_method=reuse_method,
                                     reused_source_sha256=source_hash if reuse_method else None,
                                     support=prepared[2].support,residual=prepared[2].residual)
                    print(stem,name,seed,method,round(value,4),flush=True)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


if __name__=='__main__':main()
