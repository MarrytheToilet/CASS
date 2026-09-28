"""Enumerate all six leave-one-out orders at the same28-sequence null cost."""
import argparse
import hashlib
import itertools
import json
import os
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,make_dict,generate,Ledger
from position_refine import assets,prepare,LAYERS
from freeze_extensions import freeze
from efficient_ops import selected_hiddens
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound


def extract(hlm,name,seed,examples):
    path=HERE/f'cache_exhaustive_orders_{name}_{seed}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    assert len(examples)==4;positive=[]
    for j,(x,y) in enumerate(examples):
        others=[e for i,e in enumerate(examples) if i!=j]
        assert x not in {a for a,b in others}
        orders=list(itertools.permutations(others));assert len(orders)==6
        positive.extend(icl_prompt(order,x) for order in orders)
    hp=selected_hiddens(hlm,positive,LAYERS,batch_size=24)
    hz=selected_hiddens(hlm,[zs_prompt(x) for x,y in examples],LAYERS,batch_size=4)
    result=dict(null={l:hp[l].reshape(4,6,-1).mean(1)-hz[l] for l in LAYERS},
                positive={l:hp[l].reshape(4,6,-1).mean(1) for l in LAYERS},
                orders_per_query=6,extraction_sequences=28)
    torch.save(result,path);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['combined']['config'];assert cfg['null_fraction']==1. and cfg['correction']=='uncertainty'
    G,clean=assets();full=make_dict(G,LAYERS);fullsolver=GramSolver(full);hlm=HookedLM('llama31-8b')
    source='extension_fresh' if args.fresh else 'extension_confirm';refpath=HERE/(source+'.jsonl');reference={};source_hash=None
    if refpath.exists():
        contents=refpath.read_bytes();source_hash=hashlib.sha256(contents).hexdigest()
        lines=contents.decode().splitlines()
        for i,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i==len(lines)-1:break
                raise
            if row['config'] in ['combined','combined_no_correction','icl4']:reference[(row['task'],row['seed'],row['config'])]=row
    stem='exhaustive_orders_fresh' if args.fresh else 'exhaustive_orders_confirm';ledger=Ledger(stem)
    for suite,names in [('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            queries=task.dict_pool[-50:] if args.fresh else task.eval_queries
            if args.fresh:assert not ({x for x,y in queries}&{x for x,y in task.eval_queries+task.fewshot_pool})
            D=make_dict(G,LAYERS,exclude=name) if suite=='heldout_known' else full
            solver=GramSolver(D) if suite=='heldout_known' else fullsolver
            for seed in [20,21,22]:
                methods=['exhaustive_cass','exhaustive_off','frozen_cass','frozen_off','icl4']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                new=prepare(D,solver,extract(hlm,name,seed,ex),clean)
                old=prepare(D,solver,torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True),clean)
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    refmethod={'frozen_cass':'combined','frozen_off':'combined_no_correction','icl4':'icl4'}.get(method)
                    reference_row=reference.get((name,seed,refmethod))
                    if reference_row is not None:
                        assert reference_row['queries']==[list(q) for q in queries]
                        assert reference_row['demos']==[list(e) for e in ex]
                        predictions=reference_row['predictions']
                    else:
                        prompts=[zs_prompt(x) for x,y in queries];ops=None;layers=None
                        if method=='icl4':prompts=[icl_prompt(ex,x) for x,y in queries]
                        else:
                            factory,strength,code=new if method.startswith('exhaustive') else old
                            ops,layers=factory.ops(cfg['kind'],cfg['gamma'],0. if method.endswith('_off') else strength)
                        predictions=generate(hlm,prompts,ops,layers,cfg['schedule'])
                    code=(new if method.startswith('exhaustive') else old)[2]
                    value=ledger.add(name,seed,method,queries,predictions,suite=suite,demos=ex,config_parameters=cfg,
                        fresh_queries=args.fresh,support=code.support,residual=code.residual,
                        reused_source=source if reference_row is not None else None,
                        reused_source_sha256=source_hash if reference_row is not None else None,
                        order_protocol='all six distinct permutations' if method.startswith('exhaustive') else 'original resampled orders',
                        extraction_sequences=28 if method.startswith('exhaustive') else None)
                    print(stem,name,seed,method,round(value,4),flush=True)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


if __name__=='__main__':main()
