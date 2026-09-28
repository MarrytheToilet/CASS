"""Shared-rank mechanism controls with every other selected setting frozen."""
import argparse
import hashlib
import json
import os
import numpy as np
import torch
from common import HERE, ALL_TASKS, DEV_TASKS, generate, Ledger
from position_refine import assets, prepare, LAYERS
from freeze_extensions import freeze
from fast_solver import GramSolver
from cass.dictionary import build_multilayer_dictionary
from cass.models import HookedLM
from cass.tasks import load_task, synthetic_tasks, zs_prompt
from cass.compound import COMPOUND_REGISTRY, load_compound

RANKS=[0,1,2,4]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['combined']['config']
    assert cfg==dict(null_fraction=1.,kind='context_residual',gamma=1.5,
                     correction='uncertainty',schedule='prefill')
    G,clean=assets();hlm=HookedLM('llama31-8b')
    def dictionaries(exclude=None):
        raw={l:{n:a for n,a in G[l].items() if n!=exclude} for l in LAYERS}
        ds={rank:build_multilayer_dictionary(raw,r0=rank) for rank in RANKS}
        return ds,{rank:GramSolver(d) for rank,d in ds.items()}
    full,fullsolvers=dictionaries()
    source='extension_fresh' if args.fresh else 'extension_confirm'
    path=HERE/(source+'.jsonl');contents=path.read_bytes();digest=hashlib.sha256(contents).hexdigest()
    reference={(r['task'],r['seed'],r['config']):r for r in map(json.loads,contents.decode().splitlines())}
    stem='shared_rank_fresh' if args.fresh else 'shared_rank_confirm';ledger=Ledger(stem)
    checked_rank1=False
    for suite,names in [('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),
                        ('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            queries=task.dict_pool[-50:] if args.fresh else task.eval_queries
            if args.fresh:assert not ({x for x,y in queries}&{x for x,y in task.eval_queries+task.fewshot_pool})
            ds,solvers=dictionaries(name) if suite=='heldout_known' else (full,fullsolvers)
            for seed in [20,21,22]:
                base=reference[name,seed,'combined'];off=reference[name,seed,'combined_no_correction'];icl=reference[name,seed,'icl4']
                assert base['queries']==off['queries']==icl['queries']==[list(q) for q in queries]
                assert base['demos']==off['demos']==icl['demos'] and base['config_parameters']==cfg
                ex=base['demos'];data=torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True)
                prompts=[zs_prompt(x) for x,y in queries]
                for rank in RANKS:
                    factory,strength,code=prepare(ds[rank],solvers[rank],data,clean)
                    for disable in [False,True]:
                        method=f'rank{rank}'+('_off' if disable else '')
                        if ledger.has(name,seed,method):continue
                        ops,layers=factory.ops(cfg['kind'],cfg['gamma'],0. if disable else strength)
                        reused=rank==1
                        if reused:
                            ref=off if disable else base;preds=ref['predictions']
                            assert code.support==base['support']
                            if not checked_rank1:
                                assert generate(hlm,prompts,ops,layers,cfg['schedule'])==preds
                                if disable:checked_rank1=True
                        else:preds=generate(hlm,prompts,ops,layers,cfg['schedule'])
                        value=ledger.add(name,seed,method,queries,preds,suite=suite,demos=ex,
                            config_parameters=cfg,shared_rank=rank,correction_disabled=disable,
                            support=code.support,residual=code.residual,
                            fresh_queries=args.fresh,reused_source=source if reused else None,
                            reused_source_sha256=digest if reused else None)
                        print(stem,name,seed,method,round(value,4),flush=True)
                if not ledger.has(name,seed,'icl4'):
                    ledger.add(name,seed,'icl4',queries,icl['predictions'],suite=suite,demos=ex,
                        fresh_queries=args.fresh,reused_source=source,reused_source_sha256=digest)
    (HERE/(stem+'_done.json')).write_text('{"complete":true}\n')


if __name__=='__main__':main()
