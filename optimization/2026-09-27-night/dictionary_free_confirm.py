"""Confirm a denser baseline-only development search; no evaluation selection."""
import hashlib
import json
import os
from collections import defaultdict
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,generate,Ledger
from extension_confirm import dictionary_free_ops
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound


def freeze():
    grouped=defaultdict(list);hashes={}
    for stem,expected in [('dictionary_free_dev',360),('dictionary_free_refine',192)]:
        path=HERE/(stem+'.jsonl');seen=set()
        for line in path.read_text().splitlines():
            r=json.loads(line);grouped[r['config']].append(r['acc']);seen.add(r['config'])
        assert len(seen)==expected;hashes[stem]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(grouped)==552 and all(len(v)==24 for v in grouped.values())
    means={k:float(np.mean(v)) for k,v in grouped.items()};best=max(means,key=means.get)
    result=dict(config=json.loads(best),development_accuracy=means[best],all_development_means=means,source_hashes=hashes)
    dest=HERE/'dictionary_free_refined_selected.json'
    if dest.exists():assert json.loads(dest.read_text())==result
    else:dest.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['config'];hlm=HookedLM('llama31-8b')
    ledgers={split:Ledger('dictionary_free_refined_'+split) for split in ['confirm','fresh']}
    for suite,names in [('loto',ALL_TASKS),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            for seed in [20,21,22]:
                method='dictionary_free_refined'
                if all(l.has(name,seed,method) for l in ledgers.values()):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                data=torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True)
                ops,lys=dictionary_free_ops(cfg,data,hlm,ex)
                for split,ledger in ledgers.items():
                    if ledger.has(name,seed,method):continue
                    queries=task.eval_queries if split=='confirm' else task.dict_pool[-50:]
                    pp=[zs_prompt(x) for x,y in queries];preds=generate(hlm,pp,ops,lys,cfg['schedule'])
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,development_task=name in DEV_TASKS,config_parameters=cfg,demos=ex)
                    print(split,name,seed,round(acc,4),flush=True)
    (HERE/'dictionary_free_refined_done.json').write_text('{"complete":true}\n')

if __name__=='__main__':main()
