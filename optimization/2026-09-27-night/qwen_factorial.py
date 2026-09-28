"""Held-out-task diagnostic: matched prompt/layer/gain oracle factorial."""
import os
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,load_g,make_dict,generate,Ledger
from chat_probe import render
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt
from cass.pipeline import oracle_ops

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
hlm=HookedLM('qwen3-4b');layers=[14,20,24]
raw=load_g('qwen3-4b',layers);chat={l:{} for l in layers}
for name in ALL_TASKS:
    blob=torch.load(HERE/'qwen_chat_activations'/f'{name}.pt',weights_only=True)
    for l in layers:chat[l][name]=blob['G'][:,l].float().numpy()
dictionaries={(fmt,tuple(lys)):make_dict(g,lys) for fmt,g in [('plain',raw),('chat',chat)]
              for lys in [[14,20],[24]]}
ledger=Ledger('qwen_factorial')
for name in [n for n in ALL_TASKS if n not in DEV_TASKS]:
    task=load_task(name);queries=task.eval_queries
    for (fmt,lys),D in dictionaries.items():
        pp=[zs_prompt(x) for x,y in queries]
        if fmt=='chat':pp=[render(hlm,p) for p in pp]
        for gain in [1.,2.]:
            ops,layers=oracle_ops(D,name,gamma=gain)
            for positions in [1,4]:
                label=f'{fmt}_layers{lys}_gain{gain}_positions{positions}'
                if ledger.has(name,0,label):continue
                preds=generate(hlm,pp,ops,layers,positions=positions)
                acc=ledger.add(name,0,label,queries,preds,scope='known-task oracle, held-out development tasks',
                               prompt=fmt,layers=list(lys),gain=gain,positions=positions)
                print(name,label,round(acc,4),flush=True)
(HERE/'qwen_factorial_done.json').write_text('{"complete":true}\n')
