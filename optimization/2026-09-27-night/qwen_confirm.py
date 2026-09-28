"""Freeze native-chat oracle tuning; evaluate task transfer and unseen CASS."""
import json
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
from common import HERE,ROOT,ALL_TASKS,DEV_TASKS,load_g,make_dict,generate,Ledger
from chat_probe import render
from efficient_ops import selected_hiddens,selected_zlist
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt,build_fewshot_pair_prompts
from cass.compound import COMPOUND_REGISTRY,load_compound
from cass.pipeline import ops_for,oracle_ops


def freeze():
    groups=defaultdict(list)
    for line in (HERE/'qwen_chat_oracle_dev.jsonl').read_text().splitlines():
        row=json.loads(line);groups[row['config']].append(row['acc'])
    assert all(len(v)==12 for v in groups.values())
    means={k:float(np.mean(v)) for k,v in groups.items()}
    eligible=[k for k in means if k.startswith('{')]
    best=max(eligible,key=lambda k:means[k])
    record=dict(config=json.loads(best),validation_mean=means[best],all_validation_means=means,
                note='Selected on known-task oracle development queries; unseen CASS and task-transfer results are separate evaluations.')
    (HERE/'qwen_chat_selected.json').write_text(json.dumps(record,indent=2)+'\n')
    return record['config']


def get_extraction(hlm,examples,layers,seed,chat):
    clean,corr=build_fewshot_pair_prompts(examples,random.Random(9000+seed),n_reps=6)
    prompts=clean+corr
    if chat:prompts=[render(hlm,p) for p in prompts]
    H=selected_hiddens(hlm,prompts,layers,batch_size=24)
    Z={l:(h[:24]-h[24:]).reshape(4,6,-1).mean(1) for l,h in H.items()}
    return Z,{l:h[:24].reshape(4,6,-1).mean(1) for l,h in H.items()}


def main():
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze();chat_layers=cfg['layers'];plain_layers=[24]
    all_layers=sorted(set(chat_layers+[14,20,24]))
    hlm=HookedLM('qwen3-4b')
    raw=load_g('qwen3-4b',all_layers);chat={l:{} for l in chat_layers}
    for name in ALL_TASKS:
        b=torch.load(HERE/'qwen_chat_activations'/f'{name}.pt',weights_only=True)
        for l in chat_layers:chat[l][name]=b['G'][:,l].float().numpy()
    full={'original':make_dict(raw,[14,20]),'plain':make_dict(raw,[24]),'chat':make_dict(chat,chat_layers)}
    solvers={n:GramSolver(d) for n,d in full.items()}
    ledger=Ledger('qwen_confirm')
    suites=[('heldout_known',[n for n in ALL_TASKS if n not in DEV_TASKS]),
            ('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]
    for suite,names in suites:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            if suite=='heldout_known':
                ds={'original':make_dict(raw,[14,20],exclude=name),'plain':make_dict(raw,[24],exclude=name),
                    'chat':make_dict(chat,chat_layers,exclude=name)}
                ss={n:GramSolver(d) for n,d in ds.items()}
            else:ds,ss=full,solvers
            queries=task.eval_queries;pp=[zs_prompt(x) for x,y in queries]
            cp=[render(hlm,p) for p in pp]
            for seed in [20,21,22]:
                methods=['plain_original_cass','plain_selected_cass','chat_selected_cass','chat_z',
                         'chat_replace','plain_icl4','chat_icl4','chat_zero']
                if suite=='heldout_known':methods+=['chat_selected_oracle']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4)
                examples=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                cache=HERE/f'cache_qwen_confirm_{name}_{seed}.pt'
                if cache.exists():data=torch.load(cache,weights_only=True)
                else:
                    zr,hr=get_extraction(hlm,examples,all_layers,seed,False)
                    zc,hc=get_extraction(hlm,examples,chat_layers,seed,True)
                    data=dict(raw=zr,chat=zc,chat_h=hc);torch.save(data,cache)
                codes={};zs={};operators={}
                for key,D in ds.items():
                    zl=selected_zlist(D,data['chat' if key=='chat' else 'raw']);z=np.mean(zl,axis=0)
                    code=ss[key].solve(zl);codes[key]=code;zs[key]=z
                    gain=cfg['gain'] if key=='chat' else 1.
                    operators[key]=ops_for(D,code,gamma=gain,delta_vec=z)[0]
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    prompts=cp if method.startswith('chat_') else pp
                    ops=None;lys=None;schedule='all'
                    if method in ['plain_original_cass','plain_selected_cass']:
                        key='original' if method=='plain_original_cass' else 'plain'
                        ops,lys=operators[key],ds[key].layers
                    elif method in ['chat_selected_cass','chat_z']:
                        lys=chat_layers;schedule=cfg['schedule']
                        ops=operators['chat'] if method=='chat_selected_cass' else ops_for(
                            ds['chat'],codes['chat'],gamma=cfg['gain'],injection='additive',delta_vec=zs['chat'])[0]
                    elif method=='chat_replace':
                        lys=chat_layers;ops=[]
                        for l in lys:
                            vec=data['chat_h'][l].mean(0).to('cuda')
                            def op(h,v=vec):return v.unsqueeze(0).expand_as(h).clone()
                            ops.append(op)
                    elif method=='chat_selected_oracle':
                        lys=chat_layers;schedule=cfg['schedule']
                        if cfg['kind']=='affine':ops,_=oracle_ops(full['chat'],name,gamma=cfg['gain'])
                        else:
                            from cass.steer import make_additive_op
                            ops=[make_additive_op(full['chat'].per_layer[l].anchors[name],gamma=cfg['gain']) for l in lys]
                    elif method.endswith('icl4'):
                        prompts=[icl_prompt(examples,x) for x,y in queries]
                        if method.startswith('chat_'):prompts=[render(hlm,p) for p in prompts]
                    preds=generate(hlm,prompts,ops,lys,schedule)
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,demos=examples,chat_config=cfg,
                                   support=codes['chat'].support,residual=codes['chat'].residual,
                                   znorm=float(np.linalg.norm(zs['chat'])))
                    print(suite,name,seed,method,round(acc,4),flush=True)
    (HERE/'qwen_confirm_done.json').write_text('{"complete":true}\n')


if __name__=='__main__':main()
