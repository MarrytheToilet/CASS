"""Freeze the contrast mixture on development data, then evaluate transfer."""
import json
import os
import random
from collections import defaultdict
import numpy as np
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,load_g,make_dict,generate,Ledger,correction_ops
from efficient_ops import selected_hiddens,selected_zlist
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt,build_fewshot_pair_prompts
from cass.compound import COMPOUND_REGISTRY,load_compound
from cass.pipeline import ops_for


def freeze():
    grouped=defaultdict(list)
    for line in (HERE/'signature_dev.jsonl').read_text().splitlines():
        r=json.loads(line);grouped[r['config']].append(r['acc'])
    assert len(grouped)==80 and all(len(v)==24 for v in grouped.values())
    means={k:float(np.mean(v)) for k,v in grouped.items()}
    overall=max(means,key=means.get)
    nonzero=max([k for k in means if json.loads(k)['correction']>0],key=means.get)
    record=dict(best_overall=json.loads(overall),best_correction=json.loads(nonzero),
                overall_development_accuracy=means[overall],correction_development_accuracy=means[nonzero],
                all_development_means=means,selection_queries='12 fixed development tasks, two seeds; no eval queries')
    (HERE/'signature_selected.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def extract(hlm,examples,layers,seed):
    pp,pc=build_fewshot_pair_prompts(examples,random.Random(9000+seed),n_reps=6)
    hp=selected_hiddens(hlm,pp+pc,layers,batch_size=24)
    hz=selected_hiddens(hlm,[zs_prompt(x) for x,y in examples],layers,batch_size=4)
    raw={l:(h[:24]-h[24:]).reshape(4,6,-1).mean(1) for l,h in hp.items()}
    null={l:h[:24].reshape(4,6,-1).mean(1)-hz[l] for l,h in hp.items()}
    replacement_prompts=[icl_prompt([e for i,e in enumerate(examples) if i!=j],x) for j,(x,y) in enumerate(examples)]
    H=selected_hiddens(hlm,replacement_prompts,layers,batch_size=4)
    return dict(raw=raw,null=null,replace=H)


def main():
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    selected=freeze();layers=[12,16]
    fractions=sorted({0.,selected['best_overall']['null_fraction'],selected['best_correction']['null_fraction']})
    raw=load_g('llama31-8b',layers);null={l:{} for l in layers}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:null[l][name]=b['G_by_layer'][l].float().numpy()
    mixtures={a:{l:{n:(1-a)*raw[l][n]+a*null[l][n] for n in ALL_TASKS} for l in layers} for a in fractions}
    full={a:make_dict(G,layers) for a,G in mixtures.items()};solvers={a:GramSolver(D) for a,D in full.items()}
    hlm=HookedLM('llama31-8b');ledger=Ledger('signature_confirm')
    for suite,names in [('loto',ALL_TASKS),('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in names:
            task=load_compound(name) if suite=='compound' else load_task(name)
            ds={a:make_dict(G,layers,exclude=name) for a,G in mixtures.items()} if suite=='loto' else full
            ss={a:GramSolver(D) for a,D in ds.items()} if suite=='loto' else solvers
            queries=task.eval_queries;pp=[zs_prompt(x) for x,y in queries]
            for seed in [20,21,22]:
                methods=['original','original_z','best_correction','best_overall','matched_z','matched_no_correction','replace','icl4']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4)
                ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                path=HERE/f'cache_signature_confirm_{name}_{seed}.pt'
                if path.exists():data=torch.load(path,weights_only=True)
                else:data=extract(hlm,ex,layers,seed);torch.save(data,path)
                zs={};codes={}
                for a,D in ds.items():
                    Z={l:(1-a)*data['raw'][l]+a*data['null'][l] for l in layers}
                    zs[a]=selected_zlist(D,Z);codes[a]=ss[a].solve(zs[a])
                original_norm=float(np.linalg.norm(np.mean(zs[0.],axis=0)))
                rep=[]
                for l in layers:
                    v=data['replace'][l].mean(0).to('cuda')
                    def op(h,vec=v):return vec.unsqueeze(0).expand_as(h).clone()
                    rep.append(op)
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    prompts=pp;schedule='all';cfg={};fraction=0.
                    if method in ['original','original_z']:
                        if method=='original':ops,_=correction_ops(ds[0.],codes[0.],zs[0.])
                        else:ops,_=ops_for(ds[0.],codes[0.],injection='additive',delta_vec=np.mean(zs[0.],axis=0))
                    elif method in ['best_correction','best_overall','matched_z','matched_no_correction']:
                        cfg=selected['best_overall' if method=='best_overall' else 'best_correction'].copy()
                        fraction=cfg.pop('null_fraction');schedule=cfg.pop('schedule')
                        if method=='matched_z':
                            ops,_=ops_for(ds[fraction],codes[fraction],gamma=cfg['gamma'],injection='additive',delta_vec=np.mean(zs[fraction],axis=0))
                        else:
                            if method=='matched_no_correction':cfg['correction']=0.
                            ops,_=correction_ops(ds[fraction],codes[fraction],zs[fraction],**cfg)
                    elif method=='replace':ops=rep
                    else:ops=None;prompts=[icl_prompt(ex,x) for x,y in queries]
                    preds=generate(hlm,prompts,ops,layers if ops else None,schedule)
                    code=codes[fraction]
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,development_task=name in DEV_TASKS,
                                   config_parameters=cfg,null_fraction=fraction,schedule=schedule,demos=ex,
                                   support=code.support,residual=code.residual,
                                   znorm=float(np.linalg.norm(np.mean(zs[fraction],axis=0))),original_znorm=original_norm)
                    print(suite,name,seed,method,round(acc,4),flush=True)
    (HERE/'signature_confirm_done.json').write_text('{"complete":true}\n')


if __name__=='__main__':main()
