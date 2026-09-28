"""Execution, support and coding statistics at 32/64/128/256 entries."""
import json
import argparse
import os
import time
import numpy as np
import torch
from common import HERE,ALL_TASKS,load_g,make_dict,generate,Ledger
from scale_bank import ordered_bank_names
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound,compound_components
from cass.extract import extract_fewshot_z
from cass.pipeline import z_list_from_Z,ops_for

parser=argparse.ArgumentParser();parser.add_argument('--fixed-shared',action='store_true');parser.add_argument('--context',action='store_true');args=parser.parse_args()
stem=('scale_context' if args.context else 'scale_eval')+('_fixed_shared' if args.fixed_shared else '')
from fixed_shared import build_fixed_shared
torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
hlm=HookedLM('llama31-8b');layers=[12,16];G=load_g('llama31-8b',layers)
clean={};cfg=None;context_reference={}
if args.context:
    from freeze_extensions import freeze
    from efficient_ops import selected_zlist
    from variant_ops import OperatorFactory
    cfg=freeze()['combined']['config'];fraction=cfg['null_fraction']
    for line in (HERE/'extension_confirm.jsonl').read_text().splitlines():
        rec=json.loads(line)
        if rec['config']=='combined':context_reference[(rec['task'],rec['seed'])]=rec['predictions']
    for name in ALL_TASKS:
        blob=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:G[l][name]=(1-fraction)*G[l][name]+fraction*blob['G_by_layer'][l].float().numpy()
        clean[name]={l:blob['clean_mean'][l].numpy() for l in layers}
reference=make_dict(G,layers)
order=ordered_bank_names()
for name in order:
    blob=torch.load(HERE/'scale_activations'/f'{name}.pt',weights_only=True)
    for l in layers:
        G[l][name]=blob['G_by_layer'][l].float().numpy()
        if args.context:G[l][name]=(1-fraction)*G[l][name]+fraction*blob['null_by_layer'][l].float().numpy()
    if args.context:clean[name]={l:blob['clean_mean'][l].numpy() for l in layers}
ledger=Ledger(stem);offline=[]
for size in [32,64,128,256]:
    names=ALL_TASKS+order[:size-32]
    t=time.perf_counter();subset={l:{n:G[l][n] for n in names} for l in layers}
    D=build_fixed_shared(subset,layers,reference) if args.fixed_shared else make_dict(subset,layers)
    build_s=time.perf_counter()-t;t=time.perf_counter();solver=GramSolver(D);gram_s=time.perf_counter()-t
    row=dict(size=size,shared_direction='frozen32' if args.fixed_shared else 'recomputed',operator='contextual_extension' if args.context else 'submitted',dictionary_seconds=build_s,gram_seconds=gram_s,
             total_columns=solver.A.shape[1],joint_basis_bytes=sum(a.nbytes for a in D.bases.values()),
             gram_bytes=solver.H.nbytes,concatenated_basis_bytes=solver.A.nbytes,
             scope='32 original natural-language tasks plus explicitly synthetic correlated tasks')
    coherences=[];max_pair=None;max_value=-1.
    for i,ni in enumerate(D.task_names):
        for nj in D.task_names[:i]:
            value=float(np.linalg.svd(solver.H[solver.slices[ni],solver.slices[nj]],compute_uv=False)[0])
            coherences.append(value)
            if value>max_value:max_value=value;max_pair=[ni,nj]
    row.update(maximum_coherence=max_value,maximum_pair=max_pair,median_coherence=float(np.median(coherences)))
    offline.append(row);(HERE/(stem+'_offline.json')).write_text(json.dumps(offline,indent=2)+'\n')
    for suite,targets in [('novel',list(synthetic_tasks())),('compound',list(COMPOUND_REGISTRY))]:
        for name in targets:
            task=load_compound(name) if suite=='compound' else load_task(name)
            queries=task.eval_queries;pp=[zs_prompt(x) for x,y in queries]
            for seed in [20,21,22]:
                if args.context:
                    data=torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True)
                    zl=selected_zlist(D,{l:(1-fraction)*data['raw'][l]+fraction*data['null'][l] for l in layers})
                else:
                    path=HERE/f'cache_confirm_{name}_{seed}.pt'
                    if path.exists():Z=torch.load(path,weights_only=True)
                    else:
                        rng=np.random.default_rng(100*seed+4)
                        ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                        Z=extract_fewshot_z(hlm,ex,seed=seed,batch_size=24);torch.save(Z,path)
                    zl=z_list_from_Z(D,Z)
                z=np.mean(zl,axis=0)
                for mode in ['full','shortlist32']:
                    label=f'size{size}_{mode}'
                    if ledger.has(name,seed,label):continue
                    t=time.perf_counter();usedD,usedS=D,solver;shortlist=None
                    if mode=='shortlist32' and size>32:
                        b=solver.A.T@z
                        ranked=sorted(D.task_names,key=lambda n:np.linalg.norm(b[solver.slices[n]])/solver.weights[n],reverse=True)
                        keep=set(ranked[:32]);shortlist=[n for n in D.task_names if n in keep]
                        usedS=solver.subset(shortlist);usedD=usedS.D
                    code=usedS.solve(zl);coding_s=time.perf_counter()-t
                    schedule='all';strength=None
                    if args.context:
                        V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
                        strength={'uncertainty':float(np.clip(1-(V@V.T)[np.triu_indices(4,1)].mean(),0,1)),
                                  'coverage':max(0.,1-code.residual**2)}.get(cfg['correction'],cfg['correction'])
                        factory=OperatorFactory(usedD,code,zl,clean)
                        ops,lys=factory.ops(cfg['kind'],cfg['gamma'],strength);schedule=cfg['schedule']
                    else:ops,lys=ops_for(usedD,code,delta_vec=z)
                    preds=generate(hlm,pp,ops,lys,schedule)
                    if args.context and size==32:
                        assert preds==context_reference[(name,seed)],(name,seed,'32-skill context baseline mismatch')
                    truths=compound_components(name) if suite=='compound' else []
                    recall=len(set(code.support)&set(truths))/len(truths) if truths else None
                    acc=ledger.add(name,seed,label,queries,preds,suite=suite,size=size,solver_mode=mode,
                                   support=code.support,residual=code.residual,znorm=float(np.linalg.norm(z)),
                                   coding_seconds=coding_s,constituent_recall=recall,truth=truths,shortlist=shortlist,
                                   operator='contextual_extension' if args.context else 'submitted',
                                   config_parameters=cfg,applied_correction=strength)
                    print(size,suite,name,seed,mode,round(acc,4),round(coding_s,3),flush=True)
(HERE/(stem+'_done.json')).write_text('{"complete":true}\n')
