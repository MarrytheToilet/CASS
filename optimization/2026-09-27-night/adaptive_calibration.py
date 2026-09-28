"""Four-demonstration cross-validation of gain, schedule, and replacement."""
import argparse
import json
import os
import time
import numpy as np
import torch
from common import HERE,ROOT,ALL_TASKS,load_g,make_dict,generate,Ledger,correction_ops
from fast_solver import GramSolver
from efficient_ops import extract_selected,selected_zlist,selected_hiddens
from cass.models import HookedLM
from cass.tasks import load_task,synthetic_tasks,zs_prompt,icl_prompt
from cass.compound import COMPOUND_REGISTRY,load_compound
from cass.pipeline import ops_for


@torch.no_grad()
def teacher_loss(hlm,prompt,target,ops,layers,schedule):
    prefix=hlm.tok.encode(prompt)
    suffix=hlm.tok.encode(' '+target,add_special_tokens=False)
    assert len(suffix)>0
    ids=torch.tensor([prefix+suffix],device='cuda')
    boundary=len(prefix)-1
    handles=[]
    for op,l in zip(ops or [],layers or []):
        def hook(module,inputs,output,operation=op):
            h=output[0] if isinstance(output,tuple) else output
            stop=boundary+1 if schedule=='prefill' else boundary+len(suffix)
            hh=h[:,boundary:stop,:]
            h[:,boundary:stop,:]=operation(hh.reshape(-1,h.shape[-1])).reshape_as(hh).to(h.dtype)
        handles.append(hlm.layers[l-1].register_forward_hook(hook))
    try:
        logits=hlm.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
        scored=logits[0,boundary:boundary+len(suffix)].float()
        target_ids=torch.tensor(suffix,device='cuda')
        return float(torch.nn.functional.cross_entropy(scored,target_ids))
    finally:
        for handle in handles:handle.remove()


def replacement_ops(Z,layers):
    ops=[]
    for l in layers:
        v=Z[l].mean(0).to('cuda')
        def op(h,vec=v):return vec.unsqueeze(0).expand_as(h).clone()
        ops.append(op)
    return ops


def extract_all(hlm,examples,layers,seed):
    Z=extract_selected(hlm,examples,layers,seed=seed)
    prompts=[icl_prompt([e for i,e in enumerate(examples) if i!=j],x) for j,(x,y) in enumerate(examples)]
    H=selected_hiddens(hlm,prompts,layers,batch_size=4)
    return {'Z':Z,'H':H}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suites',default='loto,novel,compound')
    parser.add_argument('--seeds',default='20,21,22');parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    hlm=HookedLM('llama31-8b');layers=[12,16]
    G=load_g('llama31-8b',layers);Dfull=make_dict(G,layers);Sfull=GramSolver(Dfull)
    candidates=[]
    for mode in ['cass','z_only']:
        for gain in [.5,1.,1.5,2.,3.]:
            for schedule in ['all','prefill']:
                candidates.append(dict(mode=mode,gamma=gain,schedule=schedule))
    candidates += [dict(mode='replace',schedule=s) for s in ['all','prefill']]
    ledger=Ledger('adaptive_eval')
    log=(HERE/'adaptive_selection.jsonl').open('a')
    suites={'loto':ALL_TASKS,'novel':list(synthetic_tasks()),'compound':list(COMPOUND_REGISTRY)}
    for suite in args.suites.split(','):
        for name in suites[suite][:args.limit]:
            task=load_compound(name) if suite=='compound' else load_task(name)
            D=make_dict(G,layers,exclude=name) if suite=='loto' else Dfull
            solver=GramSolver(D) if suite=='loto' else Sfull
            queries=task.eval_queries;prompts=[zs_prompt(x) for x,y in queries]
            for seed in map(int,args.seeds.split(',')):
                methods=['adaptive_cass','fixed_policy_z','fixed_policy_no_correction','adaptive_z',
                         'original_cass','original_z','replace','icl4']
                if all(ledger.has(name,seed,m) for m in methods):continue
                rng=np.random.default_rng(100*seed+4)
                examples=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                start=time.perf_counter()
                cache=HERE/f'cache_adaptive_{name}_{seed}.pt'
                cache_reused=cache.exists()
                if cache_reused:extractions=torch.load(cache,weights_only=True)
                else:
                    extractions={'full':extract_all(hlm,examples,layers,seed),'folds':[]}
                    for j in range(4):
                        ex=[e for i,e in enumerate(examples) if i!=j]
                        assert examples[j][0] not in {x for x,y in ex}
                        extractions['folds'].append(extract_all(hlm,ex,layers,seed))
                    torch.save(extractions,cache)
                extraction_seconds=time.perf_counter()-start
                losses=np.zeros((len(candidates),4))
                for j,(x,y) in enumerate(examples):
                    fold=extractions['folds'][j]
                    zl=selected_zlist(D,fold['Z']);code=solver.solve(zl)
                    rep=replacement_ops(fold['H'],layers)
                    for ci,cfg in enumerate(candidates):
                        if cfg['mode']=='replace':ops=rep
                        elif cfg['mode']=='z_only':
                            ops,_=ops_for(D,code,gamma=cfg['gamma'],injection='additive',delta_vec=np.mean(zl,axis=0))
                        else:
                            ops,_=correction_ops(D,code,zl,gamma=cfg['gamma'],
                                                 correction=1. if cfg['mode']=='cass' else 0.)
                        losses[ci,j]=teacher_loss(hlm,zs_prompt(x),y,ops,layers,cfg['schedule'])
                scores=losses.mean(1)
                ci=min([i for i,c in enumerate(candidates) if c['mode']!='z_only'],key=lambda i:scores[i])
                zi=min([i for i,c in enumerate(candidates) if c['mode']!='cass'],key=lambda i:scores[i])
                full=extractions['full'];zl=selected_zlist(D,full['Z']);code=solver.solve(zl)
                rep=replacement_ops(full['H'],layers)
                elapsed=time.perf_counter()-start
                selection=dict(task=name,suite=suite,seed=seed,candidates=candidates,
                               losses=losses.tolist(),cass_index=ci,z_index=zi,
                               calibration_seconds=elapsed,cache_reused=cache_reused,extraction_seconds=extraction_seconds,
                               extraction_sequences=48+4*36,replacement_sequences=4+4*3,
                               teacher_forcing_model_calls=4*len(candidates),examples=examples,
                               support=code.support,residual=code.residual,
                               znorm=float(np.linalg.norm(np.mean(zl,axis=0))))
                log.write(json.dumps(selection)+'\n');log.flush()
                for method in methods:
                    if ledger.has(name,seed,method):continue
                    pp=prompts;schedule='all'
                    if method in ['adaptive_cass','fixed_policy_z','fixed_policy_no_correction','adaptive_z']:
                        cfg=candidates[zi if method=='adaptive_z' else ci]
                        schedule=cfg['schedule']
                        if cfg['mode']=='replace':ops=rep
                        elif method in ['fixed_policy_z','adaptive_z']:
                            ops,_=ops_for(D,code,gamma=cfg['gamma'],injection='additive',delta_vec=np.mean(zl,axis=0))
                        else:
                            correction=1. if method=='adaptive_cass' else 0.
                            ops,_=correction_ops(D,code,zl,gamma=cfg['gamma'],correction=correction)
                    elif method=='original_cass':
                        ops,_=correction_ops(D,code,zl)
                    elif method=='original_z':
                        ops,_=ops_for(D,code,injection='additive',delta_vec=np.mean(zl,axis=0))
                    elif method=='replace':ops=rep
                    else:ops=None;pp=[icl_prompt(examples,x) for x,y in queries]
                    preds=generate(hlm,pp,ops,layers if ops else None,schedule)
                    acc=ledger.add(name,seed,method,queries,preds,suite=suite,
                                   selected=candidates[ci],z_selected=candidates[zi],calibration_seconds=elapsed,
                                   support=code.support,residual=code.residual,znorm=selection['znorm'])
                    print(suite,name,seed,method,round(acc,4),flush=True)
    (HERE/'adaptive_done.json').write_text(json.dumps(dict(suites=args.suites,seeds=args.seeds)))


if __name__=='__main__':main()
