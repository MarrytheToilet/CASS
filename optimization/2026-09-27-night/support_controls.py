"""Predeclared correction-only support swaps for the frozen combined variant."""
import json
import os
import numpy as np
import torch
from common import HERE,ALL_TASKS,load_g,make_dict,generate,Ledger
from efficient_ops import selected_zlist
from freeze_extensions import freeze
from variant_ops import OperatorFactory
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import zs_prompt
from cass.compound import COMPOUND_REGISTRY,compound_components,load_compound
from cass.dictionary import subcoherence_matrix
from cass.solver import SparseCode


def candidates(D,support,truth,coherence,rng):
    pos={n:i for i,n in enumerate(D.task_names)}
    def ranks(n):return tuple(D.per_layer[l].bases[n].shape[1] for l in D.layers)
    def stats(s):
        if len(s)<2:return np.zeros(2)
        m=coherence[np.ix_([pos[n] for n in s],[pos[n] for n in s])]
        v=m[np.triu_indices(len(s),1)];return np.array([v.mean(),v.max()])
    reference=stats(support);excluded=set(support)|set(truth)
    pools=[[n for n in D.task_names if n not in excluded and ranks(n)==ranks(s)] for s in support]
    if not support or any(not p for p in pools):return [],dict(reason='No alternative with identical per-layer block ranks')
    found={}
    for _ in range(6000):
        names=tuple(rng.choice(p) for p in pools)
        if len(set(names))!=len(names):continue
        key=tuple(sorted(names));v=stats(names)
        if key not in found:found[key]=(float(np.linalg.norm((v-reference)/[.02,.03])),names,v)
    selected=sorted(found.values(),key=lambda x:(x[0],x[1]))[:3]
    details=[]
    for distance,names,v in selected:
        gap=np.abs(v-reference)
        details.append(dict(support=names,coherence=v.tolist(),mean_gap=float(gap[0]),max_gap=float(gap[1]),
                            tolerance_pass=bool(gap[0]<=.02 and gap[1]<=.03),rank_match=True))
    return [list(x[1]) for x in selected],dict(reference=reference.tolist(),candidates=len(found),selected=details)


def swapped_ops(factory,cfg,strength,names,weights,norm_match=False,off=False):
    assert cfg['kind']=='context_residual','Implement controls explicitly for selected geometry'
    D=factory.D;gate=factory.gate;gamma=cfg['gamma'];ops=[]
    for l,(qr,d,mr) in zip(D.layers,factory.geometry(cfg['kind'])):
        qa=torch.tensor(np.linalg.qr(np.concatenate([D.per_layer[l].bases[n] for n in names],axis=1))[0],device='cuda',dtype=torch.float32)
        ma=torch.tensor(sum(w*factory.clean[n][l] for w,n in zip(weights,names)),device='cuda',dtype=torch.float32)
        def op(h,q0=qr,q=qa,v=d,m0=mr,m=ma):
            h=h.float();base=h+gamma*v
            corr=((m-base)@q)@q.T
            if norm_match:
                ref=((m0-base)@q0)@q0.T
                corr=corr*(ref.norm(dim=1,keepdim=True)/(corr.norm(dim=1,keepdim=True)+1e-12))
            return base+(0. if off else strength*gate)*corr
        ops.append(op)
    return ops,D.layers


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['combined']['config'];a=cfg['null_fraction'];layers=[12,16]
    G=load_g('llama31-8b',layers);clean={}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:G[l][name]=(1-a)*G[l][name]+a*b['G_by_layer'][l].float().numpy()
        clean[name]={l:b['clean_mean'][l].numpy() for l in layers}
    D=make_dict(G,layers);solver=GramSolver(D)
    coherence=np.maximum.reduce([subcoherence_matrix(D.per_layer[l]) for l in layers])
    hlm=HookedLM('llama31-8b');ledger=Ledger('support_controls');metadata={}
    for ti,name in enumerate(COMPOUND_REGISTRY):
        task=load_compound(name);truth=compound_components(name);queries=task.eval_queries;pp=[zs_prompt(x) for x,y in queries]
        for seed in [20,21,22]:
            data=torch.load(HERE/f'cache_signature_confirm_{name}_{seed}.pt',weights_only=True)
            zl=selected_zlist(D,{l:(1-a)*data['raw'][l]+a*data['null'][l] for l in layers});z=np.mean(zl,axis=0)
            code=solver.solve(zl);factory=OperatorFactory(D,code,zl,clean)
            V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
            agreement=float((V@V.T)[np.triu_indices(4,1)].mean())
            strength={'coverage':max(0.,1-code.residual**2),'uncertainty':float(np.clip(1-agreement,0,1))}.get(cfg['correction'],cfg['correction'])
            wrong,meta=candidates(D,code.support,truth,coherence,np.random.default_rng(372000+ti*100+seed))
            metadata[f'{name}|{seed}']=meta
            tw=np.array([np.linalg.norm(D.bases[n].T@z) for n in truth]);tw=tw/tw.sum()
            arms=[('inferred',code.support,factory.w,False,False),('no_correction',code.support,factory.w,False,True),
                  ('true_correction',truth,tw,False,False),('true_correction_norm',truth,tw,True,False),
                  ('true_support',truth,tw,False,False)]
            for i,names in enumerate(wrong):
                arms.extend([(f'wrong_{i}',names,factory.w,False,False),(f'wrong_norm_{i}',names,factory.w,True,False)])
            expected,_=factory.ops(cfg['kind'],cfg['gamma'],strength)
            actual,_=swapped_ops(factory,cfg,strength,code.support,factory.w)
            probe=torch.randn(25,D.d,device='cuda')
            for x,y in zip(expected,actual):torch.testing.assert_close(x(probe),y(probe),atol=3e-5,rtol=3e-5)
            for label,names,w,normed,off in arms:
                if ledger.has(name,seed,label):continue
                ops,lys=swapped_ops(factory,cfg,strength,names,w,normed,off)
                if label=='true_support':
                    coefficients={n:D.bases[n].T@z for n in truth}
                    reconstruction=sum(D.bases[n]@coefficients[n] for n in truth)
                    tc=SparseCode(coefficients,truth,reconstruction,float(np.linalg.norm(z-reconstruction)/np.linalg.norm(z)),0.)
                    tf=OperatorFactory(D,tc,zl,clean);ops,lys=tf.ops(cfg['kind'],cfg['gamma'],strength)
                preds=generate(hlm,pp,ops,lys,cfg['schedule'])
                matched=meta['selected'][int(label.split('_')[-1])]['tolerance_pass'] if label.startswith('wrong') else None
                acc=ledger.add(name,seed,label,queries,preds,suite='compound',config_parameters=cfg,
                               inferred_support=code.support,control_support=names,gate=factory.gate,strength=strength,
                               rank_coherence_matched=matched,correction_norm_matched=normed)
                print(name,seed,label,round(acc,4),flush=True)
            (HERE/'support_controls_matching.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (HERE/'support_controls_done.json').write_text('{"complete":true}\n')

if __name__=='__main__':main()
