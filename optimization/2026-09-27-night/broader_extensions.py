"""Frozen operator extensions on every predeclared broader task."""
import json
import os
import random
import numpy as np
import torch
from common import HERE,ALL_TASKS,load_g,make_dict,generate
from broader_tasks import load_broader,score,clean_answer
from extension_confirm import dictionary_free_ops
from freeze_extensions import freeze
from variant_ops import OperatorFactory
from efficient_ops import selected_hiddens,selected_zlist
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import zs_prompt,icl_prompt,build_fewshot_pair_prompts


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    selected=freeze();hlm=HookedLM('llama31-8b');layers=[12,16]
    refined_path=HERE/'dictionary_free_refined_selected.json'
    baseline_search_size=360
    if refined_path.exists() and json.loads(refined_path.read_text())['config']==selected['dictionary_free']['config']:
        baseline_search_size=552
    raw=load_g('llama31-8b',layers);null={l:{} for l in layers};clean={}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:null[l][name]=b['G_by_layer'][l].float().numpy()
        clean[name]={l:b['clean_mean'][l].numpy() for l in layers}
    fractions={0.,1.,selected['combined']['config']['null_fraction']}
    ds={f:make_dict({l:{n:(1-f)*raw[l][n]+f*null[l][n] for n in ALL_TASKS} for l in layers},layers) for f in fractions}
    ss={f:GramSolver(D) for f,D in ds.items()}
    path=HERE/'broader_extensions.jsonl';done=set();baseline={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);done.add((r['task'],r['seed'],r['method']))
    for line in (HERE/'broader_llama31-8b.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['method'] in ['icl4','zero']:baseline[(r['task'],r['seed'],r['method'])]=r
    with path.open('a') as out:
        for name,task in load_broader().items():
            queries=task.eval_queries;max_tokens=128 if name.startswith(('code-','long-')) else 32
            for seed in [20,21,22]:
                methods=['combined','combined_no_correction','oneshot','oneshot_no_correction',
                         'dictionary_free','null_no_correction','icl1','icl4']+(['zero'] if seed==20 else [])
                if all((name,seed,m) in done for m in methods):continue
                rng=np.random.default_rng(100*seed+4);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                Z=torch.load(HERE/f'cache_broader_llama31-8b_{name}_{seed}.pt',weights_only=True)
                p=HERE/f'cache_broader_null_llama31-8b_{name}_{seed}.pt'
                if p.exists():zn=torch.load(p,weights_only=True)
                else:
                    cp,_=build_fewshot_pair_prompts(ex,random.Random(9000+seed),n_reps=6)
                    hp=selected_hiddens(hlm,cp,layers,batch_size=4);hz=selected_hiddens(hlm,[zs_prompt(x) for x,y in ex],layers,batch_size=4)
                    zn={l:hp[l].reshape(4,6,-1).mean(1)-hz[l] for l in layers};torch.save(zn,p)
                data=dict(raw={l:Z[:,l] for l in layers},null=zn);factories={};codes={};zs={}
                for f,D in ds.items():
                    zl=selected_zlist(D,{l:(1-f)*data['raw'][l]+f*zn[l] for l in layers});code=ss[f].solve(zl)
                    factories[f]=OperatorFactory(D,code,zl,clean);codes[f]=code;zs[f]=zl
                p0=[zs_prompt(x) for x,y in queries];p1=[icl_prompt(ex[:1],x) for x,y in queries]
                for method in methods:
                    if (name,seed,method) in done:continue
                    key=(name,seed,method)
                    if key in baseline:
                        row=baseline[key].copy()
                        assert row['queries']==[list(q) for q in queries] and row['demos']==[list(e) for e in ex]
                        row['reused_baseline_source']='broader_llama31-8b.jsonl'
                    else:
                        cfg={};prompts=p0;ops=None;lys=None;schedule='all';f=0.;strength=0.
                        if method.startswith(('combined','oneshot')):
                            base='combined' if method.startswith('combined') else 'oneshot';cfg=selected[base]['config'].copy()
                            f=cfg.get('null_fraction',0.);V=np.asarray(zs[f]);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
                            agreement=float((V@V.T)[np.triu_indices(4,1)].mean())
                            strength={'coverage':max(0.,1-codes[f].residual**2),'uncertainty':float(np.clip(1-agreement,0,1))}.get(cfg['correction'],cfg['correction'])
                            if method.endswith('no_correction'):strength=0.
                            ops,lys=factories[f].ops(cfg['kind'],cfg['gamma'],strength);schedule=cfg['schedule']
                            if base=='oneshot':prompts=p1
                        elif method=='dictionary_free':
                            cfg=selected['dictionary_free']['config'];ops,lys=dictionary_free_ops(cfg,data,hlm,ex);schedule=cfg['schedule']
                        elif method=='null_no_correction':
                            f=1.;ops,lys=factories[f].ops('original',1.5,0.);schedule='prefill'
                        elif method=='icl1':prompts=p1
                        else:raise ValueError((name,seed,method,'missing matching baseline'))
                        preds=generate(hlm,prompts,ops,lys,schedule,batch_size=4,max_new_tokens=max_tokens,stop_after_answer=(HERE/'answer_stop_verified.json').exists())
                        scores=[score(name,p,y) for p,(x,y) in zip(preds,queries)]
                        row=dict(task=name,seed=seed,method=method,family=task.family,score=float(np.mean(scores)),
                                 per_example_score=scores,queries=queries,predictions=preds,demos=ex,
                                 variant_config=cfg,answer_stop_verified=(HERE/'answer_stop_verified.json').exists(),applied_correction=strength,support=codes[f].support,
                                 dictionary_free_search_size=baseline_search_size if method=='dictionary_free' else None,
                                 residual=codes[f].residual,max_new_tokens=max_tokens,
                                 mean_input_tokens=float(np.mean([len(hlm.tok.encode(p)) for p in prompts])))
                        if name.startswith('long-'):
                            row['position_accuracy']=float(np.mean([sum(a==b for a,b in zip(clean_answer(p).split(),y.split()))/len(y.split()) for p,(x,y) in zip(preds,queries)]))
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();done.add(key)
                    print(name,seed,method,round(row['score'],4),flush=True)
    (HERE/'broader_extensions_done.json').write_text('{"complete":true}\n')

if __name__=='__main__':main()
