"""Frozen original CASS on all predeclared broader tasks, with matched baselines."""
import json
import os
import random
import time
import numpy as np
import torch
from common import HERE, load_g, make_dict, generate, correction_ops
from broader_tasks import load_broader,score,clean_answer
from fast_solver import GramSolver
from efficient_ops import selected_hiddens
from cass.models import HookedLM
from cass.tasks import icl_prompt,zs_prompt,build_fewshot_pair_prompts
from cass.extract import extract_fewshot_z
from cass.pipeline import z_list_from_Z,ops_for

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
model=os.environ.get('CASS_PROBE_MODEL','llama31-8b')
variant='signature' if '--signature' in __import__('sys').argv else 'original'
layers=[12,16] if model=='llama31-8b' else [14,20]
hlm=HookedLM(model)
rawG=load_g(model,layers)
cfg=dict(gamma=1.,null_fraction=0.,schedule='all')
if variant=='signature':
    assert model=='llama31-8b'
    cfg=json.loads((HERE/'signature_selected.json').read_text())['best_correction']
    fraction=cfg['null_fraction'];mixed={l:{} for l in layers}
    for name in rawG[layers[0]]:
        blob=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:mixed[l][name]=(1-fraction)*rawG[l][name]+fraction*blob['G_by_layer'][l].float().numpy()
    D=make_dict(mixed,layers)
else:D=make_dict(rawG,layers)
solver=GramSolver(D)
path=HERE/f"broader_{'signature_' if variant=='signature' else ''}{model}.jsonl"
baseline_rows={}
baseline_path=HERE/f'broader_{model}.jsonl'
if variant=='signature' and baseline_path.exists():
    for line in baseline_path.read_text().splitlines():
        row=json.loads(line)
        if row['method'] in ['replace','icl4','zero']:baseline_rows[(row['task'],row['seed'],row['method'])]=row
done=set()
if path.exists():
    for line in path.read_text().splitlines():
        r=json.loads(line); done.add((r['task'],r['seed'],r['method']))
f=path.open('a')
start=time.time()
for name,task in load_broader().items():
    queries=task.eval_queries
    prompts=[zs_prompt(x) for x,y in queries]
    max_tokens=128 if name.startswith(('code-','long-')) else 32
    for seed in [20,21,22]:
        rng=np.random.default_rng(100*seed+4)
        examples=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
        methods=['cass','z_only','replace','icl4']+(['zero'] if seed==20 else [])
        if all((name,seed,m) in done for m in methods): continue
        cache=HERE/f'cache_broader_{model}_{name}_{seed}.pt'
        if cache.exists(): Z=torch.load(cache,weights_only=True)
        else:
            Z=extract_fewshot_z(hlm,examples,seed=seed,batch_size=4)
            torch.save(Z,cache)
        if variant=='signature':
            nc=HERE/f'cache_broader_null_{model}_{name}_{seed}.pt'
            if nc.exists():zn=torch.load(nc,weights_only=True)
            else:
                clean,_=build_fewshot_pair_prompts(examples,random.Random(9000+seed),n_reps=6)
                hp=selected_hiddens(hlm,clean,layers,batch_size=4)
                hz=selected_hiddens(hlm,[zs_prompt(x) for x,y in examples],layers,batch_size=4)
                zn={l:hp[l].reshape(4,6,-1).mean(1)-hz[l] for l in layers};torch.save(zn,nc)
            Z=Z.clone()
            for l in layers:Z[:,l]=(1-fraction)*Z[:,l]+fraction*zn[l]
        zl=z_list_from_Z(D,Z); z=np.mean(zl,axis=0)
        code=solver.solve(zl)
        cop,lys=ops_for(D,code,gamma=cfg['gamma'],delta_vec=z)
        zop,_=ops_for(D,code,gamma=cfg['gamma'],delta_vec=z,injection='additive')
        hp=[icl_prompt([e for i,e in enumerate(examples) if i!=j],x) for j,(x,y) in enumerate(examples)]
        H=hlm.last_token_hiddens(hp,batch_size=4)
        rop=[]
        for l in layers:
            vec=H[:,l].mean(0).to('cuda')
            def op(h,v=vec): return v.unsqueeze(0).expand_as(h).clone()
            rop.append(op)
        for method in methods:
            if (name,seed,method) in done: continue
            if (name,seed,method) in baseline_rows:
                row=baseline_rows[(name,seed,method)].copy()
                assert row['queries']==[list(q) for q in queries] and row['demos']==[list(e) for e in examples]
                row['reused_baseline_source']=baseline_path.name
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(name,seed,method,'reused',round(row['score'],4),flush=True)
                continue
            pp=[icl_prompt(examples,x) for x,y in queries] if method=='icl4' else prompts
            oo={'cass':cop,'z_only':zop,'replace':rop}.get(method)
            t=time.perf_counter()
            preds=generate(hlm,pp,oo,layers if oo else None,
                           schedule=cfg['schedule'] if method in ['cass','z_only'] else 'all',
                           batch_size=4,max_new_tokens=max_tokens,stop_after_answer=(HERE/'answer_stop_verified.json').exists())
            metrics=[score(name,p,y) for p,(x,y) in zip(preds,queries)]
            row=dict(task=name,family=task.family,seed=seed,method=method,score=float(np.mean(metrics)),
                     per_example_score=metrics,queries=queries,predictions=preds,demos=examples,
                     support=code.support,residual=code.residual,znorm=float(np.linalg.norm(z)),
                     seconds=time.perf_counter()-t,max_new_tokens=max_tokens,
                     variant=variant,variant_config=cfg,
                     answer_stop_verified=(HERE/'answer_stop_verified.json').exists(),
                     mean_input_tokens=float(np.mean([len(hlm.tok.encode(p)) for p in pp])))
            if name.startswith('long-'):
                pos=[]
                for p,(x,y) in zip(preds,queries):
                    a=clean_answer(p).split(); b=y.split()
                    pos.append(sum(u==v for u,v in zip(a,b))/len(b))
                row['position_accuracy']=float(np.mean(pos))
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(name,seed,method,round(row['score'],4),round(time.time()-start,1),flush=True)
(HERE/f"broader_{'signature_' if variant=='signature' else ''}{model}_done.json").write_text(json.dumps(dict(elapsed=time.time()-start)))
