"""Same-engine cached serving for every predeclared broader NLP task."""
import hashlib
import json
import os
import random
import sys
import time
import numpy as np
import torch
from common import HERE,ROOT,ALL_TASKS,make_dict,generate
from broader_tasks import NLP_FILES,load_broader,score
from cass.models import HookedLM
from cass.tasks import zs_prompt,icl_prompt,build_fewshot_pair_prompts
from efficient_ops import selected_hiddens,selected_zlist
from fast_solver import GramSolver
from freeze_extensions import freeze
from variant_ops import OperatorFactory
from prefix_cache import PrefixICL
sys.path.insert(0,str(ROOT/'rebuttal/2026-09-27'))
from benchmark_latency import PowerSampler


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    assert (HERE/'answer_stop_verified.json').exists()
    cfg=freeze()['combined']['config'];assert cfg['null_fraction']==1.
    layers=[12,16];start=time.perf_counter();hlm=HookedLM('llama31-8b')
    model_seconds=time.perf_counter()-start;start=time.perf_counter();G={l:{} for l in layers};clean={}
    for name in ALL_TASKS:
        blob=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:G[l][name]=blob['G_by_layer'][l].float().numpy()
        clean[name]={l:blob['clean_mean'][l].numpy() for l in layers}
    D=make_dict(G,layers);solver=GramSolver(D);library_seconds=time.perf_counter()-start
    source=HERE/'broader_extensions.jsonl'
    saved={(r['task'],r['seed'],r['method']):r for r in map(json.loads,source.read_text().splitlines())}
    result=dict(configuration=cfg,model_load_seconds=model_seconds,library_load_build_seconds=library_seconds,
                gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                torch=torch.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                seed=20,max_new_tokens=32,stop_after_answer=True,measurements=[],checks=[],caches=[],
                scope='Same uncompiled PrefixICL engine, prefix copy/expansion included; first32-query workload per repeat; all100-query parity check at batch4.',
                energy_scope='GPU board only; 100ms sampling, CPU/system energy excluded')
    dest=HERE/'broader_cached_latency.json';sampler=PowerSampler()
    def save():dest.write_text(json.dumps(result,indent=2)+'\n')
    def measure(task,label,fn,**meta):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        warm=time.perf_counter()-start
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        estimate=time.perf_counter()-start
        inner=max(1,min(20,int(np.ceil(.8/max(estimate,1e-6)))));repeats=[]
        for _ in range(7):
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(inner):value=fn()
            torch.cuda.synchronize();end=time.perf_counter();time.sleep(.15)
            energy=sampler.joules(begin,end)
            repeats.append(dict(seconds=(end-begin)/inner,gpu_board_joules=None if energy is None else energy/inner))
        energies=[r['gpu_board_joules'] for r in repeats if r['gpu_board_joules'] is not None]
        row=dict(task=task,measurement=label,warmup_seconds=warm,calls_per_repeat=inner,repeats=repeats,
                 median_seconds=float(np.median([r['seconds'] for r in repeats])),
                 median_gpu_board_joules=float(np.median(energies)) if energies else None,**meta)
        if isinstance(value,list) and all(isinstance(s,str) for s in value):
            row['predictions']=value
            row['reencoded_output_tokens_excluding_specials']=[len(hlm.tok.encode(s,add_special_tokens=False)) for s in value]
            row['queries_per_second']=len(value)/row['median_seconds']
        result['measurements'].append(row);save();print(task,label,row['median_seconds'],flush=True)
        return value
    def prepare(data):
        zl=selected_zlist(D,data);code=solver.solve(zl)
        V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
        strength=float(np.clip(1-(V@V.T)[np.triu_indices(4,1)].mean(),0,1))
        factory=OperatorFactory(D,code,zl,clean);ops,lys=factory.ops(cfg['kind'],cfg['gamma'],strength)
        return ops,lys,code
    try:
        for name in NLP_FILES:
            task=load_broader()[name];queries=task.eval_queries
            reference=saved[name,20,'combined'];ex=reference['demos']
            assert reference['queries']==[list(q) for q in queries]
            data=torch.load(HERE/f'cache_broader_null_llama31-8b_{name}_20.pt',weights_only=True)
            ops,lys,code=prepare(data)
            cp,_=build_fewshot_pair_prompts(ex,random.Random(9020),n_reps=6)
            def adapt():
                positive=selected_hiddens(hlm,cp,layers,batch_size=4)
                zero=selected_hiddens(hlm,[zs_prompt(x) for x,y in ex],layers,batch_size=4)
                new={l:positive[l].reshape(4,6,-1).mean(1)-zero[l] for l in layers}
                return new,prepare(new)
            new,prepared=measure(name,'adaptation_28_sequences_7_calls',adapt,
                                  n_sequences=28,n_forward_calls=7,batch_size=4)
            result['checks'].append(dict(task=name,kind='signature_reextraction',
                max_abs_error={l:float((new[l]-data[l]).abs().max()) for l in layers},
                identical_support=prepared[2].support==code.support))
            p0=[zs_prompt(x) for x,y in queries];p4=[icl_prompt(ex,x) for x,y in queries]
            caches={'cass':measure(name,'build_cass_prefix',lambda:PrefixICL(hlm,[],p0,prefix_override='Q:')),
                    'icl4':measure(name,'build_icl4_prefix',lambda:PrefixICL(hlm,ex,p4))}
            operator_bytes=sum(t.numel()*t.element_size() for values in OperatorFactory(D,code,selected_zlist(D,data),clean).geometry(cfg['kind']) for t in values)
            result['caches'].append(dict(task=name,operator_tensor_bytes=operator_bytes,
                **{k:dict(tokens=c.length,kv_bytes=c.cache_bytes) for k,c in caches.items()}))
            for bs in [1,4,8]:
                for method,prompts,oo in [('cass',p0,ops),('icl4',p4,None)]:
                    cache=caches[method];ll=lys if oo else None
                    def run_cached(items):
                        output=[]
                        for i in range(0,len(items),bs):
                            output+=cache.generate(items[i:i+bs],max_new_tokens=32,ops=oo,layers=ll,
                                                   schedule=cfg['schedule'],stop_after_answer=True)
                        return output
                    n=100 if bs==4 else 32;pp=prompts[:n]
                    before=(saved[name,20,'combined' if method=='cass' else 'icl4']['predictions'] if bs==4 else
                            generate(hlm,pp,oo,ll,cfg['schedule'],batch_size=bs,max_new_tokens=32,stop_after_answer=True))
                    after=run_cached(pp)
                    truths=[y for x,y in queries[:n]]
                    before_scores=[score(name,p,y) for p,y in zip(before,truths)]
                    after_scores=[score(name,p,y) for p,y in zip(after,truths)]
                    result['checks'].append(dict(task=name,kind='cached_generation',method=method,batch_size=bs,n_queries=n,
                        before=before,after=after,different_queries=sum(x!=y for x,y in zip(before,after)),
                        before_score=float(np.mean(before_scores)),after_score=float(np.mean(after_scores)),
                        per_query_before=before_scores,per_query_after=after_scores))
                    measure(name,'serve_'+method,lambda:run_cached(prompts[:32]),batch_size=bs,n_queries=32)
            save()
        result['complete']=True;result['power_error']=sampler.error;save()
    finally:sampler.close()


if __name__=='__main__':main()
