"""Same cached greedy engine for contextual CASS and four-shot ICL."""
import json
import argparse
import os
import sys
import time
import numpy as np
import torch
from common import HERE,ROOT,ALL_TASKS,load_g,make_dict,generate
from freeze_extensions import freeze
from efficient_ops import selected_zlist
from fast_solver import GramSolver
from variant_ops import OperatorFactory
from benchmark_variant import linear_ops
from prefix_cache import PrefixICL
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt,icl_prompt
from cass.evaluate import accuracy
sys.path.insert(0,str(ROOT/'rebuttal/2026-09-27'))
from benchmark_latency import PowerSampler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--answer-stop',action='store_true');args=parser.parse_args()
    stem='answer_engine_latency' if args.answer_stop else 'matched_engine_latency'
    if args.answer_stop:assert (HERE/'answer_stop_verified.json').exists()
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['combined']['config'];a=cfg['null_fraction'];layers=[12,16]
    hlm=HookedLM('llama31-8b');G=load_g('llama31-8b',layers);clean={}
    for name in ALL_TASKS:
        blob=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:G[l][name]=(1-a)*G[l][name]+a*blob['G_by_layer'][l].float().numpy()
        clean[name]={l:blob['clean_mean'][l].numpy() for l in layers}
    result=dict(configuration=cfg,checks=[],measurements=[],caches=[],gpu=torch.cuda.get_device_name(0),
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,demonstration_seed=20,
                scope='Same PrefixICL greedy engine, unquantized prefix KV cache, copy and batch expansion included; '
                      'eight-token natural generation. CASS caches Q: prefix; ICL caches four demonstrations.',
                stop_after_answer=args.answer_stop,
                adaptation_source='variant_latency.json: same frozen configuration, demonstrations, task-excluded dictionary and hardware',
                energy_scope='GPU board only; 100ms power polling; CPU/system energy excluded')
    sampler=PowerSampler()
    def save():(HERE/(stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    def measure(name,label,fn,**meta):
        t=time.perf_counter();value=fn();torch.cuda.synchronize();warm=time.perf_counter()-t
        t=time.perf_counter();value=fn();torch.cuda.synchronize();estimate=time.perf_counter()-t
        inner=max(1,min(20,int(np.ceil(.8/max(estimate,1e-6)))));rows=[]
        for _ in range(7):
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(inner):value=fn()
            torch.cuda.synchronize();end=time.perf_counter();time.sleep(.15)
            joules=sampler.joules(begin,end)
            rows.append(dict(seconds=(end-begin)/inner,gpu_board_joules=joules/inner if joules is not None else None))
        energies=[r['gpu_board_joules'] for r in rows if r['gpu_board_joules'] is not None]
        row=dict(task=name,measurement=label,warmup_seconds=warm,calls_per_repeat=inner,repeats=rows,
                 median_seconds=float(np.median([r['seconds'] for r in rows])),
                 median_gpu_board_joules=float(np.median(energies)) if energies else None,**meta)
        if isinstance(value,list) and all(isinstance(v,str) for v in value):
            row['reencoded_output_tokens_excluding_specials']=[len(hlm.tok.encode(v,add_special_tokens=False)) for v in value]
        result['measurements'].append(row);save();print(name,label,round(row['median_seconds'],4),flush=True)
        return value
    try:
        for name in ['antonym','country-capital','english-french']:
            D=make_dict(G,layers,exclude=name);solver=GramSolver(D);task=load_task(name)
            rng=np.random.default_rng(2004);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
            data=torch.load(HERE/f'cache_signature_confirm_{name}_20.pt',weights_only=True)
            zl=selected_zlist(D,{l:(1-a)*data['raw'][l]+a*data['null'][l] for l in layers});code=solver.solve(zl)
            V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
            strength={'uncertainty':float(np.clip(1-(V@V.T)[np.triu_indices(4,1)].mean(),0,1)),
                      'coverage':max(0.,1-code.residual**2)}.get(cfg['correction'],cfg['correction'])
            factory=OperatorFactory(D,code,zl,clean);original,lys=factory.ops(cfg['kind'],cfg['gamma'],strength)
            compiled,_=linear_ops(factory,cfg,strength)
            p0=[zs_prompt(x) for x,y in task.eval_queries];p4=[icl_prompt(ex,x) for x,y in task.eval_queries]
            prefixes={
                'cass':measure(name,'build_cass_prefix',lambda:PrefixICL(hlm,[],p0,prefix_override='Q:')),
                'icl4':measure(name,'build_icl4_prefix',lambda:PrefixICL(hlm,ex,p4))}
            result['caches'].append(dict(task=name,**{k:dict(tokens=p.length,kv_bytes=p.cache_bytes) for k,p in prefixes.items()},
                compiled_operator_tensor_bytes=sum(t.numel()*t.element_size() for op in compiled for t in op.__defaults__)))
            for bs in [1,8,25]:
                for method,prompts,oo in [('cass',p0,compiled),('cass_uncompiled',p0,original),('icl4',p4,None)]:
                    cache=prefixes['cass' if method.startswith('cass') else 'icl4']
                    before=generate(hlm,prompts,original if method.startswith('cass') else None,
                                    lys if method.startswith('cass') else None,cfg['schedule'],batch_size=bs,
                                    stop_after_answer=args.answer_stop)
                    after=[]
                    for i in range(0,len(prompts),bs):
                        after+=cache.generate(prompts[i:i+bs],ops=oo,layers=lys if oo else None,schedule=cfg['schedule'],stop_after_answer=args.answer_stop)
                    result['checks'].append(dict(task=name,method=method,batch_size=bs,identical=before==after,
                        different_queries=sum(x!=y for x,y in zip(before,after)),before=before,after=after,
                        original_accuracy=accuracy(before,[y for x,y in task.eval_queries]),
                        cached_accuracy=accuracy(after,[y for x,y in task.eval_queries])))
                    q=prompts[:bs]
                    measure(name,'serve_'+method+'_matched_engine',
                            lambda:cache.generate(q,ops=oo,layers=lys if oo else None,schedule=cfg['schedule'],stop_after_answer=args.answer_stop),batch_size=bs)
            save()
        result['complete']=True;result['power_error']=sampler.error;save()
    finally:sampler.close()


if __name__=='__main__':main()
