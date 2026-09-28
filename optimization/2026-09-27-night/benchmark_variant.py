"""Actual adaptation/serving cost of the development-selected combined variant."""
import json
import os
import random
import sys
import time
import numpy as np
import torch
from common import HERE,ROOT,ALL_TASKS,load_g,make_dict,generate
from freeze_extensions import freeze
from variant_ops import OperatorFactory
from efficient_ops import selected_hiddens,selected_zlist
from fast_solver import GramSolver
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt,icl_prompt,build_fewshot_pair_prompts
sys.path.insert(0,str(ROOT/'rebuttal'/'2026-09-27'))
from benchmark_latency import PowerSampler


def affine_kernel(h,Q,beta,c):
    h=h.float()
    return h+beta-c*((h@Q)@Q.T)

compiled_affine=torch.compile(affine_kernel,fullgraph=True,dynamic=True)


def linear_ops(factory,cfg,strength):
    assert cfg['kind']=='context_residual'
    c=strength*factory.gate;ops=[]
    for q,d,mu in factory.geometry(cfg['kind']):
        beta=cfg['gamma']*d+c*(((mu-cfg['gamma']*d)@q)@q.T)
        scalar=torch.tensor(c,device='cuda',dtype=torch.float32)
        def op(h,qq=q,b=beta,cc=scalar):return compiled_affine(h.contiguous(),qq,b,cc)
        ops.append(op)
    return ops,list(factory.D.layers)


def extract_minimal(hlm,ex,layers,fraction,seed=20):
    clean,corr=build_fewshot_pair_prompts(ex,random.Random(9000+seed),n_reps=6)
    # Preserve the exact confirmation batch partitions. Combining all28 null
    # sequences into one padded batch can change BF16 kernels and greedy outputs.
    H=selected_hiddens(hlm,clean,layers,batch_size=24)
    negative=selected_hiddens(hlm,corr,layers,batch_size=24) if fraction<1 else None
    zero=selected_hiddens(hlm,[zs_prompt(x) for x,y in ex],layers,batch_size=4) if fraction>0 else None
    data={}
    for l,h in H.items():
        positive=h.reshape(4,6,-1).mean(1)
        raw=(h-negative[l]).reshape(4,6,-1).mean(1) if fraction<1 else 0.
        null=positive-zero[l] if fraction>0 else 0.
        data[l]=(1-fraction)*raw+fraction*null
    return data,24+24*int(fraction<1)+4*int(fraction>0)


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['combined']['config'];fraction=cfg['null_fraction'];layers=[12,16]
    result=dict(configuration=cfg,measurements=[],checks=[],offline=[],torch=torch.__version__,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],demonstration_seed=20,
                scope='Warm model/dictionary; natural greedy output up to eight tokens; no shared-prefix KV cache',
                energy_scope='GPU board only, 100ms polling and trapezoidal integration, including idle board draw; excludes CPU/system energy')
    t=time.perf_counter();hlm=HookedLM('llama31-8b');result['model_load_seconds']=time.perf_counter()-t
    result['gpu']=torch.cuda.get_device_name(0)
    G=load_g('llama31-8b',layers);clean={}
    for name in ALL_TASKS:
        b=torch.load(HERE/'llama_null_activations'/f'{name}.pt',weights_only=True)
        for l in layers:G[l][name]=(1-fraction)*G[l][name]+fraction*b['G_by_layer'][l].float().numpy()
        clean[name]={l:b['clean_mean'][l].numpy() for l in layers}
    sampler=PowerSampler()
    def save():(HERE/'variant_latency.json').write_text(json.dumps(result,indent=2)+'\n')
    def measure(name,label,fn,**meta):
        t=time.perf_counter();value=fn();torch.cuda.synchronize();warm=time.perf_counter()-t
        t=time.perf_counter();value=fn();torch.cuda.synchronize();calib=time.perf_counter()-t
        inner=max(1,min(20,int(np.ceil(.8/max(calib,1e-6)))));repeats=[]
        for _ in range(7):
            torch.cuda.synchronize();a=time.perf_counter()
            for _ in range(inner):value=fn()
            torch.cuda.synchronize();b=time.perf_counter();time.sleep(.15)
            energy=sampler.joules(a,b)
            repeats.append(dict(seconds=(b-a)/inner,gpu_board_joules=energy/inner if energy is not None else None))
        energies=[r['gpu_board_joules'] for r in repeats if r['gpu_board_joules'] is not None]
        row=dict(task=name,measurement=label,warmup_seconds=warm,calls_per_repeat=inner,repeats=repeats,
                 median_seconds=float(np.median([r['seconds'] for r in repeats])),
                 median_gpu_board_joules=float(np.median(energies)) if energies else None,**meta)
        if isinstance(value,list) and all(isinstance(v,str) for v in value):
            row['reencoded_output_tokens_excluding_specials']=[len(hlm.tok.encode(v,add_special_tokens=False)) for v in value]
        result['measurements'].append(row);save();print(name,label,round(row['median_seconds'],4),flush=True)
        return value
    try:
        for name in ['antonym','country-capital','english-french']:
            t=time.perf_counter();D=make_dict(G,layers,exclude=name);build=time.perf_counter()-t
            t=time.perf_counter();solver=GramSolver(D);gram=time.perf_counter()-t
            result['offline'].append(dict(task=name,dictionary_build_seconds=build,gram_seconds=gram))
            task=load_task(name);rng=np.random.default_rng(2004);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
            def adapt(compiled=False):
                Z,count=extract_minimal(hlm,ex,layers,fraction)
                zl=selected_zlist(D,Z);code=solver.solve(zl);factory=OperatorFactory(D,code,zl,clean)
                V=np.asarray(zl);V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
                strength={'uncertainty':float(np.clip(1-(V@V.T)[np.triu_indices(4,1)].mean(),0,1)),
                          'coverage':max(0.,1-code.residual**2)}.get(cfg['correction'],cfg['correction'])
                # Include ready-to-use operator construction in online adaptation.
                if compiled:linear_ops(factory,cfg,strength)
                else:factory.ops(cfg['kind'],cfg['gamma'],strength)
                return factory,strength,count
            factory,strength,count=adapt();oo,lys=factory.ops(cfg['kind'],cfg['gamma'],strength)
            t=time.perf_counter();co,_=linear_ops(factory,cfg,strength)
            probe=torch.randn(25,D.d,device='cuda')
            for original,fast in zip(oo,co):torch.testing.assert_close(original(probe),fast(probe),atol=3e-5,rtol=3e-5)
            torch.cuda.synchronize();result['offline'][-1]['compiled_setup_seconds']=time.perf_counter()-t
            pp=[zs_prompt(x) for x,y in task.eval_queries]
            before=generate(hlm,pp,oo,lys,cfg['schedule']);after=generate(hlm,pp,co,lys,cfg['schedule'])
            check=dict(task=name,compiled_identical=before==after,different_generations=sum(a!=b for a,b in zip(before,after)))
            reference=[json.loads(line) for line in (HERE/'extension_confirm.jsonl').read_text().splitlines() if line.strip()]
            reference=next(r for r in reference if r['task']==name and r['seed']==20 and r['config']=='combined')
            check['minimal_extraction_generations_identical_to_confirmation']=before==reference['predictions']
            check['minimal_extraction_different_queries']=sum(a!=b for a,b in zip(before,reference['predictions']))
            result['checks'].append(check)
            measure(name,'extract_variant',lambda:extract_minimal(hlm,ex,layers,fraction),extraction_sequences=count,
                    model_calls=1+int(fraction<1)+int(fraction>0),batch_partition=[24]+([24] if fraction<1 else [])+([4] if fraction>0 else []))
            measure(name,'adapt_variant',adapt,extraction_sequences=count)
            measure(name,'adapt_variant_compiled',lambda:adapt(True),extraction_sequences=count)
            for bs in [1,8,25]:
                p0=pp[:bs];p4=[icl_prompt(ex,x) for x,y in task.eval_queries[:bs]]
                measure(name,'serve_variant',lambda:generate(hlm,p0,oo,lys,cfg['schedule'],batch_size=bs),batch_size=bs)
                measure(name,'serve_variant_compiled',lambda:generate(hlm,p0,co,lys,cfg['schedule'],batch_size=bs),batch_size=bs)
                measure(name,'serve_icl4',lambda:generate(hlm,p4,batch_size=bs),batch_size=bs)
                result['measurements'][-1]['mean_input_tokens']=float(np.mean([len(hlm.tok.encode(p)) for p in p4]))
                result['measurements'][-2]['mean_input_tokens']=float(np.mean([len(hlm.tok.encode(p)) for p in p0]))
                save()
        result['power_error']=sampler.error;result['complete']=True;save()
    finally:sampler.close()

if __name__=='__main__':main()
