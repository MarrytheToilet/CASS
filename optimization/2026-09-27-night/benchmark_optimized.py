"""Matched hardware benchmark, preserving original mathematical method."""
import json
import os
import sys
import time
import numpy as np
import torch
from common import HERE,ROOT,load_g,make_dict
sys.path.insert(0,str(ROOT/'rebuttal'/'2026-09-27'))
from benchmark_latency import PowerSampler
from fast_solver import GramSolver
from efficient_ops import extract_selected,selected_zlist,compiled_ops
from cass.models import HookedLM
from cass.extract import extract_fewshot_z
from cass.pipeline import code_for,ops_for,z_list_from_Z
from cass.tasks import load_task,zs_prompt,icl_prompt

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
hlm=HookedLM('llama31-8b');layers=[12,16]
G=load_g('llama31-8b',layers)
sampler=PowerSampler()
result=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),measurements=[],checks=[],offline=[],
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],demonstration_seed=0,
            scope='Warm model and reusable dictionary; natural greedy generation up to eight tokens; no shared-prefix cache')


def measure(task,label,fn,repeats=7,**meta):
    warm_start=time.perf_counter();value=fn();torch.cuda.synchronize()
    warm_time=time.perf_counter()-warm_start
    calibration_start=time.perf_counter();value=fn();torch.cuda.synchronize()
    calibration_time=time.perf_counter()-calibration_start
    # Group sub-second calls so 100ms board-power polling brackets enough samples.
    inner_repeats=max(1,min(20,int(np.ceil(.8/max(calibration_time,1e-6)))))
    rows=[]
    for _ in range(repeats):
        torch.cuda.synchronize();a=time.perf_counter()
        for _ in range(inner_repeats):value=fn()
        torch.cuda.synchronize();b=time.perf_counter()
        time.sleep(.15)
        energy=sampler.joules(a,b)
        rows.append(dict(seconds=(b-a)/inner_repeats,
                         gpu_board_joules=energy/inner_repeats if energy is not None else None,
                         grouped_elapsed_seconds=b-a))
    row=dict(task=task,measurement=label,warmup_seconds=warm_time,repeats=rows,
             calls_per_repeat=inner_repeats,energy_scope='GPU board only; 100ms polling; grouped calls, no CPU/system energy',**meta)
    row['median_seconds']=float(np.median([r['seconds'] for r in rows]))
    row['median_gpu_board_joules']=float(np.median([r['gpu_board_joules'] for r in rows if r['gpu_board_joules'] is not None]))
    if isinstance(value,list) and all(isinstance(v,str) for v in value):
        row['reencoded_output_tokens_excluding_specials']=[len(hlm.tok.encode(v,add_special_tokens=False)) for v in value]
    result['measurements'].append(row)
    (HERE/'optimized_latency.json').write_text(json.dumps(result,indent=2)+'\n')
    print(task,label,meta,round(row['median_seconds'],4),flush=True)
    return value


try:
    for name in ['antonym','country-capital','english-french']:
        task=load_task(name);rng=np.random.default_rng(4)
        ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
        D=make_dict(G,layers,exclude=name)
        t=time.perf_counter();solver=GramSolver(D)
        result['offline'].append(dict(task=name,gram_seconds=time.perf_counter()-t))
        Z=extract_fewshot_z(hlm,ex,seed=0,batch_size=8)
        zl=z_list_from_Z(D,Z);code=code_for(D,zl);fast=solver.solve(zl)
        np.testing.assert_allclose(code.delta,fast.delta,rtol=1e-8,atol=1e-8)
        assert code.support==fast.support
        selected8=extract_selected(hlm,ex,layers,seed=0,batch_size=8)
        selected48=extract_selected(hlm,ex,layers,seed=0,batch_size=48)
        check=dict(task=name,selected8_max_error=max(float((selected8[l]-Z[:,l]).abs().max()) for l in layers),
                   selected48_max_error=max(float((selected48[l]-Z[:,l]).abs().max()) for l in layers))
        torch.testing.assert_close(torch.stack([selected8[l] for l in layers]),Z[:,layers].transpose(0,1),atol=1e-6,rtol=1e-6)
        oo,ll=ops_for(D,code,delta_vec=np.mean(zl,axis=0))
        compile_start=time.perf_counter()
        co,_=compiled_ops(D,code,np.mean(zl,axis=0))
        probe=torch.randn(25,hlm.d,device='cuda')
        for a,b in zip(oo,co):torch.testing.assert_close(a(probe),b(probe),atol=2e-5,rtol=2e-5)
        torch.cuda.synchronize()
        result['offline'][-1]['compiled_probe_setup_seconds']=time.perf_counter()-compile_start
        pp=[zs_prompt(x) for x,y in task.eval_queries]
        before=hlm.generate(pp,batch_size=25,op=oo,layer=ll)
        compile_start=time.perf_counter()
        after=hlm.generate(pp,batch_size=25,op=co,layer=ll)
        result['offline'][-1]['first_compiled_generation_including_possible_specialization_seconds']=time.perf_counter()-compile_start
        check['compiled_generation_identical']=before==after
        check['compiled_different_queries']=sum(a!=b for a,b in zip(before,after))
        result['checks'].append(check)
        measure(name,'extract_original_b8',lambda:extract_fewshot_z(hlm,ex,seed=0,batch_size=8))
        measure(name,'extract_selected_b8',lambda:extract_selected(hlm,ex,layers,seed=0,batch_size=8))
        measure(name,'extract_selected_b48',lambda:extract_selected(hlm,ex,layers,seed=0,batch_size=48))
        def adapt_original():
            zz=z_list_from_Z(D,extract_fewshot_z(hlm,ex,seed=0,batch_size=8))
            return ops_for(D,code_for(D,zz),delta_vec=np.mean(zz,axis=0))
        def adapt_fast(bs):
            zz=selected_zlist(D,extract_selected(hlm,ex,layers,seed=0,batch_size=bs))
            return ops_for(D,solver.solve(zz),delta_vec=np.mean(zz,axis=0))
        measure(name,'adapt_original',adapt_original,repeats=5)
        measure(name,'adapt_optimized_b8',lambda:adapt_fast(8))
        fops,fl=measure(name,'adapt_optimized_b48',lambda:adapt_fast(48))
        batched_preds=hlm.generate(pp,batch_size=25,op=fops,layer=fl)
        check['batch48_generation_identical']=before==batched_preds
        check['batch48_different_queries']=sum(a!=b for a,b in zip(before,batched_preds))
        for bs in [1,8,25]:
            q0=pp[:bs];q4=[icl_prompt(ex,x) for x,y in task.eval_queries[:bs]]
            measure(name,'serve_original',lambda:hlm.generate(q0,batch_size=bs,op=oo,layer=ll),batch_size=bs)
            measure(name,'serve_compiled',lambda:hlm.generate(q0,batch_size=bs,op=co,layer=ll),batch_size=bs)
            measure(name,'serve_icl4',lambda:hlm.generate(q4,batch_size=bs),batch_size=bs)
    result['power_error']=sampler.error
    (HERE/'optimized_latency.json').write_text(json.dumps(result,indent=2)+'\n')
finally:sampler.close()
