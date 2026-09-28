"""Include reusable-prefix ICL as an explicit serving-cost comparator."""
import json
import os
import sys
import time
import numpy as np
import torch
from common import HERE,ROOT,generate
from prefix_cache import PrefixICL
from cass.models import HookedLM
from cass.tasks import load_task,icl_prompt
from cass.evaluate import accuracy
sys.path.insert(0,str(ROOT/'rebuttal'/'2026-09-27'))
from benchmark_latency import PowerSampler


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    hlm=HookedLM('llama31-8b');sampler=PowerSampler();result=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,checks=[],measurements=[],prefixes=[],
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],demonstration_seed=20,
        scope='Reusable unquantized demonstration-prefix KV cache; copying and batch expansion included in serving; greedy eight-token budget',
        energy_scope='Sampled GPU board energy only; 100ms polling; no CPU/system energy',
        implementation='Native greedy decode loop; original HF generate remains the uncached comparator. Numerical prediction differences are recorded explicitly.')
    def save():(HERE/'cached_icl_latency.json').write_text(json.dumps(result,indent=2)+'\n')
    def measure(name,label,fn,**meta):
        t=time.perf_counter();value=fn();torch.cuda.synchronize();warm=time.perf_counter()-t
        t=time.perf_counter();value=fn();torch.cuda.synchronize();seconds=time.perf_counter()-t
        inner=max(1,min(20,int(np.ceil(.8/max(seconds,1e-6)))));rows=[]
        for _ in range(7):
            torch.cuda.synchronize();a=time.perf_counter()
            for _ in range(inner):value=fn()
            torch.cuda.synchronize();b=time.perf_counter();time.sleep(.15);energy=sampler.joules(a,b)
            rows.append(dict(seconds=(b-a)/inner,gpu_board_joules=energy/inner if energy is not None else None))
        energies=[r['gpu_board_joules'] for r in rows if r['gpu_board_joules'] is not None]
        row=dict(task=name,measurement=label,calls_per_repeat=inner,warmup_seconds=warm,repeats=rows,
                 median_seconds=float(np.median([r['seconds'] for r in rows])),
                 median_gpu_board_joules=float(np.median(energies)) if energies else None,**meta)
        if isinstance(value,list) and all(isinstance(v,str) for v in value):
            row['reencoded_output_tokens_excluding_specials']=[len(hlm.tok.encode(v,add_special_tokens=False)) for v in value]
        result['measurements'].append(row);save();print(name,label,round(row['median_seconds'],4),flush=True)
        return value
    try:
        for name in ['antonym','country-capital','english-french']:
            task=load_task(name);rng=np.random.default_rng(2004);ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
            pp=[icl_prompt(ex,x) for x,y in task.eval_queries]
            prefix=measure(name,'build_prefix_cache',lambda:PrefixICL(hlm,ex,pp))
            result['prefixes'].append(dict(task=name,cache_bytes=prefix.cache_bytes,prefix_tokens=prefix.length))
            for bs in [1,8,25]:
                before=generate(hlm,pp,batch_size=bs);after=[]
                for i in range(0,len(pp),bs):after+=prefix.generate(pp[i:i+bs])
                result['checks'].append(dict(task=name,batch_size=bs,identical=before==after,
                    different_queries=sum(a!=b for a,b in zip(before,after)),before=before,after=after,
                    original_accuracy=accuracy(before,[y for x,y in task.eval_queries]),cached_accuracy=accuracy(after,[y for x,y in task.eval_queries])))
                q=pp[:bs]
                measure(name,'serve_icl4_uncached',lambda:generate(hlm,q,batch_size=bs),batch_size=bs)
                measure(name,'serve_icl4_prefix_cached',lambda:prefix.generate(q),batch_size=bs)
            save()
        result['power_error']=sampler.error;result['complete']=True;save()
    finally:sampler.close()

if __name__=='__main__':main()
