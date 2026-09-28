"""Resolve end-to-end numerical parity of the primary batch8 optimization."""
import json
import os
import numpy as np
import torch
from common import HERE,load_g,make_dict,generate
from fast_solver import GramSolver
from efficient_ops import extract_selected,selected_zlist
from cass.models import HookedLM
from cass.extract import extract_fewshot_z
from cass.pipeline import code_for,ops_for,z_list_from_Z
from cass.tasks import load_task,zs_prompt
from cass.evaluate import accuracy


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    hlm=HookedLM('llama31-8b');layers=[12,16];G=load_g('llama31-8b',layers);records=[]
    for name in ['antonym','country-capital','english-french']:
        task=load_task(name);rng=np.random.default_rng(4)
        demos=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
        D=make_dict(G,layers,exclude=name)
        oldZ=extract_fewshot_z(hlm,demos,seed=0,batch_size=8)
        newZ=extract_selected(hlm,demos,layers,seed=0,batch_size=8)
        for l in layers:assert torch.equal(oldZ[:,l],newZ[l])
        oldz=z_list_from_Z(D,oldZ);newz=selected_zlist(D,newZ)
        assert all(np.array_equal(a,b) for a,b in zip(oldz,newz))
        oldcode=code_for(D,oldz);newcode=GramSolver(D).solve(newz)
        assert oldcode.support==newcode.support
        oldops,lys=ops_for(D,oldcode,delta_vec=np.mean(oldz,axis=0))
        newops,_=ops_for(D,newcode,delta_vec=np.mean(newz,axis=0))
        prompts=[zs_prompt(x) for x,y in task.eval_queries];targets=[y for x,y in task.eval_queries]
        for batch in [1,8,25]:
            before=generate(hlm,prompts,oldops,lys,batch_size=batch)
            after=generate(hlm,prompts,newops,lys,batch_size=batch)
            row=dict(task=name,extraction_batch_size=8,serving_batch_size=batch,identical=before==after,
                     different_queries=sum(a!=b for a,b in zip(before,after)),queries=task.eval_queries,
                     before=before,after=after,original_accuracy=accuracy(before,targets),optimized_accuracy=accuracy(after,targets),
                     activations_and_signatures_identical=True,support_identical=True,
                     reconstruction_max_error=float(np.abs(oldcode.delta-newcode.delta).max()))
            records.append(row);print(name,batch,row['identical'],flush=True)
    result=dict(complete=True,checks=records,scope='Uncompiled original operator, identical extraction batches; only selective-layer capture and cached Gram solver differ.')
    (HERE/'optimized_generation_checks.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
