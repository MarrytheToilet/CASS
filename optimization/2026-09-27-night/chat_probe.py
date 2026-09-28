"""Qwen native-chat extraction and injection; development split only."""
import json
import os
import random
import time
import numpy as np
import torch
from common import HERE,DEV_TASKS,dev_split,make_dict,generate,Ledger
from cass.models import HookedLM
from cass.tasks import ALL_TASKS,load_task,zs_prompt,icl_prompt,build_pair_prompts
from cass.steer import make_affine_op,make_additive_op

SYSTEM='Infer the input-output rule from the examples when provided. Return only the output, without explanation.'


def render(hlm,text):
    rendered=hlm.tok.apply_chat_template([{'role':'system','content':SYSTEM},
                                        {'role':'user','content':text}],
                                       tokenize=False,add_generation_prompt=True,
                                       enable_thinking=False)
    # The shared extraction/generation helpers tokenize plain strings with the
    # tokenizer's default post-processor. Llama's post-processor adds a BOS even
    # when the Python add_bos_token attribute is false. Remove the template BOS
    # here only if that post-processor will reinsert it. Qwen has no BOS and is
    # unchanged. The resulting input IDs equal template IDs tokenized without
    # additional special tokens (see check_native_tokenization.py).
    bos=hlm.tok.bos_token
    if bos and rendered.startswith(bos):
        probe=hlm.tok.encode('x')
        if probe and probe[0]==hlm.tok.bos_token_id:rendered=rendered[len(bos):]
    return rendered


def main():
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    hlm=HookedLM('qwen3-4b')
    folder=HERE/'qwen_chat_activations';folder.mkdir(exist_ok=True)
    layers=[8,12,16,20,24,28,32]
    start=time.time();G={l:{} for l in layers}
    for name in ALL_TASKS:
        path=folder/f'{name}.pt'
        if path.exists(): blob=torch.load(path,weights_only=True)
        else:
            task=load_task(name)
            clean,corr=build_pair_prompts(task.dict_pool,100,10,random.Random(7000))
            pos=hlm.last_token_hiddens([render(hlm,p) for p in clean],batch_size=8)
            neg=hlm.last_token_hiddens([render(hlm,p) for p in corr],batch_size=8)
            blob={'G':(pos-neg).half(),'n_pairs':100,'n_shots':10,'prompt':SYSTEM}
            torch.save(blob,path)
        for l in layers:G[l][name]=blob['G'][:,l].float().numpy()
        print('extracted',name,round(time.time()-start,1),flush=True)
    D={l:make_dict(G,[l]).per_layer[l] for l in layers}
    ledger=Ledger('qwen_chat_oracle_dev')
    configs=[]
    for l in layers:
        for kind in ['clean_add','affine']:
            for gain in [.5,1.,2.,4.]:
                for schedule in ['all','prefill']:
                    configs.append(dict(layers=[l],kind=kind,gain=gain,schedule=schedule))
    for pair in [[16,20],[20,24],[24,28],[12,24],[16,24]]:
        for gain in [.5,1.,2.]:
            configs.append(dict(layers=pair,kind='affine',gain=gain,schedule='prefill'))
    (HERE/'qwen_chat_dev_grid.json').write_text(json.dumps(configs,indent=2))
    for name in DEV_TASKS:
        demos,queries=dev_split(load_task(name),10)
        prompts=[render(hlm,zs_prompt(x)) for x,y in queries]
        for mode in ['zero','icl4']:
            if ledger.has(name,10,mode):continue
            pp=prompts if mode=='zero' else [render(hlm,icl_prompt(demos,x)) for x,y in queries]
            ledger.add(name,10,mode,queries,generate(hlm,pp),demos=demos)
        for cfg in configs:
            label=json.dumps(cfg,sort_keys=True,separators=(',',':'))
            if ledger.has(name,10,label):continue
            ops=[]
            for l in cfg['layers']:
                dl=D[l]
                if cfg['kind']=='affine':
                    op=make_affine_op(dl.anchors[name],dl.bases[name],dl.anchors[name],
                                      gamma=cfg['gain'],beta=2.,alpha_max=1.)
                else:op=make_additive_op(dl.anchors[name],gamma=cfg['gain'])
                ops.append(op)
            ledger.add(name,10,label,queries,generate(hlm,prompts,ops,cfg['layers'],cfg['schedule']),
                       scope='known-task oracle, native-chat development split')
        print('evaluated',name,round(time.time()-start,1),flush=True)
    (HERE/'qwen_chat_dev_done.json').write_text(json.dumps(dict(elapsed=time.time()-start)))


if __name__=='__main__':main()
