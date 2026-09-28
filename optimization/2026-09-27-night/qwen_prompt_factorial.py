"""Separate native format from a task-agnostic instruction in oracle steering."""
import json
import os
import random
import torch
from common import HERE,ALL_TASKS,DEV_TASKS,make_dict,generate,Ledger
from chat_probe import SYSTEM,render
from efficient_ops import selected_hiddens
from cass.models import HookedLM
from cass.tasks import load_task,zs_prompt,build_pair_prompts
from cass.pipeline import oracle_ops


def format_prompt(hlm,text,native,instruction):
    if not native:return (SYSTEM+'\n\n' if instruction else '')+text
    if instruction:return render(hlm,text)
    return hlm.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                      add_generation_prompt=True,enable_thinking=False)


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    hlm=HookedLM('qwen3-4b');layers=[14,20,24];all_g={}
    assert hlm.tok.bos_token is None
    # Remine all four protocols with identical batch boundaries and hardware.
    # The instruction contains no task name, target answer or task-specific rule.
    for native in [False,True]:
        for instruction in [False,True]:
            tag=f'native{int(native)}_instruction{int(instruction)}'
            directory=HERE/('qwen_prompt_factorial_'+tag);directory.mkdir(exist_ok=True)
            G={l:{} for l in layers}
            for name in ALL_TASKS:
                path=directory/(name+'.pt')
                if path.exists():blob=torch.load(path,weights_only=True)
                else:
                    positive,negative=build_pair_prompts(load_task(name).dict_pool,100,10,random.Random(7000))
                    hp=selected_hiddens(hlm,[format_prompt(hlm,p,native,instruction) for p in positive],layers,batch_size=8)
                    hn=selected_hiddens(hlm,[format_prompt(hlm,p,native,instruction) for p in negative],layers,batch_size=8)
                    blob=dict(G={l:(hp[l]-hn[l]).half() for l in layers},native=native,instruction=instruction,
                              prompt_rng_seed=7000,n_pairs=100,n_shots=10,batch_size=8)
                    torch.save(blob,path)
                for l in layers:G[l][name]=blob['G'][l].float().numpy()
            all_g[(native,instruction)]=G;print('mining complete',tag,flush=True)
    dictionaries={(native,instruction,tuple(lys)):make_dict(G,lys)
                  for (native,instruction),G in all_g.items() for lys in [[14,20],[24]]}
    ledger=Ledger('qwen_prompt_factorial')
    for name in [n for n in ALL_TASKS if n not in DEV_TASKS]:
        task=load_task(name);queries=task.eval_queries
        for (native,instruction,lys),D in dictionaries.items():
            label=f'native{int(native)}_instruction{int(instruction)}_layers{lys}'
            if ledger.has(name,0,label):continue
            prompts=[format_prompt(hlm,zs_prompt(x),native,instruction) for x,y in queries]
            ops,layers=oracle_ops(D,name,gamma=2.)
            predictions=generate(hlm,prompts,ops,layers,positions=1,schedule='all')
            value=ledger.add(name,0,label,queries,predictions,native=native,instruction=instruction,layers=list(lys),
                gain=2.,positions=1,scope='Known-task oracle diagnostic; all four prompt protocols remined with matched extraction batches. No few-shot support inference.')
            print(name,label,round(value,4),flush=True)
    (HERE/'qwen_prompt_factorial_done.json').write_text('{"complete":true}\n')


if __name__=='__main__':main()
