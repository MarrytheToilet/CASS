"""Mine generated skills; save only needed layers to keep the artifact compact."""
import os
import random
import time
import torch
from common import HERE
from scale_bank import bank_tasks,ordered_bank_names
from cass.models import HookedLM
from efficient_ops import selected_hiddens
from cass.tasks import build_pair_prompts,zs_prompt

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
model='llama31-8b';hlm=HookedLM(model)
out=HERE/'scale_activations';out.mkdir(exist_ok=True)
start=time.time();tasks=bank_tasks()
for name in ordered_bank_names():
    p=out/f'{name}.pt'
    if p.exists():continue
    task=tasks[name]
    pp,pc=build_pair_prompts(task.dict_pool,100,10,random.Random(7000))
    H=selected_hiddens(hlm,pp+pc,[12,16],batch_size=24)
    H0=selected_hiddens(hlm,[zs_prompt(task.dict_pool[i%len(task.dict_pool)][0]) for i in range(100)],[12,16],batch_size=25)
    torch.save(dict(G_by_layer={l:(h[:100]-h[100:]).half() for l,h in H.items()},
                    null_by_layer={l:(h[:100]-H0[l]).half() for l,h in H.items()},
                    clean_mean={l:h[:100].mean(0) for l,h in H.items()},
                    family=task.family,n_pairs=100,n_shots=10,seed=0),p)
    print(name,round(time.time()-start,1),flush=True)
(HERE/'scale_extraction_done.json').write_text('{"complete":true}\n')
