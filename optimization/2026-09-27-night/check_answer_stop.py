"""Validate the semantic stop rule against saved outputs and matched generation."""
import json
import os
import time
import numpy as np
import torch
from common import HERE,generate
from answer_stop import answer_complete
from broader_tasks import clean_answer,score,load_broader
from cass.models import HookedLM
from cass.tasks import icl_prompt,zs_prompt

torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
checked=0
for path in HERE.glob('broader_*.jsonl'):
    for line in path.read_text().splitlines():
        r=json.loads(line)
        for full,(_,target) in zip(r['predictions'],r['queries']):
            for k in range(1,len(full)+1):
                if answer_complete(full[:k]):
                    assert clean_answer(full[:k]).strip()==clean_answer(full).strip(),(r['task'],full[:k],full)
                    assert score(r['task'],full[:k],target)==score(r['task'],full,target)
                    checked+=1;break
hlm=HookedLM('llama31-8b');rows=[];tasks=load_broader()
for name in ['long-title','long-reverse-words','code-sum','code-sort3-max']:
    task=tasks[name];queries=task.eval_queries[:8];ex=task.fewshot_pool[:4]
    for shots in [0,1,4]:
        pp=[icl_prompt(ex[:shots],x) if shots else zs_prompt(x) for x,y in queries]
        t=time.perf_counter();full=generate(hlm,pp,batch_size=4,max_new_tokens=128);full_s=time.perf_counter()-t
        t=time.perf_counter();early=generate(hlm,pp,batch_size=4,max_new_tokens=128,stop_after_answer=True);early_s=time.perf_counter()-t
        assert [clean_answer(p).strip() for p in full]==[clean_answer(p).strip() for p in early],(name,shots)
        assert [score(name,p,y) for p,(x,y) in zip(full,queries)]==[score(name,p,y) for p,(x,y) in zip(early,queries)]
        rows.append(dict(task=name,shots=shots,full_seconds=full_s,early_seconds=early_s,full=full,early=early))
        print(name,shots,full_s,early_s,flush=True)
(HERE/'answer_stop_verified.json').write_text(json.dumps(dict(saved_outputs_checked=checked,matched_generations=rows),indent=2)+'\n')
print('Answer-stopping verification passed',checked,flush=True)
