"""Frozen native-template Llama settings on all seventeen broader probes."""
import json
import os
import random
import numpy as np
import torch
import llama_native as native
from common import HERE,generate
from broader_tasks import load_broader,score,clean_answer
from chat_probe import render
from efficient_ops import selected_hiddens
from cass.models import HookedLM
from cass.tasks import zs_prompt,icl_prompt,build_fewshot_pair_prompts


def signature(hlm,name,seed,examples):
    path=HERE/f'cache_native_broader_{name}_{seed}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    layers=native.shared.LAYERS
    positive,negative=build_fewshot_pair_prompts(examples,random.Random(9000+seed),n_reps=6)
    hp=selected_hiddens(hlm,[render(hlm,p) for p in positive+negative],layers,batch_size=4)
    hz=selected_hiddens(hlm,[render(hlm,zs_prompt(x)) for x,y in examples],layers,batch_size=4)
    data=dict(raw={l:(h[:24]-h[24:]).reshape(4,6,-1).mean(1) for l,h in hp.items()},
              null={l:h[:24].reshape(4,6,-1).mean(1)-hz[l] for l,h in hp.items()},
              positive={l:h[:24].reshape(4,6,-1).mean(1) for l,h in hp.items()})
    torch.save(data,path);return data


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    selected=native.freeze();hlm=HookedLM(native.MODEL)
    raw,null,clean=native.shared.mine(hlm,native.MODEL)
    original=dict(mode='context',null_fraction=0.,layers=[12,16],gamma=1.,correction=1.,kind='original',schedule='all')
    settings=[(c['null_fraction'],c['layers']) for c in [selected['best_correction'],original]]
    ds,ss=native.shared.dictionaries(raw,null,settings=settings)
    path=HERE/'broader_native.jsonl';done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);done.add((r['task'],r['seed'],r['method']))
    assert (HERE/'answer_stop_verified.json').exists()
    with path.open('a') as out:
        for name,task in load_broader().items():
            queries=task.eval_queries;limit=128 if name.startswith(('code-','long-')) else 32
            for seed in [20,21,22]:
                methods=['best_correction','matched_no_correction','dictionary_free','original','icl1','icl4']+(['zero'] if seed==20 else [])
                if all((name,seed,m) in done for m in methods):continue
                rng=np.random.default_rng(100*seed+4)
                ex=[task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool),4,replace=False)]
                data=signature(hlm,name,seed,ex);pre=native.shared.setup(ds,ss,data,clean)
                pp=[render(hlm,zs_prompt(x)) for x,y in queries]
                for method in methods:
                    if (name,seed,method) in done:continue
                    cfg={};ops=None;layers=None;prompts=pp;schedule='all';code=None
                    if method in ['best_correction','matched_no_correction','dictionary_free','original']:
                        cfg=original if method=='original' else selected['best_correction' if method=='matched_no_correction' else method]
                        ops,layers=native.ops(cfg,pre,data,method=='matched_no_correction');schedule=cfg['schedule']
                        if cfg['mode']=='context':code=pre[cfg['null_fraction'],tuple(cfg['layers'])][1]
                    elif method.startswith('icl'):
                        n=1 if method=='icl1' else 4
                        prompts=[render(hlm,icl_prompt(ex[:n],x)) for x,y in queries]
                    predictions=generate(hlm,prompts,ops,layers,schedule,batch_size=4,max_new_tokens=limit,stop_after_answer=True)
                    values=[score(name,p,y) for p,(x,y) in zip(predictions,queries)]
                    row=dict(task=name,seed=seed,method=method,family=task.family,score=float(np.mean(values)),
                             per_example_score=values,queries=queries,predictions=predictions,demos=ex,
                             variant_config=cfg,template='native',answer_stop_verified=True,max_new_tokens=limit,
                             development_source_sha256=selected['source_sha256'],
                             support=code.support if code else [],residual=code.residual if code else None,
                             mean_input_tokens=float(np.mean([len(hlm.tok.encode(p)) for p in prompts])))
                    if name.startswith('long-'):
                        row['position_accuracy']=float(np.mean([sum(a==b for a,b in zip(clean_answer(p).split(),y.split()))/len(y.split()) for p,(x,y) in zip(predictions,queries)]))
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();done.add((name,seed,method))
                    print(name,seed,method,round(row['score'],4),flush=True)
    assert len(done)==323
    (HERE/'broader_native_done.json').write_text('{"complete":true}\n')


if __name__=='__main__':main()
