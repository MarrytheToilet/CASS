"""Four-example loss comparison; no evaluation label selects the serving tier."""
import hashlib
import json
import os
import time
import numpy as np
import torch
from common import HERE,Ledger
from adaptive_calibration import teacher_loss
from freeze_extensions import freeze
from cass.models import HookedLM
from cass.tasks import icl_prompt


def read(stem):return [json.loads(s) for s in (HERE/(stem+'.jsonl')).read_text().splitlines()]


def main():
    torch.set_num_threads(1);assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2','3']
    cfg=freeze()['combined']['config'];assert cfg['null_fraction']==1.
    source=HERE/'adaptive_combined_selection.jsonl';old=read('adaptive_combined_selection')
    assert len(old)==171 and len({(r['task'],r['seed']) for r in old})==171
    dest=HERE/'teacher_route_selections.jsonl'
    selected={(r['task'],r['seed']):r for r in read('teacher_route_selections')} if dest.exists() else {}
    hlm=HookedLM('llama31-8b')
    with dest.open('a') as stream:
        for r in old:
            key=r['task'],r['seed']
            if key in selected:continue
            assert r['frozen_config']==cfg
            candidates=r['candidates']
            indices=[i for i,c in enumerate(candidates) if c['mode']=='cass' and c['gamma']==cfg['gamma'] and c['schedule']==cfg['schedule']]
            assert len(indices)==1
            cass_loss=r['losses'][indices[0]];examples=r['examples'];assert len(examples)==4
            t=time.perf_counter();icl_loss=[]
            for j,(x,y) in enumerate(examples):
                context=[e for i,e in enumerate(examples) if i!=j]
                assert x not in {a for a,b in context}
                icl_loss.append(teacher_loss(hlm,icl_prompt(context,x),y,None,[],cfg['schedule']))
            row=dict(task=r['task'],seed=r['seed'],demos=examples,frozen_config=cfg,
                     cass_fold_losses=cass_loss,icl3_fold_losses=icl_loss,
                     uses_icl=bool(np.mean(icl_loss)<np.mean(cass_loss)),
                     incremental_icl_teacher_seconds=time.perf_counter()-t,
                     minimal_extraction_sequences=112,minimal_teacher_forwards=8,
                     reused_cass_loss_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            stream.write(json.dumps(row)+'\n');stream.flush();selected[key]=row
            print(r['task'],r['seed'],'ICL' if row['uses_icl'] else 'CASS',flush=True)
    assert len(selected)==171
    for split in ['confirm','fresh']:
        stem='extension_'+split;rows=read(stem);index={(r['task'],r['seed'],r['config']):r for r in rows}
        source_hash=hashlib.sha256((HERE/(stem+'.jsonl')).read_bytes()).hexdigest()
        ledger=Ledger('teacher_route_'+split)
        for key,decision in selected.items():
            name,seed=key;cass=index[name,seed,'combined'];off=index[name,seed,'combined_no_correction'];icl=index[name,seed,'icl4']
            assert cass['demos']==off['demos']==icl['demos']==decision['demos']
            assert cass['queries']==off['queries']==icl['queries']
            assert cass['config_parameters']==cfg and cass['znorm']>=5.125
            arms={'teacher_route':icl if decision['uses_icl'] else cass,
                  'teacher_route_off':icl if decision['uses_icl'] else off,
                  'residual_route':icl if cass['residual']>.7 else cass,
                  'frozen_cass':cass,'icl4':icl}
            for method,row in arms.items():
                if ledger.has(name,seed,method):continue
                uses_icl=(method=='icl4' or (method.startswith('teacher_route') and decision['uses_icl']) or
                          (method=='residual_route' and cass['residual']>.7))
                ledger.add(name,seed,method,cass['queries'],row['predictions'],suite=cass['suite'],demos=cass['demos'],
                    uses_icl=uses_icl,config_parameters=cfg,source_sha256=source_hash,reused_method=row['config'],
                    selection_rule='ICL3 mean fourfold NLL < fixed CASS mean fourfold NLL',
                    fresh_queries=split=='fresh')
        (HERE/('teacher_route_'+split+'_done.json')).write_text('{"complete":true}\n')
    (HERE/'teacher_routing_done.json').write_text('{"complete":true}\n')


if __name__=='__main__':main()
