"""Conditional paired effects for the complete native/instruction factorial."""
import json
import hashlib
import pandas as pd
from common import HERE,DEV_TASKS
from summarize_night import read_rows,paired
from cass.evaluate import accuracy

rows=read_rows('qwen_prompt_factorial');assert len(rows)==160
for row in rows:
    assert row['task'] not in DEV_TASKS
    assert abs(accuracy(row['predictions'],[y for x,y in row['queries']])-row['acc'])<1e-12
    row['layer_setting']='pair14_20' if row['layers']==[14,20] else 'layer24'
df=pd.DataFrame(rows);assert not df.duplicated(['task','native','instruction','layer_setting']).any()
result=dict(n_tasks=20,n_configurations=8,scope='Oracle intervention with matched remining; conditional paired task effects, not a unique architectural cause.',
            source_sha256=hashlib.sha256((HERE/'qwen_prompt_factorial.jsonl').read_bytes()).hexdigest(),
            means=df.groupby('config').acc.mean().to_dict(),effects=[])
for axis,levels in [('native',(True,False)),('instruction',(True,False)),('layer_setting',('layer24','pair14_20'))]:
    other=[c for c in ['native','instruction','layer_setting'] if c!=axis]
    for key,part in df.groupby(other):
        table=part.pivot(index='task',columns=axis,values='acc')
        result['effects'].append(dict(axis=axis,contrast=list(levels),fixed={c:(bool(v) if c!='layer_setting' else v) for c,v in zip(other,key)},
                                      **paired(table[levels[0]],table[levels[1]])))
(HERE/'qwen_prompt_factorial_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
