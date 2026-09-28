"""Paired effects of prompt format, intervention depth, gain, and positions."""
import json
import hashlib
import numpy as np
import pandas as pd
from common import HERE,DEV_TASKS
from summarize_night import read_rows,paired
from cass.evaluate import accuracy

rows=read_rows('qwen_factorial')
assert len(rows)==20*2*2*2*2,'Wait for complete factorial'
for r in rows:
    assert r['task'] not in DEV_TASKS
    assert abs(accuracy(r['predictions'],[y for x,y in r['queries']])-r['acc'])<1e-12
    r['layer_setting']='pair14_20' if r['layers']==[14,20] else 'layer24'
df=pd.DataFrame(rows)
assert not df.duplicated(['task','prompt','layer_setting','gain','positions']).any()
table=df.pivot(index='task',columns='config',values='acc')
result=dict(n_tasks=20,scope='Known-task oracle intervention on tasks excluded from the development target set; '
                            '20 paired tasks, one deterministic oracle per configuration, no few-shot support inference.',
            source_sha256=hashlib.sha256((HERE/'qwen_factorial.jsonl').read_bytes()).hexdigest(),
            configuration_means=table.mean().to_dict(),effects=[])
for axis,levels in [('prompt',('chat','plain')),('layer_setting',('layer24','pair14_20')),
                    ('gain',(2.,1.)),('positions',(4,1))]:
    other=[c for c in ['prompt','layer_setting','gain','positions'] if c!=axis]
    for key,part in df.groupby(other):
        wide=part.pivot(index='task',columns=axis,values='acc')
        fixed={name:(value.item() if isinstance(value,np.generic) else value) for name,value in zip(other,key)}
        result['effects'].append(dict(axis=axis,contrast=list(levels),fixed=fixed,
                                      **paired(wide[levels[0]],wide[levels[1]])))
(HERE/'qwen_factorial_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
