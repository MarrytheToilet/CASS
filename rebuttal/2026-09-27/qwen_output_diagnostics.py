"""Descriptive output-format diagnosis, not an alternative semantic accuracy."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from cass.evaluate import _words,exact_match

values=defaultdict(list)
examples={}
for line in (HERE/'qwen_probe_generations.jsonl').read_text().splitlines():
    row=json.loads(line)
    p,t=_words(row['pred'],False),_words(row['target'],False)
    contains=any(p[i:i+len(t)]==t for i in range(max(0,len(p)-len(t)+1)))
    prefix=exact_match(row['pred'],row['target'])
    values[row['task'],row['method']].append((prefix,contains))
    if row['method']=='pair_last1' and row['task'] in ['person-sport','english-french','present-past']:
        if not prefix and row['task'] not in examples:
            examples[row['task']]=row
out=dict(scope='Diagnostic only: target-word occurrence anywhere in the first line can count negation or incidental mention; it is not semantic correctness and does not replace the original metric.',
         first_observed_failures=examples,per_task={})
for (task,method),pairs in values.items():
    prefix,contains=np.mean(pairs,axis=0)
    out['per_task'].setdefault(task,{})[method]=dict(n=len(pairs),original_prefix_score=float(prefix),target_occurrence=float(contains))
out['means']={}
for method in sorted({m for t,m in values}):
    a=[np.mean(pairs,axis=0) for (task,m),pairs in values.items() if m==method]
    prefix,contains=np.mean(a,axis=0)
    out['means'][method]=dict(original_prefix_score=float(prefix),target_occurrence=float(contains))
(HERE/'qwen_output_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['means'],indent=2))
