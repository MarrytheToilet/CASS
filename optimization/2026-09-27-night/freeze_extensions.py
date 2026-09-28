"""Select extension configurations strictly from the prespecified development split."""
import hashlib
import json
from collections import defaultdict
import numpy as np
from common import HERE


def select(stem, predicate=lambda c: True, expected=None):
    path=HERE/(stem+'.jsonl'); rows=defaultdict(list)
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['config'].startswith('{'):rows[r['config']].append(r['acc'])
    if expected is not None:assert len(rows)==expected,(stem,len(rows))
    assert all(len(v)==24 for v in rows.values())
    eligible={k:float(np.mean(v)) for k,v in rows.items() if predicate(json.loads(k))}
    winner=max(eligible,key=eligible.get)
    return dict(config=json.loads(winner),development_accuracy=eligible[winner],
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_development_means=eligible)


def freeze():
    result=dict(combined=select('combined_dev',expected=120),
                oneshot=select('llama_variants_dev',lambda c:c['shots']==1 and c['correction']>0,96),
                dictionary_free=select('dictionary_free_dev',expected=360))
    result['selection']='Twelve fixed development tasks, query split separate from all eval queries; two seeds.'
    path=HERE/'extension_selected.json'
    if path.exists():assert json.loads(path.read_text())==result,'Frozen selection changed'
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':freeze()
