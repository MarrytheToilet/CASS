"""Describe formatting/continuation errors without modifying any scorer."""
import hashlib
import json
import re
from collections import defaultdict
import numpy as np
from common import HERE, ROOT
from robust_metrics import score, CASE_TASKS


def main():
    output=dict(scope='Post-hoc output diagnosis only; no parsing rule, setting, or accuracy value is changed.',experiments={})
    lines=['# Qwen output-format diagnosis','',output['scope'],'',
        'The original metric accepts a normalized answer prefix. The two '
        'complete-first-line checks respectively normalize punctuation or preserve '
        'it literally; both retain task-relevant case. Suffix-only cases are '
        'literal mismatches repaired solely by removing final periods, question '
        'marks or exclamation marks. This diagnostic classification uses references '
        'and is not a proposed serving-time postprocessor.','']
    for model in ['qwen3-4b','qwen25-3b']:
        for split in ['confirm','fresh']:
            stem=model+'_contrast_'+split
            path=HERE/(stem+'.jsonl')
            if not (HERE/(stem+'_done.json')).exists():continue
            raw=[json.loads(s) for s in path.read_text().splitlines()]
            entry=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=[],examples=[])
            for suite,n in [('heldout_known',20),('novel',15),('compound',10)]:
                for method in ['best_correction','matched_no_correction','plain_original','icl4']:
                    current=[r for r in raw if r['suite']==suite and r['config']==method]
                    assert len(current)==n*3 and len({r['task'] for r in current})==n
                    values=defaultdict(list)
                    for r in current:
                        per=defaultdict(list)
                        def normalize(text):
                            line=' '.join(text.strip().split('\n')[0].split())
                            return line if '+' in r['task'] or r['task'] in CASE_TASKS else line.lower()
                        for pred,(x,y) in zip(r['predictions'],r['queries']):
                            prefix=score(pred,y,r['task'],'task_case_prefix')
                            words=score(pred,y,r['task'],'task_case_full_line')
                            literal=score(pred,y,r['task'],'task_case_literal_line')
                            suffix=bool(words and not literal and normalize(pred).rstrip('.!?')==normalize(y))
                            per['case_aware_prefix'].append(prefix)
                            per['normalized_full_line'].append(words)
                            per['literal_full_line'].append(literal)
                            per['prefix_but_not_full_line'].append(bool(prefix and not words))
                            per['full_line_punctuation_difference'].append(bool(words and not literal))
                            per['suffix_punctuation_only'].append(suffix)
                            category='suffix_punctuation_only' if suffix else ('extra_words' if prefix and not words else None)
                            if category and method=='best_correction' and r['seed']==20:
                                key=(suite,category)
                                if sum((e['suite'],e['category'])==key for e in entry['examples'])<6:
                                    entry['examples'].append(dict(suite=suite,category=category,task=r['task'],query=x,target=y,prediction=pred))
                        for metric,items in per.items():values[metric].append(float(np.mean(items)))
                    entry['groups'].append(dict(suite=suite,method=method,n_tasks=n,
                                               **{k:float(np.mean(v)) for k,v in values.items()}))
            output['experiments'][stem]=entry
            lines += ['## '+stem,'',
                '| Suite | Method | Case-aware prefix | Normalized full line | Literal full line | Prefix-only | Punctuation mismatch | Suffix-only mismatch |',
                '|---|---|---:|---:|---:|---:|---:|---:|']
            for r in entry['groups']:
                metrics=['case_aware_prefix','normalized_full_line','literal_full_line','prefix_but_not_full_line','full_line_punctuation_difference','suffix_punctuation_only']
                lines.append('| '+r['suite']+' | '+r['method']+' | '+' | '.join(f'{r[m]:.4f}' for m in metrics)+' |')
            lines += ['', 'Values first average queries within each task/seed, then '
                'average the three seeds and equally weight tasks. Error-category '
                'rates are descriptive fractions, not new accuracy metrics.','',
                'Examples are the first six encountered per suite/category at seed20, '
                'with their raw strings retained in `qwen_format_diagnostics.json`.', '',
                'Source SHA256: `'+entry['source_sha256']+'`.','']
    (HERE/'qwen_format_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    target=ROOT/'rebuttal/2026-09-27-optimized/qwen_format_evidence.md'
    target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
