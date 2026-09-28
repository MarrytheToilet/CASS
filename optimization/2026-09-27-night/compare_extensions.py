"""Matched evidence tables across separately logged, frozen configurations."""
import hashlib
import json
import numpy as np
import pandas as pd
from common import HERE, ROOT, DEV_TASKS
from summarize_night import read_rows, paired
from cass.evaluate import accuracy
from robust_metrics import score


def main():
    result = {'scope': 'Task means over seeds 20/21/22; 50,000 paired task-bootstrap draws. '
                        'Original and extended operators use identical demonstrations and queries. '
                        'The 20 non-development LOTO targets are the primary known-task comparison.',
              'splits': {}, 'sources': {}}
    for split in ['confirm', 'fresh']:
        sources = [(f'extension_{split}', None),
                   (f'dictionary_free_refined_{split}', None)]
        if split == 'confirm':
            sources.append(('signature_confirm', {'original': 'original', 'best_correction': 'mixed'}))
        records, reference = [], {}
        for stem, mapping in sources:
            rows = read_rows(stem)
            if not rows:
                continue
            result['sources'][stem] = hashlib.sha256((HERE / (stem + '.jsonl')).read_bytes()).hexdigest()
            for r in rows:
                if mapping is not None and r['config'] not in mapping:
                    continue
                method = mapping[r['config']] if mapping is not None else r['config']
                key = (r['task'], r['seed'])
                identity = (r['queries'], r.get('demos'))
                if key in reference:
                    assert identity == reference[key], (stem, key, 'unmatched inputs')
                else:
                    reference[key] = identity
                assert abs(accuracy(r['predictions'], [y for x, y in r['queries']],
                                    case_sensitive='+' in r['task']) - r['acc']) < 1e-12
                values = {'submitted_prefix': r['acc']}
                for metric in ['task_case_prefix', 'task_case_full_line', 'task_case_literal_line']:
                    values[metric] = float(np.mean([score(p, y, r['task'], metric)
                                                   for p, (x, y) in zip(r['predictions'], r['queries'])]))
                for metric, value in values.items():
                    records.append(dict(task=r['task'], seed=r['seed'], method=method,
                                        suite=r['suite'], metric=metric, value=value))
        if not records:
            continue
        df = pd.DataFrame(records)
        assert not df.duplicated(['task', 'seed', 'method', 'metric']).any()
        suites = list(df.groupby('suite'))
        suites.append(('nondevelopment_loto', df[(df.suite == 'loto') & ~df.task.isin(DEV_TASKS)]))
        comparisons = [('combined', 'original'), ('combined', 'mixed'),
                       ('combined', 'combined_no_correction'),
                       ('combined', 'dictionary_free'), ('combined', 'dictionary_free_refined'),
                       ('oneshot', 'oneshot_no_correction'), ('oneshot', 'icl1'),
                       ('combined', 'icl4'), ('oneshot', 'icl4')]
        split_out = {}
        for suite, part in suites:
            if part.empty:
                continue
            split_out[suite] = {}
            for metric, group in part.groupby('metric'):
                table = group.pivot(index=['task', 'seed'], columns='method', values='value')
                entry = {'means': {}, 'comparisons': {}}
                for method in table:
                    column = table[method].dropna()
                    valid = column.groupby(level='task').count()
                    column = column[column.index.get_level_values('task').isin(valid[valid == 3].index)]
                    means = column.groupby(level='task').mean()
                    if len(means):
                        entry['means'][method] = {'n_tasks': len(means), 'accuracy': float(means.mean())}
                for a, b in comparisons:
                    if a not in table or b not in table:
                        continue
                    pairs = table[[a, b]].dropna()
                    valid = pairs.groupby(level='task').size()
                    pairs = pairs[pairs.index.get_level_values('task').isin(valid[valid == 3].index)]
                    means = pairs.groupby(level='task').mean()
                    if len(means):
                        entry['comparisons'][a + '__' + b] = paired(means[a], means[b])
                split_out[suite][metric] = entry
        result['splits'][split] = split_out
    (HERE / 'extension_comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['splits'].get('confirm', {}).get('nondevelopment_loto', {}), indent=2))


if __name__ == '__main__':
    main()
