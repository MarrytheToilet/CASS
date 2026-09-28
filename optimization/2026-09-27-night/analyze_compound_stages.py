"""Descriptive stage-output matches; no model reasoning or mechanism inference."""
import json
from collections import defaultdict
import numpy as np
from common import HERE, ROOT
from cass.compound import COMPOUND_REGISTRY, _load_pairs, _STR_OPS
from summarize_night import read_rows, paired


def first_line(s):
    return ' '.join(s.strip().split('\n')[0].split())


def main():
    maps = {name: _load_pairs(base) for name, (base, _) in COMPOUND_REGISTRY.items()}
    records = []
    old = ROOT / 'rebuttal/2026-09-27/compound_control_generations.jsonl'
    for line in old.read_text().splitlines():
        r = json.loads(line)
        if r['condition'] in ['inferred', 'no_correction', 'true_support']:
            records.append(dict(source='submitted_controls', method=r['condition'], task=r['task'],
                                seed=r['seed'], input=r['input'], target=r['target'], prediction=r['pred']))
    for r in read_rows('support_controls'):
        if r['config'] not in ['inferred', 'no_correction', 'true_support']:
            continue
        for (x, y), pred in zip(r['queries'], r['predictions']):
            records.append(dict(source='contextual_controls', method=r['config'], task=r['task'],
                                seed=r['seed'], input=x, target=y, prediction=pred))
    for r in read_rows('signature_confirm'):
        if r['suite'] != 'compound' or r['config'] != 'original':
            continue
        for (x, y), pred in zip(r['queries'], r['predictions']):
            records.append(dict(source='submitted_new_seeds', method='inferred', task=r['task'],
                                seed=r['seed'], input=x, target=y, prediction=pred))
    categories = ['final', 'first_stage_only', 'second_stage_on_input', 'input_echo', 'other']
    grouped = defaultdict(list)
    examples = {}
    for r in records:
        name, x = r['task'], r['input']
        _, second = COMPOUND_REGISTRY[name]
        y1 = maps[name][x]
        direct_second = _STR_OPS[second](x)
        assert r['target'] == _STR_OPS[second](y1)
        pred = first_line(r['prediction'])
        references = [r['target'], y1, direct_second, x]
        category = next((label for label, answer in zip(categories, references)
                         if pred == first_line(answer)), 'other')
        grouped[r['source'], r['method'], name, r['seed']].append(category)
        key = r['source'], r['method'], category
        if category != 'final' and key not in examples:
            examples[key] = dict(r, predicted_first_line=pred, first_stage_output=y1,
                                 second_stage_on_input=direct_second, category=category)
    per_task_seed = []
    for (source, method, task, seed), values in grouped.items():
        per_task_seed.append(dict(source=source, method=method, task=task, seed=seed,
                                  n_queries=len(values), **{c:values.count(c)/len(values) for c in categories}))
    summaries = []
    for source, method in sorted({(r['source'], r['method']) for r in per_task_seed}):
        rows = [r for r in per_task_seed if (r['source'], r['method']) == (source, method)]
        by_task = defaultdict(list)
        for row in rows:
            by_task[row['task']].append(row)
        assert all(len(v) == 3 for v in by_task.values())
        means = {c:float(np.mean([np.mean([r[c] for r in rs]) for rs in by_task.values()]))
                 for c in categories}
        summaries.append(dict(source=source, method=method, n_tasks=len(by_task), **means))
    paired_diagnostics = {}
    for category in ['final', 'first_stage_only']:
        vectors = []
        for source in ['contextual_controls', 'submitted_new_seeds']:
            by_task = defaultdict(list)
            for row in per_task_seed:
                if row['source'] == source and row['method'] == 'inferred':
                    by_task[row['task']].append(row[category])
            assert set(by_task) == set(COMPOUND_REGISTRY)
            vectors.append([np.mean(by_task[n]) for n in sorted(COMPOUND_REGISTRY)])
        paired_diagnostics[category] = paired(*vectors)
    result = dict(summary=summaries, per_task_seed=per_task_seed, examples=list(examples.values()),
                  contextual_minus_submitted_same_seeds=paired_diagnostics,
                  scope='Post-hoc descriptive matching of the case-sensitive whitespace-normalized first line. '
                        'Categories are exclusive in the listed priority order. A final match can also match '
                        'an individual stage, so these counts do not identify internal sequential execution. '
                        'Other includes semantically plausible alternatives and formatting differences. '
                        'Examples are the first encountered output in each source/method/category, not a random sample.')
    (HERE / 'compound_stage_diagnostics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
