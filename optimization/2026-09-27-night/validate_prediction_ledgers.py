"""Validate saved generations, row identities, and declared target-query splits."""
import hashlib
import json
from functools import lru_cache
import numpy as np
from common import HERE, DEV_TASKS, dev_split
from cass.tasks import load_task
from cass.compound import load_compound
from cass.evaluate import accuracy
from broader_tasks import load_broader, score


@lru_cache(None)
def task_data(name):
    return load_compound(name) if '+' in name else load_task(name)


def normalized_pairs(xs):
    return [list(x) for x in xs]


def main():
    broader = load_broader()
    reports = []
    total_predictions = 0
    for path in sorted(HERE.glob('*.jsonl')):
        lines = path.read_text().splitlines()
        if not lines:
            continue
        rows = []
        trailing_partial = False
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                assert i == len(lines) - 1, (path.name, 'interior malformed row')
                trailing_partial = True
        if not rows or 'predictions' not in rows[0] or 'queries' not in rows[0]:
            continue
        seen = set()
        identities = {}
        stems = path.stem
        development = stems.endswith(('_dev', '_refine'))
        fresh = stems.endswith('_fresh')
        checked_splits = 0
        n_predictions = 0
        for row in rows:
            name, seed = row['task'], row['seed']
            method = row.get('config', row.get('method'))
            key = name, seed, method
            assert key not in seen, (path.name, key, 'duplicate')
            seen.add(key)
            predictions, queries = row['predictions'], row['queries']
            assert len(predictions) == len(queries) > 0, (path.name, key, 'length')
            assert len({x for x, y in queries}) == len(queries), (path.name, key, 'duplicate query')
            if name in broader:
                task = broader[name]
                scores = [score(name, p, y) for p, (x, y) in zip(predictions, queries)]
                assert abs(float(np.mean(scores)) - row['score']) < 1e-12
                np.testing.assert_allclose(scores, row['per_example_score'], atol=1e-12, rtol=0)
            else:
                task = task_data(name)
                value = accuracy(predictions, [y for x, y in queries], case_sensitive='+' in name)
                assert abs(value - row['acc']) < 1e-12, (path.name, key, 'metric')
            demos = row.get('demos')
            if demos is not None:
                assert len(demos) == 4
                assert not ({x for x, y in demos} & {x for x, y in queries}), (path.name, key, 'demo/query overlap')
            if development:
                assert name in DEV_TASKS and seed in [10, 11], (path.name, key, 'development domain')
                expected_demos, expected_queries = dev_split(task, seed)
            else:
                expected_queries = task.dict_pool[-50:] if fresh else task.eval_queries
                if seed in [20, 21, 22]:
                    rng = np.random.default_rng(100 * seed + 4)
                    expected_demos = [task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool), 4, replace=False)]
                else:
                    expected_demos = None
            assert queries == normalized_pairs(expected_queries), (path.name, key, 'query split')
            if demos is not None and expected_demos is not None:
                assert demos == normalized_pairs(expected_demos), (path.name, key, 'demo draw')
            if fresh:
                assert not ({x for x, y in queries} & {x for x, y in task.eval_queries + task.fewshot_pool})
            ts = name, seed
            if ts in identities:
                old_queries, old_demos = identities[ts]
                assert queries == old_queries, (path.name, key, 'within-ledger arm queries')
                if demos is not None and old_demos is not None:
                    assert demos == old_demos, (path.name, key, 'within-ledger available arm demos')
                if demos is not None:
                    identities[ts] = (queries, demos)
            else:
                identities[ts] = (queries, demos)
            checked_splits += 1
            n_predictions += len(predictions)
        total_predictions += n_predictions
        reports.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            n_rows=len(rows), n_tasks=len({r['task'] for r in rows}),
                            n_predictions=n_predictions, n_split_checks=checked_splits,
                            development=development, fresh=fresh, trailing_partial=trailing_partial,
                            metric_verified=True, row_keys_unique=True, query_splits_verified=True,
                            available_demo_draws_verified=True, within_ledger_queries_and_available_demos_identical=True))
    result = dict(files=reports, n_files=len(reports), n_predictions=total_predictions,
                  scope='Saved-output and input-split audit. Does not establish model-training independence, '
                        'prove causality, or retroactively make previously seen evaluation sets blind. '
                        'Fresh target queries are disjoint from that target’s development/demo/evaluation inputs; '
                        'target mining exclusion is implemented in the corresponding run scripts.')
    (HERE / 'prediction_ledger_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(n_files=len(reports), n_predictions=total_predictions, all_passed=True)))


if __name__ == '__main__':
    main()
