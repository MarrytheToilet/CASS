"""Audit all predeclared NLP cached-serving measurements and render evidence."""
import hashlib
import json
import math
import numpy as np
from common import HERE, ROOT
from broader_tasks import NLP_FILES, clean_answer


def resampled_difference(left, right, rng):
    """Independent within-run resampling; repeat indices are not paired trials."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    aa = np.median(a[rng.integers(0, len(a), (50000, len(a)))], axis=1)
    bb = np.median(b[rng.integers(0, len(b), (50000, len(b)))], axis=1)
    return [float(x) for x in np.quantile(aa - bb, [.025, .975])]


def main():
    source = HERE / 'broader_cached_latency.json'
    raw = json.loads(source.read_text())
    assert raw.get('complete') is True
    assert raw['cuda_visible_devices'] in ['2', '3']
    assert raw['max_new_tokens'] == 32 and raw['stop_after_answer'] is True
    rows = raw['measurements']
    assert set(r['task'] for r in rows) == set(NLP_FILES)
    lookup = {(r['task'], r['measurement'], r.get('batch_size')): r for r in rows}
    assert len(lookup) == len(rows) == 7 * 9
    checks = raw['checks']
    assert len(checks) == 7 * 7
    rng = np.random.default_rng(20260927)
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  gpu=raw['gpu'], cuda_visible_devices=raw['cuda_visible_devices'],
                  torch=raw['torch'], model_load_seconds=raw['model_load_seconds'],
                  library_load_build_seconds=raw['library_load_build_seconds'],
                  complete=True, rows=[], generation_checks=[], adaptation_checks=[],
                  interval_scope='Independent bootstrap of seven within-run grouped timing repeats; not between-run uncertainty or per-request tail latency.',
                  power_error=raw.get('power_error'),
                  scope=raw['scope'], energy_scope=raw['energy_scope'])
    for name in NLP_FILES:
        adapt = lookup[name, 'adaptation_28_sequences_7_calls', 4]
        cp = lookup[name, 'build_cass_prefix', None]
        ip = lookup[name, 'build_icl4_prefix', None]
        overhead = adapt['median_seconds'] + cp['median_seconds'] - ip['median_seconds']
        matching = [c for c in checks if c['task'] == name]
        signature = [c for c in matching if c['kind'] == 'signature_reextraction']
        assert len(signature) == 1
        report['adaptation_checks'].append(dict(task=name,
            adaptation_seconds=adapt['median_seconds'],
            adaptation_gpu_board_joules=adapt['median_gpu_board_joules'],
            cass_prefix_seconds=cp['median_seconds'], icl4_prefix_seconds=ip['median_seconds'],
            extra_warm_setup_seconds=overhead, **{k:v for k,v in signature[0].items() if k!='task'}))
        generation = [c for c in matching if c['kind'] == 'cached_generation']
        assert {(c['method'],c['batch_size']) for c in generation} == {(m,b) for m in ['cass','icl4'] for b in [1,4,8]}
        for c in generation:
            n = 100 if c['batch_size'] == 4 else 32
            assert c['n_queries'] == n
            assert all(len(c[k]) == n for k in ['before','after','per_query_before','per_query_after'])
            assert c['different_queries'] == sum(a!=b for a,b in zip(c['before'],c['after']))
            assert abs(c['before_score'] - np.mean(c['per_query_before'])) < 1e-10
            assert abs(c['after_score'] - np.mean(c['per_query_after'])) < 1e-10
            report['generation_checks'].append({k:v for k,v in c.items() if k not in ['before','after','per_query_before','per_query_after']} | {
                'changed_scores':sum(a!=b for a,b in zip(c['per_query_before'],c['per_query_after'])),
                'changed_first_answers':sum(clean_answer(a)!=clean_answer(b) for a,b in zip(c['before'],c['after']))})
        for bs in [1,4,8]:
            cass, icl = [lookup[name, 'serve_'+m, bs] for m in ['cass','icl4']]
            for r in [cass,icl]:
                assert r['n_queries'] == len(r['predictions']) == 32
                assert len(r['repeats']) == 7
                assert math.isclose(r['median_seconds'], np.median([p['seconds'] for p in r['repeats']]), rel_tol=1e-12)
                assert math.isclose(r['queries_per_second'], 32/r['median_seconds'], rel_tol=1e-12)
            delta = icl['median_seconds'] - cass['median_seconds']
            ci = resampled_difference([p['seconds'] for p in icl['repeats']],
                                      [p['seconds'] for p in cass['repeats']], rng)
            report['rows'].append(dict(task=name,batch_size=bs,n_queries=32,
                cass_seconds=cass['median_seconds'],icl4_seconds=icl['median_seconds'],
                cass_qps=cass['queries_per_second'],icl4_qps=icl['queries_per_second'],
                throughput_ratio=cass['queries_per_second']/icl['queries_per_second'],
                workload_seconds_saving_ci95=ci,
                cass_gpu_board_joules=cass['median_gpu_board_joules'],
                icl4_gpu_board_joules=icl['median_gpu_board_joules'],
                cass_reencoded_output_tokens=sum(cass['reencoded_output_tokens_excluding_specials']),
                icl4_reencoded_output_tokens=sum(icl['reencoded_output_tokens_excluding_specials']),
                nominal_break_even_queries=max(0,overhead)*32/delta if delta>0 else None,
                positive_saving_interval=ci[0]>0))
    (HERE/'broader_cached_latency_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Cached serving across all seven broader NLP tasks','',
        f"Device: {raw['gpu']}, physical GPU{raw['cuda_visible_devices']}; "
        f"Llama-3.1-8B-Instruct, BF16, PyTorch {raw['torch']}.", '',
        'All seven previously specified NLP tasks are included. The frozen contextual '
        'operator and ICL4 use the same uncompiled greedy engine, a reusable prefix '
        'cache, a 32-token limit and first-answer stopping. Every timed repeat processes '
        'the same first 32 queries; each row gives workload throughput, not individual '
        'request latency. Prefix copying and expansion are timed. Different generated '
        'answer lengths remain part of the workload. Settings are fixed and seed 20 '
        'uses the same demonstrations as the broader accuracy study.','',
        report['interval_scope']+' '+report['energy_scope']+'.','',
        '| Task | Batch | CASS queries/s | ICL4 queries/s | Throughput ratio | CASS J/32 queries | ICL4 J/32 queries | Workload time saving, s [95% CI] |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    for r in report['rows']:
        lo,hi=r['workload_seconds_saving_ci95']
        lines.append(f"| {r['task']} | {r['batch_size']} | {r['cass_qps']:.2f} | {r['icl4_qps']:.2f} | {r['throughput_ratio']:.3f} | {number(r['cass_gpu_board_joules'])} | {number(r['icl4_gpu_board_joules'])} | [{lo:.4f}, {hi:.4f}] |")
    lines += ['', '## Warm task setup', '',
        'Models and dictionaries are already resident. Adaptation executes 28 '
        'sequences in seven batch-4 calls. Both methods’ prefix construction is '
        'measured separately; CASS adaptation includes extraction, sparse coding '
        'and operator construction. These batch-4 times are distinct from the '
        'two-call short-task measurements. The JSON retains model/library load '
        'times and nominal amortization estimates; a finite reliable break-even '
        'is not established when the serving-time saving includes zero.', '',
        '| Task | Adaptation, s | Adaptation, GPU J | CASS prefix, s | ICL4 prefix, s | Same support on re-extraction | Maximum signature difference |',
        '|---|---:|---:|---:|---:|---|---:|']
    for r in report['adaptation_checks']:
        lines.append(f"| {r['task']} | {r['adaptation_seconds']:.4f} | {number(r['adaptation_gpu_board_joules'])} | {r['cass_prefix_seconds']:.4f} | {r['icl4_prefix_seconds']:.4f} | {r['identical_support']} | {max(r['max_abs_error'].values()):.6g} |")
    lines += ['', '## Generation and score parity', '',
        'Batch 4 compares all 100 query outputs to the saved accuracy run. Batches 1/8 '
        'compare the first 32 outputs to fresh uncached generation at that batch size. '
        'This check does not assume cached and uncached BF16 outputs are identical. '
        'Some saved ICL4 accuracy baselines predate first-answer stopping and contain '
        'continuations after the answer. Their raw-string differences therefore '
        'include those removed continuations; parsed first-answer differences and '
        'score differences are recorded separately. All timed arms use the same '
        'first-answer stopping contract. '
        'All raw before/after strings and per-query scores are retained.', '',
        '| Task | Method | Batch | Queries | Different raw strings | Different first answers | Different scores | Uncached score | Cached score |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in report['generation_checks']:
        lines.append(f"| {c['task']} | {c['method']} | {c['batch_size']} | {c['n_queries']} | {c['different_queries']} | {c['changed_first_answers']} | {c['changed_scores']} | {c['before_score']:.4f} | {c['after_score']:.4f} |")
    lines += ['', 'Source SHA256: `'+report['source_sha256']+'`.', '']
    target=ROOT/'rebuttal/2026-09-27-optimized/broader_cost_evidence.md'
    target.write_text('\n'.join(lines))
    print(target)


if __name__=='__main__':main()
