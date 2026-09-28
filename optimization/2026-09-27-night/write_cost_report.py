"""Render measured cost evidence without selecting a favorable serving contract."""
import json
from common import HERE, ROOT


def main():
    data = json.loads((HERE / 'latency_night_summary.json').read_text())
    audit = json.loads((HERE / 'timing_numerical_audit.json').read_text())
    raw = json.loads((HERE / 'optimized_latency.json').read_text())
    lines = [
        '# Measured adaptation and serving costs', '',
        f"Device: {raw['gpu']}. Llama-3.1-8B-Instruct, BF16, SDPA; physical GPU3. "
        'Models and dictionary caches are warm. Timings include host-side adaptation '
        'and synchronized GPU execution. Energy is sampled GPU-board energy, '
        'excluding CPU and system energy.', '',
        'All three predeclared timing tasks are reported. Seven grouped measurement '
        'repeats support within-run bootstrap intervals; they do not measure '
        'between-run or device variability. Fixed eight-token-budget decoding '
        'and answer-delimited decoding are separate, predeclared contracts. '
        'The latter may finish after different numbers of tokens for the two methods.', '',
        '## Original operator: exact implementation optimization', '',
        '| Task | Original extraction, s | Original adaptation, s | Optimized adaptation, s | Adaptation speedup | Original GPU J | Optimized GPU J |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for r in data['original_implementation']:
        energy={m['measurement']:m['median_gpu_board_joules'] for m in raw['measurements'] if m['task']==r['task']}
        lines.append(f"| {r['task']} | {r['extraction_b8_seconds']:.4f} | "
                     f"{r['original_adaptation_seconds']:.4f} | {r['optimized_adaptation_seconds']:.4f} | "
                     f"{r['adaptation_speedup']:.2f}× | {energy['adapt_original']:.2f} | {energy['adapt_optimized_b8']:.2f} |")
    lines += ['',
        'The original 48 sequences are six batch-8 model calls. Selective-layer '
        'collection and a cached Gram solver preserve the extraction batch shape, '
        'weighted objective, regularization path and operator. All 447 recorded '
        'query/batch comparisons retain identical generations (149 queries at '
        'serving batch sizes 1, 8 and 25). This is a tested numerical-equivalence '
        'result, not a proof for all inputs. Batch-48 extraction is faster but '
        'changes some BF16 generations and is excluded from the exact comparison.', '',
        '## Contextual variant: reusable-prefix serving', '',
        'The frozen contextual variant needs 28 extraction sequences, executed '
        'as 24 clean-context plus four zero-context sequences in two model calls. '
        'This partition preserves all 149 confirmation generations on the timing '
        'tasks. Its adaptation takes 0.196/0.165/0.411 s and approximately '
        '50.4/50.6/76.5 GPU-board joules, respectively. Dictionary/model loading '
        'is separate. The primary results below use the uncompiled operator.', '',
        'Both CASS and ICL4 use the same greedy engine and an unquantized reusable '
        'prefix cache. Cache copying and expansion are timed. CASS caches its '
        'query prefix; ICL4 caches the demonstrations. Each serving time and energy '
        'is for one batch, not one query.', '']
    for key, label in [('matched_cached_engine', 'Natural decoding, at most eight tokens'),
                       ('answer_delimited_cached_engine', 'First-answer-delimited decoding, at most eight tokens')]:
        lines += [f'### {label}', '',
            '| Task | Batch | CASS, ms | ICL4, ms | ICL4/CASS | CASS, J | ICL4, J | Latency saving 95% CI, ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in data[key]:
            if r['compiled']:
                continue
            lo, hi = [v * 1000 for v in r['latency_saving_ci95']]
            lines.append(f"| {r['task']} | {r['batch_size']} | {r['cass_seconds']*1000:.2f} | "
                         f"{r['icl4_seconds']*1000:.2f} | {r['serving_speedup']:.3f} | "
                         f"{r['cass_gpu_board_joules']:.2f} | {r['icl4_gpu_board_joules']:.2f} | "
                         f"[{lo:.2f}, {hi:.2f}] |")
        lines.append('')
    lines += [
        'Prefix caching largely removes the repeated demonstration-prefill cost '
        'on these short prompts. At batch 25, eight-token-budget serving is near '
        'parity (1.002–1.026×), and answer-delimited serving is task dependent '
        '(0.826–1.031×). These measurements do not support an order-of-magnitude '
        'wall-clock speedup. With uncached ICL4, the contextual variant takes '
        '0.196–0.198 s per batch versus 0.251–0.260 s, giving nominal warm-start '
        'amortization at 74–165 queries. That calculation does not apply to cached ICL.', '',
        'A break-even estimate divides extra adaptation cost by per-query serving '
        'savings. When the saving is indistinguishable from zero or negative, '
        'a reliable finite break-even is not established. The machine-readable '
        'summary retains the raw point estimates and their uncertainty rather '
        'than treating large ratios as measured thresholds.', '',
        '## Numerical and setup checks', '',
        'Cached ICL preserves all 447 per-query scores under both submitted prefix '
        'and literal-first-line metrics in the dedicated cache check. In the matched '
        'engine checks, CASS changes one antonym answer at batch 1 and one at batch '
        '25 (both compiled and uncompiled); the other scores are unchanged. '
        'Consequently cached-engine timing is not labeled bitwise identical to '
        'HF generation. Raw outputs and score differences are retained.', '',
        'Compilation is an optional implementation variant, not required for the '
        'primary timing rows. Compilation/setup observations, prefix-cache '
        'construction, model loading, and dictionary Gram construction are '
        'recorded separately. Warm measurements are not cold-process measurements. '
        'All five timing sources and the exact-generation check are hash-linked '
        'in `latency_night_summary.json` and `timing_numerical_audit.json`.', '',
        'The paper’s 10.2× figure describes amortized input tokens relative to '
        'ten-shot prompting. It is not a latency or energy measurement.', '']
    footprint_path = HERE / 'task_cache_footprint.json'
    if footprint_path.exists():
        foot = json.loads(footprint_path.read_text())
        lines += ['## Retained task-state tensors', '', foot['scope'], '',
            'The ratio is the median of paired task/seed ratios; it is not the '
            'ratio of the two displayed group medians. Rows are a snapshot and '
            'their task counts must be checked before citing a complete suite.', '',
            '| Suite | Tasks | Task/seeds | CASS median, MiB | ICL4 median, MiB | Median paired ICL4/CASS |',
            '|---|---:|---:|---:|---:|---:|']
        for r in foot['groups']:
            lines.append(f"| {r['suite']} | {r['n_tasks']} | {r['n_task_seeds']} | "
                         f"{r['median_cass_mib']:.3f} | {r['median_icl4_mib']:.3f} | "
                         f"{r['median_per_task_seed_ratio']:.2f}× |")
        lines += ['', 'These sizes describe per-task retained tensors, not total '
                  'VRAM, measured concurrency, or total-system throughput.', '']
    target = ROOT / 'rebuttal/2026-09-27-optimized/cost_evidence.md'
    target.write_text('\n'.join(lines))
    print(target)


if __name__ == '__main__':
    main()
