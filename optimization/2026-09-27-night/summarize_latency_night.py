"""Combine completed timing experiments without equating token and time savings."""
import hashlib
import json
import math
import numpy as np
from common import HERE


def breakeven(startup_a,startup_b,batch_a,batch_b,bs):
    if any(v is None for v in [startup_a,startup_b,batch_a,batch_b]):return None
    saving=(batch_b-batch_a)/bs
    if saving<=0:return None
    return max(0,int(math.ceil((startup_a-startup_b)/saving)))


def repeat_uncertainty(a,b):
    result={'uncertainty_scope':'Independent bootstrap of seven grouped repeat measurements; conditional on this run, not between-run or device variability.'}
    for column,label in [('seconds','latency'),('gpu_board_joules','gpu_board_energy')]:
        x=np.asarray([r[column] for r in a['repeats'] if r.get(column) is not None])
        y=np.asarray([r[column] for r in b['repeats'] if r.get(column) is not None])
        if not len(x) or not len(y):continue
        rng=np.random.default_rng(20260927)
        delta=np.median(y[rng.integers(len(y),size=(50000,len(y)))],axis=1)-np.median(x[rng.integers(len(x),size=(50000,len(x)))],axis=1)
        interval=np.quantile(delta,[.025,.975]).tolist()
        result[label+'_saving_ci95']=interval
        result[label+'_saving_positive_over_repeat_interval']=interval[0]>0
    return result


def main():
    sources={};data={};expected={'optimized_latency':45,'variant_latency':36,
                              'cached_icl_latency':21,'matched_engine_latency':33,'answer_engine_latency':33}
    for stem,n in expected.items():
        path=HERE/(stem+'.json')
        if not path.exists():continue
        value=json.loads(path.read_text())
        if len(value['measurements'])!=n:continue
        if stem!='optimized_latency' and not value.get('complete'):continue
        assert value['cuda_visible_devices'] in ['2','3']
        data[stem]=value;sources[stem]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(sources=sources,scope='Warm models and dictionary caches; medians of repeated measurements. '
                'Compilation/model loading/offline mining are reported separately. GPU-board energy excludes CPU/system energy.',
                original_implementation=[],contextual_uncached=[],matched_cached_engine=[],answer_delimited_cached_engine=[],correctness={},setup={})
    for stem,value in data.items():
        result['correctness'][stem]=value['checks']
        result['setup'][stem]={k:value[k] for k in ['offline','model_load_seconds','caches','prefixes'] if k in value}
        result['setup'][stem]['first_measured_calls']=[
            {k:r[k] for k in ['task','measurement','batch_size','warmup_seconds'] if k in r}
            for r in value['measurements']]
        result['setup'][stem]['first_call_scope']='Warmup calls may include shape-specific compilation; earlier correctness checks may already have compiled other shapes. Not a clean-process cold-start benchmark.'
    def row(stem,task,label,bs=None):
        return next(r for r in data[stem]['measurements'] if r['task']==task and r['measurement']==label
                    and (bs is None or r.get('batch_size')==bs))
    if 'optimized_latency' in data:
        for task in ['antonym','country-capital','english-french']:
            old=row('optimized_latency',task,'adapt_original');fast=row('optimized_latency',task,'adapt_optimized_b8')
            entry=dict(task=task,original_adaptation_seconds=old['median_seconds'],
                       optimized_adaptation_seconds=fast['median_seconds'],adaptation_speedup=old['median_seconds']/fast['median_seconds'],
                       optimized_extraction_batch_size=8,
                       single_batch48_adaptation_seconds=row('optimized_latency',task,'adapt_optimized_b48')['median_seconds'],
                       batch48_scope='Separate throughput measurement; BF16 batch-shape changes may alter greedy generations. Main matched comparison preserves extraction batch8.',
                       extraction_b8_seconds=row('optimized_latency',task,'extract_original_b8')['median_seconds'],
                       selective_b48_extraction_seconds=row('optimized_latency',task,'extract_selected_b48')['median_seconds'],serving=[])
            for bs in [1,8,25]:
                for compiled in [False,True]:
                    a=row('optimized_latency',task,'serve_compiled' if compiled else 'serve_original',bs)
                    b=row('optimized_latency',task,'serve_icl4',bs)
                    entry['serving'].append(dict(batch_size=bs,compiled=compiled,cass_seconds=a['median_seconds'],icl4_seconds=b['median_seconds'],
                        cass_gpu_board_joules=a['median_gpu_board_joules'],icl4_gpu_board_joules=b['median_gpu_board_joules'],
                        break_even_queries=breakeven(fast['median_seconds'],0,a['median_seconds'],b['median_seconds'],bs),
                        board_energy_break_even_queries=breakeven(fast['median_gpu_board_joules'],0,a['median_gpu_board_joules'],b['median_gpu_board_joules'],bs)))
            result['original_implementation'].append(entry)
    if 'variant_latency' in data:
        for task in ['antonym','country-capital','english-french']:
            for compiled in [False,True]:
                adapt=row('variant_latency',task,'adapt_variant_compiled' if compiled else 'adapt_variant')
                for bs in [1,8,25]:
                    a=row('variant_latency',task,'serve_variant_compiled' if compiled else 'serve_variant',bs)
                    b=row('variant_latency',task,'serve_icl4',bs)
                    result['contextual_uncached'].append(dict(task=task,compiled=compiled,batch_size=bs,
                        adaptation_seconds=adapt['median_seconds'],cass_seconds=a['median_seconds'],icl4_seconds=b['median_seconds'],
                        cass_gpu_board_joules=a['median_gpu_board_joules'],icl4_gpu_board_joules=b['median_gpu_board_joules'],
                        break_even_queries=breakeven(adapt['median_seconds'],0,a['median_seconds'],b['median_seconds'],bs),
                        board_energy_break_even_queries=breakeven(adapt['median_gpu_board_joules'],0,a['median_gpu_board_joules'],b['median_gpu_board_joules'],bs)))
    for engine,outkey in [('matched_engine_latency','matched_cached_engine'),('answer_engine_latency','answer_delimited_cached_engine')]:
        if not {'variant_latency',engine}<=set(data):continue
        for task in ['antonym','country-capital','english-french']:
            pc=row(engine,task,'build_cass_prefix');pi=row(engine,task,'build_icl4_prefix')
            for compiled in [False,True]:
                adapt=row('variant_latency',task,'adapt_variant_compiled' if compiled else 'adapt_variant')
                for bs in [1,8,25]:
                    a=row(engine,task,'serve_cass_matched_engine' if compiled else 'serve_cass_uncompiled_matched_engine',bs)
                    b=row(engine,task,'serve_icl4_matched_engine',bs)
                    start_a=adapt['median_seconds']+pc['median_seconds'];start_b=pi['median_seconds']
                    ea=adapt['median_gpu_board_joules'];ep=pc['median_gpu_board_joules']
                    energy_a=ea+ep if ea is not None and ep is not None else None
                    result[outkey].append(dict(task=task,compiled=compiled,batch_size=bs,
                        cass_startup_seconds=start_a,icl4_startup_seconds=start_b,
                        cass_seconds=a['median_seconds'],icl4_seconds=b['median_seconds'],
                        serving_speedup=b['median_seconds']/a['median_seconds'],
                        cass_gpu_board_joules=a['median_gpu_board_joules'],icl4_gpu_board_joules=b['median_gpu_board_joules'],
                        break_even_queries=breakeven(start_a,start_b,a['median_seconds'],b['median_seconds'],bs),
                        board_energy_break_even_queries=breakeven(energy_a,pi['median_gpu_board_joules'],a['median_gpu_board_joules'],b['median_gpu_board_joules'],bs),
                        break_even_scope='Nominal point estimate from medians. A near-zero difference does not establish a reliable break-even; consult repeat interval. Startup uncertainty is not propagated.',
                        **repeat_uncertainty(a,b)))
    (HERE/'latency_night_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Complete timing sources:',list(data))
    print(json.dumps(result['matched_cached_engine'],indent=2))


if __name__=='__main__':main()
