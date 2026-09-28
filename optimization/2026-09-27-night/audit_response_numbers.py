"""Cross-check the principal rebuttal numbers against their named evidence sources.

This supplements, rather than replaces, reading the scientific interpretation.
It checks selected numerical claims and tables, not every numeral in prose.
"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESP=ROOT/'rebuttal/2026-09-27-optimized'
OLD=ROOT/'rebuttal/2026-09-27'


def main():
    sources={};checks=[]
    docs={name:(RESP/f'response_{name}.md').read_text() for name in ['BiuA','Bm1S','cajh','DuwT','cajh_compact']}
    docs['RESULTS_zh']=(RESP/'RESULTS_zh.md').read_text()
    assert len(docs['cajh_compact']) < 5000
    def source(path):
        raw=path.read_bytes();sources[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def require(reviewer,identifier,expected,provenance):
        assert expected in docs[reviewer], (reviewer,identifier,expected)
        checks.append(dict(reviewer=reviewer,claim=identifier,verified_text=expected,source=provenance))
    def interval(value,key='diff'):
        lo,hi=value['ci95']
        return f"{100*value[key]:+.2f} [{100*lo:.2f}, {100*hi:.2f}]"
    live=source(HERE/'live_summary.json')
    robust=source(HERE/'metric_robustness.json')['experiments']
    matched=source(OLD/'matched_results.json')
    for suite,label in [('loto','LOTO32'),('novel','Novel15')]:
        r=matched[suite]['fixed_original_route'];m=r['accuracies']
        require('BiuA',suite+' original identical-route table',
            f"| {label} | {m['cass']:.4f} | {m['zvec']:.4f} | {interval(r['cass_minus_zvec'])} |",
            'matched_results.json / '+suite+'/fixed_original_route')
    suites=[('heldout_known','Non-development LOTO20'),('novel','Novel15'),('compound','Compound10')]
    for suite,label in suites:
        r=live['extension_confirm']['suites'][suite]
        methods=['combined','combined_no_correction','icl4']
        values=[live['llama_confirm']['suites'][suite]['means']['original']]+[r['means'][m] for m in methods]
        for reviewer in ['cajh','cajh_compact']:
            require(reviewer,suite+' four-example steering table',
                '| '+label+' | '+' | '.join(f'{v:.4f}' for v in values)+' |',
                'live_summary.json / llama_confirm and extension_confirm/'+suite)
        gain=interval(r['combined__combined_no_correction'])
        for reviewer in ['BiuA','Bm1S','cajh','cajh_compact']:
            require(reviewer,suite+' contextual correction gain',gain,
                    'live_summary.json / extension_confirm/'+suite+'/combined__combined_no_correction')
        summary_label={'heldout_known':'非开发 LOTO20','novel':'Novel15','compound':'Compound10'}[suite]
        fresh_means=live['extension_fresh']['suites'][suite]['means']
        summary_values=[values[0],r['means']['combined'],fresh_means['original'],fresh_means['combined']]
        require('RESULTS_zh',suite+' author overview table',
                '| '+summary_label+' | '+' | '.join(f'{v:.4f}' for v in summary_values)+' |',
                'live_summary.json / llama_confirm, extension_confirm, extension_fresh/'+suite)
    full='/'.join(f"{live['extension_confirm']['suites'][s]['means']['combined']:.4f}" for s,_ in suites)
    off='/'.join(f"{live['extension_confirm']['suites'][s]['means']['combined_no_correction']:.4f}" for s,_ in suites)
    require('Bm1S','contextual full and off triples',full,'live_summary.json / extension_confirm')
    require('Bm1S','contextual off triple',off,'live_summary.json / extension_confirm')
    fresh='/'.join(f"{live['extension_fresh']['suites'][s]['means']['combined']:.4f}" for s,_ in suites)
    require('cajh','additional-query means',fresh,'live_summary.json / extension_fresh')
    for suite,label in suites:
        r=live['extension_fresh']['suites'][suite]
        require('BiuA',suite+' fresh prefix gain',interval(r['combined__combined_no_correction']),
                'live_summary.json / extension_fresh/'+suite)
        strict=robust['extension_fresh'][suite]['task_case_literal_line']['paired']['combined__combined_no_correction']
        require('BiuA',suite+' fresh literal gain',interval(strict),
                'metric_robustness.json / extension_fresh/'+suite+'/task_case_literal_line')
    comp=source(OLD/'fresh_compound_summary.json')
    for policy,label in [('unrouted','Steering on every task'),('fixed_route','Submitted route, identical replacements')]:
        values=[comp[policy][m]['mean'] for m in ['inferred','true_support','z_only']]
        require('BiuA',policy+' compound support table',
                '| '+label+' | '+' | '.join(f'{v:.4f}' for v in values)+' |',
                'fresh_compound_summary.json / '+policy)
    for policy in ['unrouted','fixed_route']:
        r=comp['paired_differences'][policy]['inferred minus wrong_matched']['result']
        for reviewer in ['BiuA','Bm1S']:
            # Bm1S separates the point estimate and interval with a word.
            require(reviewer,policy+' wrong-support interval',
                    f"[{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]",
                    'fresh_compound_summary.json / paired_differences/'+policy+'/inferred minus wrong_matched')
    support=source(HERE/'support_controls_summary.json')
    for method in ['inferred','no_correction','true_support']:
        value=support['means'][method]
        if isinstance(value,dict):value=value.get('mean',value.get('accuracy'))
        for reviewer in ['BiuA','Bm1S']:
            require(reviewer,'new compound '+method,f'{value:.4f}',
                    'support_controls_summary.json / means/'+method)
    coh=source(OLD/'coherence_results.json')
    for model,r in coh.items():
        require('Bm1S',model+' maximum coherence',f"{r['full']['max_joint']:.4f}",
                'coherence_results.json / '+model+'/full/max_joint')
    for suite,label in suites:
        r=live['qwen3-4b_contrast_confirm']['suites'][suite]
        for method in ['best_correction','matched_no_correction','plain_original','icl4']:
            for reviewer in ['cajh','cajh_compact']:
                require(reviewer,'Qwen3 '+suite+'/'+method,f"{r['means'][method]:.4f}",
                        'live_summary.json / qwen3-4b_contrast_confirm/'+suite+'/means/'+method)
        require('cajh','Qwen3 '+suite+' gain over original',interval(r['best_correction__plain_original']),
                'live_summary.json / qwen3-4b_contrast_confirm/'+suite+'/best_correction__plain_original')
        for metric in ['task_case_full_line','task_case_literal_line']:
            value=robust['qwen3-4b_contrast_confirm'][suite][metric]['means']['best_correction']
            require('cajh','Qwen3 '+suite+'/'+metric,f'{value:.4f}',
                    'metric_robustness.json / qwen3-4b_contrast_confirm/'+suite+'/'+metric)
    broad=source(HERE/'broader_analysis.json')
    for suite,label in suites:
        entry=live['qwen25-3b_contrast_confirm']['suites'][suite]
        for method in ['best_correction','best_overall','plain_original']:
            require('cajh','Qwen2.5 '+suite+'/'+method,f"{entry['means'][method]:.4f}",
                    'live_summary.json / qwen25-3b_contrast_confirm/'+suite+'/means/'+method)
        require('cajh','Qwen2.5 '+suite+' gain over original',interval(entry['best_correction__plain_original']),
                'live_summary.json / qwen25-3b_contrast_confirm/'+suite+'/best_correction__plain_original')
        for model in ['qwen3-4b','qwen25-3b']:
            value=live[model+'_contrast_fresh']['suites'][suite]['means']['best_correction']
            require('cajh',model+' '+suite+' additional-query accuracy',f'{value:.4f}',
                    'live_summary.json / '+model+'_contrast_fresh/'+suite)
    families={r['metric_family']:r for r in broad['families']}
    for family,methods in [
        ('classification_accuracy',['combined','original_cass','icl4']),
        ('entity_token_f1',['combined','original_cass','icl4']),
        ('reading_token_f1',['combined','original_cass','icl4']),
        ('long_structured_whole_answer',['combined','original_cass','icl4']),
        ('code_expression_execution',['oneshot','icl1']),
        ('code_multistep_execution',['oneshot','icl1'])]:
        for method in methods:
            value=families[family]['means'][method]['score']
            require('cajh',family+'/'+method,f'{value:.4f}',
                    'broader_analysis.json / families/'+family+'/means/'+method)
    scale=source(HERE/'scale_context_fixed_shared_analysis.json')['scaling']
    broad_cost=source(HERE/'broader_cached_latency_analysis.json')
    ratios=[r['throughput_ratio'] for r in broad_cost['rows'] if r['batch_size']==8]
    require('cajh','all-seven-task cached throughput range',
            f'{min(ratios):.3f} to {max(ratios):.3f}',
            'broader_cached_latency_analysis.json / rows/batch_size8')
    setup=[r['adaptation_seconds'] for r in broad_cost['adaptation_checks']]
    require('cajh','all-seven-task warm adaptation range',
            f'{min(setup):.3f}–{max(setup):.3f}',
            'broader_cached_latency_analysis.json / adaptation_checks')
    rank=live['shared_rank_confirm']['suites']['novel']['means']
    require('DuwT','fixed-setting shared-rank curve',
            '/'.join(f'{rank["rank"+str(k)]:.4f}' for k in [0,1,2,4]),
            'live_summary.json / shared_rank_confirm/novel')
    known=source(HERE/'known_task_calibration.json')
    for r in known['results']:
        if r['stem']=='scale_eval_fixed_shared' and r['size']==256 and r['solver_mode']=='full' and r['features'] in ['norm','constant_prior','residual']:
            require('cajh','known-task risk Brier '+r['features'],f"{r['brier']:.4f}",
                    'known_task_calibration.json / scale_eval_fixed_shared/256/full/'+r['features'])
    for split in ['confirm','fresh']:
        triple='/'.join(f"{live['position_'+split]['suites'][s]['means']['dictionary_free']:.4f}" for s,_ in suites)
        for reviewer in ['BiuA','Bm1S']:
            require(reviewer,'position-tuned dictionary-free '+split,triple,
                    'live_summary.json / position_'+split+'/dictionary_free')
    full=[next(r for r in scale if r['size']==n and r['mode']=='full') for n in [32,64,128,256]]
    sequence='/'.join(f"{r['accuracy']:.4f}" for r in full)
    short=next(r for r in scale if r['size']==256 and r['mode']=='shortlist32')
    for reviewer in ['cajh','DuwT']:
        require(reviewer,'fixed shared direction scale curve',sequence,
                'scale_context_fixed_shared_analysis.json / scaling/full')
        for name,value,precision in [('shortlist accuracy',short['accuracy'],4),
                                     ('full coding seconds',full[-1]['coding_median_seconds'],3),
                                     ('shortlist coding seconds',short['coding_median_seconds'],3)]:
            require(reviewer,name,f'{value:.{precision}f}',
                    'scale_context_fixed_shared_analysis.json / scaling/256')
    result=dict(n_checks=len(checks),checks=checks,source_sha256=sources,
                response_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in docs.items()},
                scope='Selected principal numerical claims only. Does not automate scientific interpretation, certify every numeral, or replace source-generation validation.')
    (HERE/'response_number_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(n_checks=len(checks),all_passed=True)))


if __name__=='__main__':main()
