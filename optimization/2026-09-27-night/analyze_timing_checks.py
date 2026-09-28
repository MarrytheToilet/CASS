"""Separate raw-generation parity from answer-score parity in timing artifacts."""
import hashlib
import json
from common import HERE
from robust_metrics import score
from cass.tasks import load_task


def main():
    result=dict(scope='Only the predeclared three timing tasks and their recorded batches; not a general numerical-equivalence guarantee.',sources={},checks=[])
    for stem in ['optimized_latency','variant_latency','cached_icl_latency','matched_engine_latency','answer_engine_latency','optimized_generation_checks']:
        path=HERE/(stem+'.json')
        if not path.exists():continue
        data=json.loads(path.read_text())
        if stem!='optimized_latency' and not data.get('complete'):continue
        if stem=='optimized_latency' and len(data['measurements'])!=45:continue
        result['sources'][stem]=hashlib.sha256(path.read_bytes()).hexdigest()
        for check in data['checks']:
            item={k:v for k,v in check.items() if k not in ['before','after','queries']}
            item['source']=stem
            if 'before' in check and 'after' in check:
                task=load_task(check['task']);queries=task.eval_queries
                assert len(check['before'])==len(check['after'])==len(queries)
                item['n_queries']=len(queries);item['metrics']={}
                for metric in ['task_case_prefix','task_case_literal_line']:
                    before=[score(p,y,task.name,metric) for p,(x,y) in zip(check['before'],queries)]
                    after=[score(p,y,task.name,metric) for p,(x,y) in zip(check['after'],queries)]
                    item['metrics'][metric]=dict(different_query_scores=sum(a!=b for a,b in zip(before,after)),
                        reference_accuracy=sum(before)/len(before),timed_accuracy=sum(after)/len(after))
            result['checks'].append(item)
    (HERE/'timing_numerical_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed sources:',list(result['sources']))
    for stem in result['sources']:
        items=[r for r in result['checks'] if r['source']==stem and 'metrics' in r]
        if items:
            print(stem,{metric:sum(r['metrics'][metric]['different_query_scores'] for r in items)
                        for metric in ['task_case_prefix','task_case_literal_line']})


if __name__=='__main__':main()
