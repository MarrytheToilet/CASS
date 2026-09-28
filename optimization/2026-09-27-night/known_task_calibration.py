"""Known-only supervised risk calibration; transfer to every scale condition."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import HERE, ROOT, ALL_TASKS
from analyze_scaling import fit, predict

FEATURES=[('residual',['residual']),('norm',['znorm']),('joint',['residual','znorm'])]


def main():
    hashes={}
    def read(stem):
        raw=(HERE/(stem+'.jsonl')).read_bytes();hashes[stem]=hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.decode().splitlines()]
    result=dict(protocol='Additional post-hoc known-task calibration; see KNOWN_RISK_PROTOCOL.md.',
        observation_unit='One task and one four-demonstration seed; failure is that seed accuracy < 0.5.',
        uncertainty='50,000 paired task-cluster bootstrap draws; the three seeds stay together; conditional on the frozen known-task fit.',
        fitting_scope='Supervised auxiliary logistic calibration from32 known tasks only; no language-model weights are updated.',
        fits={},results=[],comparisons=[],source_sha256=hashes)
    cache={}
    for family,stem,method in [('submitted','llama_confirm','original'),('context','extension_confirm','combined')]:
        rows=read(stem)
        train=pd.DataFrame([r for r in rows if r['suite']=='loto' and r['config']==method]).sort_values(['task','seed'])
        assert len(train)==96 and set(train.task)==set(ALL_TASKS)
        assert set(train.seed)=={20,21,22} and not train.duplicated(['task','seed']).any()
        assert np.isfinite(train[['acc','residual','znorm']].to_numpy()).all()
        prior=float((train.acc<.5).mean());models={};serialized={}
        for name,columns in FEATURES:
            fitted=fit(train,columns);models[name]=fitted
            scaler,model,_=fitted
            serialized[name]=dict(columns=columns,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coefficient=model.coef_.tolist() if model else None,
                intercept=model.intercept_.tolist() if model else None,
                classes=model.classes_.tolist() if model else None,
                iterations=model.n_iter_.tolist() if model else None)
        result['fits'][family]=dict(source=stem,method=method,n_tasks=32,n_observations=96,
            failure_count=int((train.acc<.5).sum()),training_prior=prior,models=serialized,
            training_records=[dict(task=r.task,seed=int(r.seed),acc=float(r.acc),
                residual=float(r.residual),znorm=float(r.znorm)) for r in train.itertuples()])
        cache[family]=dict(models=models,prior=prior,
            reference={(r['task'],r['seed']):r for r in rows if r['suite'] in ['novel','compound'] and r['config']==method})
    for stem in ['scale_eval','scale_eval_fixed_shared','scale_context','scale_context_fixed_shared']:
        raw=read(stem);df=pd.DataFrame(raw);family='context' if 'context' in stem else 'submitted';assets=cache[family]
        audit=json.loads((HERE/(stem+'_analysis.json')).read_text())
        assert audit['source_sha256']==hashes[stem] and audit['generation_metrics_verified']
        assert len(df)==600 and not (set(df.task)&set(ALL_TASKS))
        for r in raw:
            if r['size']==32 and r['solver_mode']=='full':
                ref=assets['reference'][r['task'],r['seed']]
                assert r['queries']==ref['queries'] and r['predictions']==ref['predictions']
        for mode in ['full','shortlist32']:
            for size in [32,64,128,256]:
                current=df[(df.solver_mode==mode)&(df['size']==size)].sort_values(['task','seed'])
                assert len(current)==75 and current.groupby('task').size().eq(3).all()
                assert np.isfinite(current[['acc','residual','znorm']].to_numpy()).all()
                y=(current.acc<.5).astype(int).to_numpy()
                values={name:predict(assets['models'][name],current,columns) for name,columns in FEATURES}
                values['constant_prior']=np.full(len(y),assets['prior'])
                indices=np.random.default_rng(20260927).integers(25,size=(50000,25))
                losses={}
                for name,p in values.items():
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    loss=(p-y)**2;task_loss=loss.reshape(25,3).mean(1);losses[name]=task_loss
                    ece=0.
                    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
                        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
                        if mask.any():ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
                    records=[dict(task=str(r.task),seed=int(r.seed),accuracy=float(r.acc),
                        observed_failure=int(t),predicted_failure=float(v)) for r,t,v in zip(current.itertuples(),y,p)]
                    result['results'].append(dict(stem=stem,solver_mode=mode,size=size,features=name,n_tasks=25,n_observations=75,
                        failure_rate=float(y.mean()),mean_risk=float(p.mean()),brier=float(loss.mean()),
                        brier_ci95=np.quantile(task_loss[indices].mean(1),[.025,.975]).tolist(),ece5=float(ece),
                        auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,predictions=records))
                for name,_ in FEATURES:
                    delta=losses[name]-losses['constant_prior']
                    result['comparisons'].append(dict(stem=stem,solver_mode=mode,size=size,features=name,
                        brier_minus_prior=float(delta.mean()),
                        ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()))
    (HERE/'known_task_calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Known-task risk calibration transferred to unseen targets','',
        result['protocol'],'',result['fitting_scope'],'',result['observation_unit'],'',result['uncertainty'],'',
        'All four mappings and every scale/solver condition are retained. '
        'Settings remain fixed at standardized L2 logistic regression, C=1. '
        'The constant predictor uses the known-training failure rate. This '
        'additional protocol does not replace the original eight-task or '
        'leave-one-task-out calibration analyses, and is not blind validation.','',
        '| Operator | Training tasks | Four-example observations | Observed failures | Training prior |',
        '|---|---:|---:|---:|---:|']
    for family,r in result['fits'].items():
        lines.append(f"| {family} | 32 | 96 | {r['failure_count']} | {r['training_prior']:.4f} |")
    for stem in ['scale_eval','scale_eval_fixed_shared','scale_context','scale_context_fixed_shared']:
        lines += ['', '## '+stem,'',
            '| Size | Solver | Features | Brier [95% CI] | ECE5 | AUROC | Mean risk | Failure rate |',
            '|---:|---|---|---:|---:|---:|---:|---:|']
        for r in result['results']:
            if r['stem']!=stem:continue
            lo,hi=r['brier_ci95'];auc='undefined' if r['auc'] is None else f"{r['auc']:.4f}"
            lines.append(f"| {r['size']} | {r['solver_mode']} | {r['features']} | {r['brier']:.4f} [{lo:.4f}, {hi:.4f}] | {r['ece5']:.4f} | {auc} | {r['mean_risk']:.4f} | {r['failure_rate']:.4f} |")
        lines += ['', '| Size | Solver | Features | Brier difference from training-prior predictor [95% CI] |',
                  '|---:|---|---|---:|']
        for r in result['comparisons']:
            if r['stem']!=stem:continue
            lo,hi=r['ci95'];lines.append(f"| {r['size']} | {r['solver_mode']} | {r['features']} | {r['brier_minus_prior']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', 'All source hashes, fitted coefficients/scalers, known-training '
        'records and held-out predictions are retained in `known_task_calibration.json`. '
        'Uncertainty does not include refitting the known-task mapping.','']
    target=ROOT/'rebuttal/2026-09-27-optimized/known_calibration_evidence.md'
    target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
