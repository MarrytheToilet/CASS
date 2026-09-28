"""Post-hoc, fixed-rule leave-one-task-out calibration sensitivity analysis."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import HERE
from analyze_scaling import fit,predict,metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stem',required=True);args=ap.parse_args()
    path=HERE/(args.stem+'.jsonl')
    audit=json.loads((HERE/(args.stem+'_analysis.json')).read_text())
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit['generation_metrics_verified'] and audit['source_sha256']==source_hash
    rows=[json.loads(line) for line in path.read_text().splitlines()];df=pd.DataFrame(rows)
    output=dict(source_sha256=source_hash,protocol='Post-hoc fixed-rule leave-one-task-out cross-fitting on25 tasks; C=1, failure<0.5; no label-based setting selection.',
                uncertainty='Paired task bootstrap conditional on cross-fitted predictions; does not refit overlapping calibration models.',
                auc_scope='Pooled leave-one-out probability AUROC is diagnostic only: fold-specific class priors can reverse ranks. Use raw-score AUROC for a common uncalibrated ranking; Brier is the primary calibrated score.',
                results=[],comparisons=[],raw_score_discrimination=[])
    for mode in ['full','shortlist32']:
        averaged=df[df.solver_mode==mode].groupby(['size','task']).agg(
            acc=('acc','mean'),residual=('residual','mean'),znorm=('znorm','mean')).reset_index()
        base=averaged[averaged['size']==32].set_index('task').sort_index()
        for size in [32,64,128,256]:
            current=averaged[averaged['size']==size].set_index('task').sort_index()
            assert list(base.index)==list(current.index) and len(current)==25
            y=(current.acc<.5).astype(int).to_numpy();probabilities={}
            for label,values in [('residual',current.residual.to_numpy()),('negative_norm',-current.znorm.to_numpy())]:
                output['raw_score_discrimination'].append(dict(size=size,mode=mode,signal=label,
                    auc=float(roc_auc_score(y,values)) if len(set(y))==2 else None))
            for label,columns in [('residual',['residual']),('norm',['znorm']),('joint',['residual','znorm']),('constant_prior',[])]:
                for policy in ['frozen32','refit']:
                    train_source=base if policy=='frozen32' else current
                    values=[]
                    for name in current.index:
                        train=train_source.drop(name)
                        if columns:value=float(predict(fit(train,columns),current.loc[[name]],columns)[0])
                        else:value=float((train.acc<.5).mean())
                        values.append(value)
                    p=np.asarray(values);probabilities[label,policy]=p
                    measures=metrics(y,p);measures['pooled_crossfit_auc_diagnostic_only']=measures.pop('auc')
                    output['results'].append(dict(size=size,mode=mode,features=label,policy=policy,
                        predictions={str(n):dict(predicted_failure=float(v),observed_failure=int(t)) for n,v,t in zip(current.index,p,y)},**measures))
            indices=np.random.default_rng(20260927).integers(25,size=(50000,25))
            pairs=[]
            for label in ['residual','norm','joint']:
                pairs.append(((label,'refit'),(label,'frozen32')))
                pairs.extend([((label,p),('constant_prior',p)) for p in ['frozen32','refit']])
            for a,b in pairs:
                difference=(probabilities[a]-y)**2-(probabilities[b]-y)**2
                output['comparisons'].append(dict(size=size,mode=mode,a=list(a),b=list(b),
                    brier_difference=float(difference.mean()),
                    ci95=np.quantile(difference[indices].mean(1),[.025,.975]).tolist()))
    target=HERE/(args.stem+'_crossfit_calibration.json')
    target.write_text(json.dumps(output,indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
