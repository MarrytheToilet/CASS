"""Predeclared held-out-task residual calibration and support stability."""
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.preprocessing import StandardScaler
from common import HERE
from cass.evaluate import accuracy
from robust_metrics import score as robust_score

CALIBRATION_TASKS=['country-continent','animal-baby','verb-gerund','english-italian','word-last-letter',
                   'antonym+capitalize','present-past+capitalize','present-past+capitalize-first-letter']


def fit(train,columns):
    x=train[columns].to_numpy();y=(train.acc<.5).astype(int).to_numpy()
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1.,solver='lbfgs',random_state=20260927).fit(scaler.transform(x),y) if len(set(y))==2 else None
    return scaler,model,float(y.mean())


def predict(fitted,test,columns):
    scaler,model,prior=fitted
    return model.predict_proba(scaler.transform(test[columns].to_numpy()))[:,1] if model else np.full(len(test),prior)


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);ece=0.
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any():ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    bootstrap=np.random.default_rng(20260927).integers(0,len(y),(50000,len(y)))
    losses=(p-y)**2
    brier_ci=np.quantile(losses[bootstrap].mean(1),[.025,.975]).tolist()
    return dict(n_tasks=len(y),failure_rate=float(y.mean()),mean_risk=float(p.mean()),brier_ci95=brier_ci,
                brier=float(brier_score_loss(y,p)),ece5=float(ece),
                auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stem',default='scale_eval');args=parser.parse_args()
    records=[json.loads(s) for s in (HERE/(args.stem+'.jsonl')).read_text().splitlines()]
    for r in records:
        assert abs(accuracy(r['predictions'],[y for x,y in r['queries']],case_sensitive='+' in r['task'])-r['acc'])<1e-12
        r['acc_literal']=float(np.mean([robust_score(p,y,r['task'],'task_case_literal_line') for p,(x,y) in zip(r['predictions'],r['queries'])]))
    df=pd.DataFrame(records)
    assert len(df)==4*25*3*2 and not df.duplicated(['task','seed','config']).any()
    for (task,seed),group in df.groupby(['task','seed']):
        queries=group.iloc[0]['queries']
        assert all(row['queries']==queries for _,row in group.iterrows()),(task,seed,'query mismatch')
    summary=dict(source_sha256=hashlib.sha256((HERE/(args.stem+'.jsonl')).read_bytes()).hexdigest(),
                 protocol=dict(calibration_tasks=CALIBRATION_TASKS,heldout_tasks=17,
                               failure_definition='task-mean accuracy across three seeds < 0.5',
                               calibration='L2 logistic regression, C=1, fitted to eight disjoint tasks',
                               probability_metrics='Brier primary; five-bin ECE diagnostic; raw residual is not a probability',
                               scope='Auxiliary task-level calibration fit, separate from the frozen-backbone steering operator. '
                                     'Its eight calibration-task labels are declared, and no held-out-task label is used for fitting.'),
                 generation_metrics_verified=True,matched_queries_verified=True,
                 scaling=[],calibration=[],calibration_comparisons=[])
    for mode in ['full','shortlist32']:
        current=df[df.solver_mode==mode]
        averaged=current.groupby(['size','task']).agg(acc=('acc','mean'),acc_literal=('acc_literal','mean'),residual=('residual','mean'),
                    znorm=('znorm','mean'),recall=('constituent_recall','mean')).reset_index()
        base=averaged[averaged['size']==32].set_index('task')
        fitted32={label:fit(base.loc[CALIBRATION_TASKS],columns) for label,columns in
                  [('residual',['residual']),('norm',['znorm']),('joint',['residual','znorm'])]}
        support32={(r['task'],r['seed']):set(r['support']) for r in records if r['size']==32 and r['solver_mode']==mode}
        for size in [32,64,128,256]:
            part=current[current['size']==size];means=averaged[averaged['size']==size].set_index('task')
            jaccard=[];within=[]
            for _,row in part.iterrows():
                a=set(row.support);b=support32[(row.task,row.seed)]
                jaccard.append(len(a&b)/len(a|b) if a|b else 1.)
            for name,group in part.groupby('task'):
                supports=[set(x) for x in group.support]
                for i in range(3):
                    for j in range(i):
                        union=supports[i]|supports[j]
                        within.append(len(supports[i]&supports[j])/len(union) if union else 1.)
            delta=means.acc-base.acc
            indices=np.random.default_rng(20260927).integers(len(delta),size=(50000,len(delta)))
            delta_literal=means.acc_literal-base.acc_literal
            summary['scaling'].append(dict(size=size,mode=mode,accuracy=float(means.acc.mean()),
                paired_accuracy_change_from32=float(delta.mean()),
                paired_accuracy_change_ci95=np.quantile(delta.to_numpy()[indices].mean(1),[.025,.975]).tolist(),
                paired_literal_change_from32=float(delta_literal.mean()),
                paired_literal_change_ci95=np.quantile(delta_literal.to_numpy()[indices].mean(1),[.025,.975]).tolist(),
                literal_first_line_accuracy=float(means.acc_literal.mean()),
                accuracy_by_suite={suite:float(group.groupby('task').acc.mean().mean()) for suite,group in part.groupby('suite')},
                literal_accuracy_by_suite={suite:float(group.groupby('task').acc_literal.mean().mean()) for suite,group in part.groupby('suite')},
                failure_label_changes_under_literal_metric=int(((means.acc<.5)!=(means.acc_literal<.5)).sum()),
                compound_constituent_recall=float(means.recall.dropna().mean()),
                support_jaccard_to32=float(np.mean(jaccard)),support_jaccard_across_seeds=float(np.mean(within)),
                residual_mean=float(means.residual.mean()),coding_median_seconds=float(part.coding_seconds.median())))
            train=means.loc[CALIBRATION_TASKS];test=means.drop(CALIBRATION_TASKS);y=(test.acc<.5).astype(int)
            probability={}
            for label,columns in [('residual',['residual']),('norm',['znorm']),('joint',['residual','znorm'])]:
                for policy,fitted in [('frozen32',fitted32[label]),('refit',fit(train,columns))]:
                    p=predict(fitted,test,columns)
                    probability[(label,policy)]=p
                    summary['calibration'].append(dict(size=size,mode=mode,features=label,policy=policy,
                        predictions={str(n):dict(predicted_failure=float(v),observed_failure=int(y.loc[n])) for n,v in zip(test.index,p)},**metrics(y,p)))
            for policy,train_for_prior in [('frozen32',base.loc[CALIBRATION_TASKS]),('refit',train)]:
                prior=float((train_for_prior.acc<.5).mean())
                p=np.full(len(y),prior);probability[('constant_prior',policy)]=p
                summary['calibration'].append(dict(size=size,mode=mode,features='constant_prior',policy=policy,
                    predictions={str(n):dict(predicted_failure=float(v),observed_failure=int(y.loc[n])) for n,v in zip(test.index,p)},**metrics(y,p)))
            contrasts=[]
            for label in ['residual','norm','joint']:
                contrasts.append(((label,'refit'),(label,'frozen32')))
                contrasts.extend([((label,policy),('constant_prior',policy)) for policy in ['frozen32','refit']])
            indices=np.random.default_rng(20260927).integers(len(y),size=(50000,len(y)))
            for a,b in contrasts:
                delta=(probability[a]-y.to_numpy())**2-(probability[b]-y.to_numpy())**2
                summary['calibration_comparisons'].append(dict(size=size,mode=mode,a=list(a),b=list(b),
                    n_tasks=len(y),brier_difference=float(delta.mean()),ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
                    interpretation='Paired held-out-task Brier difference; negative favors the first method. Conditional on eight calibration tasks.'))
    (HERE/(args.stem+'_analysis.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['scaling'],indent=2))


if __name__=='__main__':main()
