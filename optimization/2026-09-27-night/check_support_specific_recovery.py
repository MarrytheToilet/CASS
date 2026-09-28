"""Check planted two-block signals satisfying the corrected theorem assumptions.

These are numerical sparse-recovery checks, not language-model task accuracy.
Every pair passing the precomputed upper bound is included, across all five
original dictionaries; no pair is selected by its recovery outcome.
"""
import hashlib
import json
import sys
import numpy as np
from common import HERE,ROOT
from fast_solver import GramSolver
sys.path.insert(0,str(ROOT/'rebuttal/2026-09-27'))
from analyze_evidence import dictionary

records=[];sources={}
for model in ['llama31-8b','llama32-3b','gemma2-2b','qwen3-4b','qwen25-3b']:
    path=HERE/(model+'_support_specific_bounds.json');source=json.loads(path.read_text())
    assert source['complete'] and source['n_pairs']==496
    sources[model]=hashlib.sha256(path.read_bytes()).hexdigest()
    selected=[r for r in source['pairs'] if r['certified']]
    if not selected:continue
    _,D=dictionary(model);solver=GramSolver(D)
    for index,rec in enumerate(selected):
        S=rec['support'];theta=rec['theta_upper']+source['margin']
        qmin=min(solver.weights[n] for n in D.task_names if n not in S)
        epsilon=1e-5;lam=2*epsilon/(qmin*(1-theta))
        sigma=np.sqrt(rec['gram_min_eigenvalue'])
        bound=epsilon/sigma+lam*np.sqrt(sum(solver.weights[n]**2 for n in S))/sigma**2
        assert bound<1.,(model,S,bound)
        for seed in [0,1,2]:
            rng=np.random.default_rng(61000+index*3+seed)
            c={n:rng.normal(size=D.bases[n].shape[1]) for n in S}
            c={n:v/np.linalg.norm(v) for n,v in c.items()}
            e=rng.normal(size=solver.A.shape[0]);e*=epsilon/np.linalg.norm(e)
            z=sum(D.bases[n]@c[n] for n in S)+e
            fitted=solver.fit(z,lam,max_iter=1500,tol=1e-10)
            assert set(fitted.support)==set(S),(model,S,fitted.support)
            error=np.sqrt(sum(np.linalg.norm(fitted.coeffs[n]-c[n])**2 for n in S))
            assert error<=bound+1e-8
            records.append(dict(model=model,support=S,seed=seed,theta_upper=theta,lambda_value=lam,
                                epsilon=epsilon,error_bound=float(bound),actual_error=float(error),
                                exact_support=True))
    print(model,len(selected),'eligible supports, all planted checks passed',flush=True)
(HERE/'support_specific_recovery_checks.json').write_text(json.dumps(dict(
    scope='Planted union-of-subspaces signals with known coefficients and bounded noise; '
          'not evidence of semantic composition or recovery of observed task signatures.',
    sources=sources,checks=records,n_checks=len(records)),indent=2)+'\n')
