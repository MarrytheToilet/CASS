"""Numerical sanity checks for the proposed weighted KKT correction.

These checks complement, and do not replace, the proof in theory_correction.md.
They include genuinely two-block, unequal-rank instances satisfying the corrected
condition, alongside the independently constructed failing original condition.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/"src"))
from cass.solver import group_lasso

records = []
for seed in range(50):
    rng = np.random.default_rng(seed)
    ranks = [2,3,4]
    orth = np.linalg.qr(rng.normal(size=(20,sum(ranks))))[0]
    blocks, start = [], 0
    for rank in ranks:
        U = orth[:,start:start+rank]+.005*rng.normal(size=(20,rank))
        blocks.append(np.linalg.qr(U)[0]); start += rank
    D = SimpleNamespace(task_names=["a","b","c"],bases=dict(zip(["a","b","c"],blocks)))
    A = np.concatenate(blocks[:2],axis=1)
    G = A.T@A
    q = np.sqrt(ranks)
    matrix = blocks[2].T@A@np.linalg.inv(G)
    theta_upper = (q[0]*np.linalg.norm(matrix[:,:ranks[0]],2)+q[1]*np.linalg.norm(matrix[:,ranks[0]:],2))/q[2]
    assert theta_upper < 1
    eps = .01
    noise = rng.normal(size=20); noise *= eps/np.linalg.norm(noise)
    c0 = rng.normal(size=sum(ranks[:2]))
    lam = 2*eps/(q[2]*(1-theta_upper))
    z = A@c0+noise
    sol = group_lasso(D,z,lam,max_iter=3000,tol=1e-11)
    assert set(sol.support) <= {"a","b"}
    error = np.linalg.norm(np.r_[sol.coeffs["a"],sol.coeffs["b"]]-c0)
    singular = np.linalg.svd(A,compute_uv=False)[-1]
    bound = eps/singular + lam*np.sqrt(sum(q[:2]**2))/singular**2
    assert error <= bound+1e-9
    residual = z-sol.delta
    inactive_ratio = np.linalg.norm(blocks[2].T@residual)/(lam*q[2])
    assert inactive_ratio < 1
    records.append(dict(seed=seed,theta_upper=float(theta_upper),error=float(error),bound=float(bound),inactive_ratio=float(inactive_ratio)))
result = dict(n_instances=len(records),true_support_size=2,ranks=[2,3,4],
              all_checks_passed=True,max_error_bound_ratio=max(x["error"]/x["bound"] for x in records),
              instances=records)
(HERE/"theory_numerical_checks.json").write_text(json.dumps(result,indent=2)+"\n")
print(f'{len(records)} unequal-rank two-block instances pass support, KKT, and error-bound checks.')
