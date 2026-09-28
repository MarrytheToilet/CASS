"""Numerical lower/upper bounds for the corrected weighted two-block condition.

For M=[M1,M2], theta=max_{||v_i||<=1}||M1v1+M2v2||.
Every unit u gives the lower bound ||M1.T u||+||M2.T u||.
For every t in (0,1), sqrt(lambda_max(M1M1.T/t+M2M2.T/(1-t)))
is an upper bound, by weighted Cauchy-Schwarz. Optimization of t only tightens a
valid bound; no local optimum is assumed exact. A numerical margin is recorded.
"""
import argparse
import itertools
import json
import sys
import time
import numpy as np
from scipy.optimize import minimize_scalar
from common import HERE,ROOT
from fast_solver import GramSolver
sys.path.insert(0,str(ROOT/'rebuttal/2026-09-27'))
from analyze_evidence import dictionary
from cass.compound import COMPOUND_REGISTRY,compound_components


def bounds(M1,M2):
    A=M1@M1.T;B=M2@M2.T
    eigen,U=np.linalg.eigh(A+B);u=U[:,-1]
    lower=0.
    for _ in range(40):
        au=A@u;bu=B@u
        p=np.sqrt(max(0.,float(u@au)));q=np.sqrt(max(0.,float(u@bu)))
        lower=max(lower,p+q)
        grad=au/max(p,1e-15)+bu/max(q,1e-15)
        if np.linalg.norm(grad)<1e-15:break
        u=grad/np.linalg.norm(grad)
    # At least the ordinary spectral norm is a valid lower bound as well.
    lower=max(lower,float(np.sqrt(max(0.,eigen[-1]))))
    def upper(t):return float(np.sqrt(max(0.,np.linalg.eigvalsh(A/t+B/(1-t))[-1])))
    opt=minimize_scalar(upper,bounds=(1e-6,1-1e-6),method='bounded',options={'xatol':1e-7})
    value=min(upper(.5),upper(opt.x),float(np.linalg.norm(M1,2)+np.linalg.norm(M2,2)))
    assert lower<=value+1e-7,(lower,value)
    return float(lower),float(value),float(opt.x)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',default='llama31-8b',nargs='?');args=parser.parse_args()
    _,D=dictionary(args.model);solver=GramSolver(D);started=time.time()
    truth={tuple(sorted(compound_components(n))):n for n in COMPOUND_REGISTRY}
    result=dict(model=args.model,margin=1e-6,scope='Numerical geometry check for every two-skill support; '
                'does not assert a generative signal/noise model or certify the support-capped lambda path.',pairs=[])
    for index,S in enumerate(itertools.combinations(D.task_names,2)):
        columns=np.concatenate([np.arange(solver.slices[n].start,solver.slices[n].stop) for n in S])
        G=solver.H[np.ix_(columns,columns)];eig=np.linalg.eigvalsh(G)
        rec=dict(support=S,gram_min_eigenvalue=float(eig[0]),compound=truth.get(tuple(sorted(S))))
        if eig[0]<1e-10:
            rec.update(certified=False,reason='Numerically singular restricted Gram')
        else:
            inverse=np.linalg.inv(G);r1=solver.slices[S[0]].stop-solver.slices[S[0]].start
            w=np.concatenate([np.full(solver.slices[n].stop-solver.slices[n].start,solver.weights[n]) for n in S])
            lower=0.;upper=0.;worst_upper=None;worst_lower=None
            for name in D.task_names:
                if name in S:continue
                M=(solver.H[solver.slices[name],:][:,columns]@inverse)*w[None,:]/solver.weights[name]
                lo,hi,t=bounds(M[:,:r1],M[:,r1:])
                if lo>lower:lower=lo;worst_lower=name
                if hi>upper:upper=hi;worst_upper=name
            rec.update(theta_lower=lower,theta_upper=upper,worst_lower=worst_lower,worst_upper=worst_upper,
                       certified=bool(upper+result['margin']<1.),excluded_by_lower_bound=bool(lower-result['margin']>=1.))
        result['pairs'].append(rec)
        if (index+1)%32==0:
            print(args.model,index+1,'pairs',sum(r['certified'] for r in result['pairs']),'certified',round(time.time()-started,1),flush=True)
            (HERE/(args.model+'_support_specific_bounds.json')).write_text(json.dumps(result,indent=2)+'\n')
    result.update(complete=True,n_pairs=len(result['pairs']),n_certified=sum(r['certified'] for r in result['pairs']),elapsed_seconds=time.time()-started)
    (HERE/(args.model+'_support_specific_bounds.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(args.model,result['n_pairs'],result['n_certified'],'complete',flush=True)


if __name__=='__main__':main()
