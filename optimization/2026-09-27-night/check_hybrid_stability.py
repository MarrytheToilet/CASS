"""Numerical sanity checks of the direct operator bound, across clipping regimes."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(20260927);checks=[]
for i in range(1200):
    dim=12;h=rng.normal(size=dim)*10**rng.uniform(-2,2)
    mu=rng.normal(size=dim);ms=mu+rng.normal(size=dim)*10**rng.uniform(-4,1)
    d=rng.normal(size=dim);ds=d+rng.normal(size=dim)*10**rng.uniform(-4,1)
    rank=int(rng.integers(0,dim+1));ranks=int(rng.integers(0,dim+1))
    q=np.linalg.qr(rng.normal(size=(dim,dim)))[0][:,:rank]
    qs=np.linalg.qr(rng.normal(size=(dim,dim)))[0][:,:ranks]
    P=q@q.T;Ps=qs@qs.T;g=float(rng.random());gs=float(rng.random());beta=10**rng.uniform(-2,1);gamma=10**rng.uniform(-2,1);eta=1e-8
    if i%3==0:Ps=P.copy();gs=g
    if i%5==0:g=0.;gs=1.
    def operator(direction,anchor,projector,gate):
        alpha=min(1.,beta*np.linalg.norm((np.eye(dim)-projector)@(h-anchor))/(np.linalg.norm(h)+eta))
        return h+gamma*(1-gate+gate*alpha)*direction+gate*alpha*projector@(anchor-h),alpha
    out,a=operator(d,mu,P,g);ref,astar=operator(ds,ms,Ps,gs)
    B=np.linalg.norm(mu-ms)+np.linalg.norm(ms-h)*np.linalg.norm(P-Ps,2)
    bound=gamma*np.linalg.norm(d-ds)+B+(gamma*np.linalg.norm(ds)+np.linalg.norm(ms-h))*(abs(g-gs)+beta*B/(np.linalg.norm(h)+eta))
    error=np.linalg.norm(out-ref);assert error<=bound+1e-10,(i,error,bound)
    checks.append(dict(error=float(error),bound=float(bound),alpha=a,reference_alpha=astar))
result=dict(n_instances=len(checks),all_pass=True,max_bound_ratio=max(r['error']/r['bound'] for r in checks),
            unclipped_instances=int(sum(r['alpha']<1 and r['reference_alpha']<1 for r in checks)),
            mixed_clipping_instances=int(sum((r['alpha']<1)!=(r['reference_alpha']<1) for r in checks)),
            note='Numerical checks complement the analytic proof; they are not a proof or an empirical performance guarantee.')
(Path(__file__).resolve().parent/'hybrid_stability_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
