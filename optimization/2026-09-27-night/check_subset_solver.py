"""Check cached shortlist geometry and solves against a fresh Gram construction."""
import json
import time
import numpy as np
import torch
from common import HERE, ROOT, load_g, make_dict
from fast_solver import GramSolver
from cass.pipeline import z_list_from_Z

torch.set_num_threads(1)
D=make_dict(load_g('llama31-8b',[12,16]),[12,16]);full=GramSolver(D)
rng=np.random.default_rng(20260927);rows=[]
for size in [4,16,24]:
    names=list(rng.choice(D.task_names,size=size,replace=False))
    t=time.perf_counter();cached=full.subset(names);reuse=time.perf_counter()-t
    t=time.perf_counter();fresh=GramSolver(D.subset(names));rebuild=time.perf_counter()-t
    np.testing.assert_array_equal(cached.A,fresh.A)
    np.testing.assert_allclose(cached.H,fresh.H,atol=1e-12,rtol=1e-12)
    for name in ['antonym','english-french','word-length','antonym+capitalize']:
        Z=torch.load(ROOT/'results/llama31-8b/zcache'/f'{name}_k4_s0.pt',weights_only=True).float()
        zl=z_list_from_Z(D,Z);a=cached.solve(zl);b=fresh.solve(zl)
        assert a.support==b.support
        np.testing.assert_allclose(a.delta,b.delta,atol=1e-10,rtol=1e-10)
        rows.append(dict(size=size,task=name,cache_seconds=reuse,rebuild_seconds=rebuild,
                         maximum_delta_error=float(np.max(np.abs(a.delta-b.delta)))))
(HERE/'subset_solver_checks.json').write_text(json.dumps(dict(checks=rows,n_checks=len(rows)),indent=2)+'\n')
print('All',len(rows),'cached-shortlist checks passed')
