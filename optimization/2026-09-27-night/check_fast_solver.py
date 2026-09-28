"""Check actual cached signatures with unequal block ranks, and timing."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from fast_solver import GramSolver
from common import HERE, ROOT
sys.path.insert(0,str(ROOT/'rebuttal'/'2026-09-27'))
from analyze_evidence import dictionary
from cass.pipeline import z_list_from_Z
from cass.solver import solve_capped

torch.set_num_threads(1)
_,D = dictionary('llama31-8b')
start = time.perf_counter()
solver = GramSolver(D)
build_time = time.perf_counter()-start
rows=[]
for name in ['antonym','country-capital','english-french','present-past','singular-plural',
             'next-item','word-length','person-sport','english-italian','number-plus-one',
             'antonym+capitalize','present-past+capitalize-first-letter']:
    for seed in [0,1,2]:
        p = ROOT/'results'/'llama31-8b'/'zcache'/f'{name}_k4_s{seed}.pt'
        if not p.exists(): continue
        Z=torch.load(p,map_location='cpu',weights_only=True).float()
        zl=z_list_from_Z(D,Z)
        start=time.perf_counter();a=solve_capped(D,zl);original=time.perf_counter()-start
        start=time.perf_counter();b=solver.solve(zl);fast=time.perf_counter()-start
        assert a.support==b.support,(name,seed,a.support,b.support)
        np.testing.assert_allclose(a.delta,b.delta,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(a.residual,b.residual,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(a.lam,b.lam,rtol=1e-12)
        rows.append(dict(task=name,seed=seed,original_seconds=original,fast_seconds=fast,
                         max_delta_error=float(np.max(np.abs(a.delta-b.delta))),support=a.support))
        print(name,seed,round(original,3),round(fast,3),round(original/fast,1),flush=True)
result=dict(offline_cache_seconds=build_time,n_checks=len(rows),checks=rows,
            median_speedup=float(np.median([r['original_seconds']/r['fast_seconds'] for r in rows])))
(HERE/'fast_solver_checks.json').write_text(json.dumps(result,indent=2)+'\n')
