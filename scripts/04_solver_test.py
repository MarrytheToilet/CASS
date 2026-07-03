"""Synthetic unit tests for the group-LASSO solver (support recovery)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cass.dictionary import SkillDictionary
from cass.solver import (group_lasso, lambda_max, select_lambda_loo, solve,
                         least_squares, block_omp, simplex_weights)


def random_dictionary(d=512, T=20, r=4, rng=None):
    rng = rng or np.random.default_rng(0)
    names = [f"t{i}" for i in range(T)]
    D = SkillDictionary(task_names=names, U0=np.zeros((d, 0)))
    for n in names:
        A = rng.standard_normal((d, r))
        Q, _ = np.linalg.qr(A)
        D.bases[n] = Q
        D.anchors[n] = Q @ rng.standard_normal(r)
        D.spectra[n] = np.ones(r)
        D.raw_means[n] = D.anchors[n]
    return D


def test_support_recovery(noise=0.05, k0=2, trials=20):
    rng = np.random.default_rng(42)
    hits, exact = 0, 0
    for _ in range(trials):
        D = random_dictionary(rng=rng)
        S0 = list(rng.choice(D.task_names, k0, replace=False))
        z = np.zeros(512)
        for n in S0:
            z += D.bases[n] @ rng.standard_normal(4)
        z += noise * np.linalg.norm(z) * rng.standard_normal(512) / np.sqrt(512)
        code = group_lasso(D, z, lam=0.2 * lambda_max(D, z))
        if set(S0) <= set(code.support):
            hits += 1
        if set(S0) == set(code.support):
            exact += 1
    print(f"support recovery (k0={k0}, noise={noise}): "
          f"contains true {hits}/{trials}, exact {exact}/{trials}")
    assert hits >= trials * 0.9


def test_loo_lambda():
    rng = np.random.default_rng(7)
    D = random_dictionary(rng=rng)
    S0 = ["t3", "t11"]
    z_list = []
    base = {n: rng.standard_normal(4) for n in S0}
    for _ in range(4):
        z = sum(D.bases[n] @ (base[n] + 0.15 * rng.standard_normal(4)) for n in S0)
        z += 0.05 * np.linalg.norm(z) * rng.standard_normal(512) / np.sqrt(512)
        z_list.append(z)
    lam = select_lambda_loo(D, z_list)
    code = solve(D, z_list, lam=lam)
    print(f"LOO lambda={lam:.3f} support={code.support} residual={code.residual:.3f}")
    assert set(S0) <= set(code.support) and len(code.support) <= 6


def test_baselines():
    rng = np.random.default_rng(3)
    D = random_dictionary(rng=rng)
    z = D.bases["t5"] @ rng.standard_normal(4)
    ls = least_squares(D, z)
    omp = block_omp(D, z, max_support=3)
    sx = simplex_weights(D, z)
    print(f"LS residual={ls.residual:.4f} | OMP support={omp.support} "
          f"residual={omp.residual:.4f} | simplex residual={sx.residual:.3f}")
    assert omp.support[0] == "t5" and omp.residual < 1e-6 and ls.residual < 1e-6


def test_convergence_speed():
    import time
    rng = np.random.default_rng(1)
    D = random_dictionary(d=4096, T=31, r=8, rng=rng)
    z = sum(D.bases[n] @ rng.standard_normal(8) for n in ["t1", "t2"])
    t0 = time.time()
    code = group_lasso(D, z, lam=0.2 * lambda_max(D, z))
    print(f"d=4096 T=31 solve: {(time.time()-t0)*1000:.1f} ms, "
          f"support={code.support}")


if __name__ == "__main__":
    test_support_recovery()
    test_support_recovery(noise=0.2)
    test_support_recovery(k0=3)
    test_loo_lambda()
    test_baselines()
    test_convergence_speed()
    print("ALL SOLVER TESTS PASSED")
