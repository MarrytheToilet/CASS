"""Group-sparse coding over the skill dictionary.

Main solver: group LASSO by block coordinate descent with exact group
soft-threshold updates (valid because each U_t is orthonormal).
Baselines: least squares, block OMP, simplex-weighted anchors.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SparseCode:
    coeffs: dict           # name -> c_t [r_t]
    support: list          # names with ||c_t|| > 0
    delta: np.ndarray      # synthesized vector sum U_t c_t
    residual: float        # ||z - delta|| / ||z||
    lam: float


def _synth(D, coeffs):
    d = D.U0.shape[0]
    v = np.zeros(d)
    for n, c in coeffs.items():
        if np.any(c):
            v += D.bases[n] @ c
    return v


def group_lasso(D, z, lam, max_iter=200, tol=1e-6, rng=None) -> SparseCode:
    names = D.task_names
    rng = rng or np.random.default_rng(0)
    z = z.astype(np.float64)
    c = {n: np.zeros(D.bases[n].shape[1]) for n in names}
    r = z.copy()                     # residual z - sum U_t c_t, kept incrementally
    for _ in range(max_iter):
        max_rel = 0.0
        for n in rng.permutation(names):
            U = D.bases[n]
            r_full = r + U @ c[n]    # residual with block n removed
            b = U.T @ r_full
            nb = np.linalg.norm(b)
            thr = lam * np.sqrt(U.shape[1])
            c_new = np.zeros_like(b) if nb <= thr else (1 - thr / nb) * b
            max_rel = max(max_rel, np.linalg.norm(c_new - c[n]) /
                          (np.linalg.norm(c[n]) + 1e-8))
            r = r_full - U @ c_new
            c[n] = c_new
        if max_rel < tol:
            break
    delta = _synth(D, c)
    support = [n for n in names if np.linalg.norm(c[n]) > 1e-10]
    res = np.linalg.norm(z - delta) / (np.linalg.norm(z) + 1e-12)
    return SparseCode(c, support, delta, res, lam)


def lambda_max(D, z) -> float:
    return max(np.linalg.norm(D.bases[n].T @ z) / np.sqrt(D.bases[n].shape[1])
               for n in D.task_names)


def lambda_path(D, z, n_points=20, floor=0.01):
    lmax = lambda_max(D, z)
    return np.geomspace(lmax * 0.99, lmax * floor, n_points)


def select_lambda_loo(D, z_list, n_points=20, k1_frac=0.3):
    """Leave-one-example-out lambda selection.

    z_list: per-example diff vectors (already deshared). For each candidate
    lambda, fit on the mean of k-1 examples, score the residual on the held-out
    one; pick the lambda with lowest mean held-out residual.
    k=1 falls back to fixed lam = k1_frac * lambda_max.
    """
    z_list = [z.astype(np.float64) for z in z_list]
    z_mean = np.mean(z_list, axis=0)
    if len(z_list) == 1:
        return k1_frac * lambda_max(D, z_mean)
    path = lambda_path(D, z_mean, n_points)
    scores = np.zeros(len(path))
    for j, z_out in enumerate(z_list):
        z_fit = np.mean([z for i, z in enumerate(z_list) if i != j], axis=0)
        for li, lam in enumerate(path):
            code = group_lasso(D, z_fit, lam)
            zo = z_out / (np.linalg.norm(z_out) + 1e-12)
            de = code.delta / (np.linalg.norm(code.delta) + 1e-12) \
                if np.linalg.norm(code.delta) > 0 else code.delta
            scores[li] += np.linalg.norm(zo - de)
    return float(path[int(np.argmin(scores))])


def solve(D, z_list, lam=None) -> SparseCode:
    """Full pipeline: lambda by LOO, then group LASSO on the mean vector."""
    z = np.mean([np.asarray(zz, dtype=np.float64) for zz in z_list], axis=0)
    if lam is None:
        lam = select_lambda_loo(D, z_list)
    return group_lasso(D, z, lam)


# ---------- baseline solvers (ablations) ----------

def least_squares(D, z) -> SparseCode:
    names = D.task_names
    A = np.concatenate([D.bases[n] for n in names], axis=1)
    x, *_ = np.linalg.lstsq(A, z.astype(np.float64), rcond=None)
    c, i = {}, 0
    for n in names:
        r = D.bases[n].shape[1]
        c[n] = x[i:i + r]
        i += r
    delta = _synth(D, c)
    res = np.linalg.norm(z - delta) / (np.linalg.norm(z) + 1e-12)
    return SparseCode(c, list(names), delta, res, 0.0)


def block_omp(D, z, max_support=3) -> SparseCode:
    z = z.astype(np.float64)
    names = list(D.task_names)
    support, r = [], z.copy()
    for _ in range(max_support):
        gains = {n: np.linalg.norm(D.bases[n].T @ r) for n in names
                 if n not in support}
        best = max(gains, key=gains.get)
        support.append(best)
        A = np.concatenate([D.bases[n] for n in support], axis=1)
        x, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ x
    c, i = {n: np.zeros(D.bases[n].shape[1]) for n in D.task_names}, 0
    for n in support:
        rk = D.bases[n].shape[1]
        c[n] = x[i:i + rk]
        i += rk
    delta = _synth(D, c)
    res = np.linalg.norm(z - delta) / (np.linalg.norm(z) + 1e-12)
    return SparseCode(c, support, delta, res, 0.0)


def simplex_weights(D, z, iters=500, lr=0.1) -> SparseCode:
    """Convex combination of task anchor vectors (scalar weight per task)."""
    z = z.astype(np.float64)
    names = D.task_names
    M = np.stack([D.anchors[n] for n in names], axis=1)   # [d, T]
    T = M.shape[1]
    w = np.ones(T) / T
    for _ in range(iters):
        g = M.T @ (M @ w - z)
        w = w - lr * g / (np.linalg.norm(M, ord=2) ** 2 + 1e-12)
        w = _project_simplex(w)
    delta = M @ w
    c = {n: np.array([w[i]]) for i, n in enumerate(names)}
    support = [n for i, n in enumerate(names) if w[i] > 1e-4]
    res = np.linalg.norm(z - delta) / (np.linalg.norm(z) + 1e-12)
    return SparseCode(c, support, delta, res, 0.0)


def _project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)
