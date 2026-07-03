"""Shared synthesis pipeline: z -> sparse code -> injection operator."""
import numpy as np

from .solver import solve, least_squares, block_omp, simplex_weights
from .steer import make_additive_op, make_affine_op


def code_for(D, z_list, solver="group_lasso", lam=None):
    z_list = [np.asarray(z, dtype=np.float64) for z in z_list]
    z = np.mean(z_list, axis=0)
    if solver == "group_lasso":
        return solve(D, z_list, lam=lam)
    if solver == "ls":
        return least_squares(D, z)
    if solver == "omp":
        return block_omp(D, z)
    if solver == "simplex":
        return simplex_weights(D, z)
    raise ValueError(solver)


def rescale_delta(D, code):
    delta = code.delta
    dn = np.linalg.norm(delta)
    if not code.support or dn < 1e-8:
        return delta
    w = np.array([np.linalg.norm(code.coeffs[n]) for n in code.support])
    w = w / w.sum()
    target = float(sum(wi * np.linalg.norm(D.anchors[n])
                       for wi, n in zip(w, code.support)))
    return delta * (target / dn)


def anchor_for(D, code):
    if not code.support:
        return np.zeros_like(code.delta)
    w = np.array([np.linalg.norm(code.coeffs[n]) for n in code.support])
    w = w / w.sum()
    return sum(wi * D.anchors[n] for wi, n in zip(w, code.support))


def op_for(D, code, gamma, beta, alpha_max, injection="affine", rescale=True):
    delta = rescale_delta(D, code) if rescale else code.delta
    if injection == "additive" or not code.support:
        return make_additive_op(delta, gamma=2.0 if injection == "additive"
                                else gamma)
    mu_S = anchor_for(D, code)
    if injection == "projection":
        return make_affine_op(np.zeros_like(delta), code, mu_S, gamma=0.0,
                              beta=beta, alpha_max=alpha_max, dictionary=D)
    return make_affine_op(delta, code, mu_S, gamma=gamma, beta=beta,
                          alpha_max=alpha_max, dictionary=D)
