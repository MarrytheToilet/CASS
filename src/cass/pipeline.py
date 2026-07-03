"""Shared synthesis pipeline over a MultiLayerDictionary:
z (stacked) -> group-sparse code (shared support across layers)
            -> per-layer injection operators.
"""
import numpy as np

from .dictionary import MultiLayerDictionary
from .solver import solve, least_squares, block_omp, simplex_weights
from .steer import make_additive_op, make_affine_op


def code_for(D, z_list, solver="group_lasso", lam=None):
    """z_list: list of stacked per-example vectors (already deshared)."""
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


def _support_weights(code):
    w = np.array([np.linalg.norm(code.coeffs[n]) for n in code.support])
    return w / w.sum()


def ops_for(mld: MultiLayerDictionary, code, gamma=1.0, beta=2.0,
            alpha_max=1.0, injection="affine", rescale=True, delta_vec=None):
    """Returns (ops, layers) for HookedLM.generate multi-layer injection.

    delta_vec: stacked direction to inject. Default (None) uses the dictionary
    reconstruction code.delta; pass the mean z for the hybrid scheme (denoised
    few-shot direction + dictionary support/affine correction), which is the
    main CASS configuration.

    A single global rescale factor matches the stacked delta norm to the
    support-weighted stacked anchor norm, preserving cross-layer energy ratios.
    """
    delta = (np.asarray(delta_vec, dtype=np.float64) if delta_vec is not None
             else code.delta).copy()
    if rescale and code.support:
        w = _support_weights(code)
        target = float(sum(wi * np.linalg.norm(mld.anchors[n])
                           for wi, n in zip(w, code.support)))
        dn = np.linalg.norm(delta)
        if dn > 1e-8:
            delta *= target / dn

    ops, layers = [], []
    delta_by_layer = mld.split(delta)
    for l in mld.layers:
        dl = delta_by_layer[l]
        if injection == "additive" or not code.support:
            ops.append(make_additive_op(dl, gamma=gamma))
            layers.append(l)
            continue
        Dl = mld.per_layer[l]
        w = _support_weights(code)
        mu_l = sum(wi * Dl.anchors[n] for wi, n in zip(w, code.support))
        B_l = np.concatenate([Dl.bases[n] for n in code.support], axis=1)
        if injection == "projection":
            ops.append(make_affine_op(np.zeros_like(dl), B_l, mu_l, gamma=0.0,
                                      beta=beta, alpha_max=alpha_max))
        else:
            ops.append(make_affine_op(dl, B_l, mu_l, gamma=gamma, beta=beta,
                                      alpha_max=alpha_max))
        layers.append(l)
    return ops, layers


def oracle_ops(mld: MultiLayerDictionary, task_name, gamma=1.0, beta=2.0,
               alpha_max=1.0):
    """Own-subspace affine injection at every layer (the validated oracle)."""
    ops = []
    for l in mld.layers:
        Dl = mld.per_layer[l]
        ops.append(make_affine_op(Dl.anchors[task_name], Dl.bases[task_name],
                                  Dl.anchors[task_name], gamma=gamma,
                                  beta=beta, alpha_max=alpha_max))
    return ops, list(mld.layers)


def naive_ops(mld: MultiLayerDictionary, gamma=1.5):
    """Naive composition baseline: additive mean of all anchors, all layers."""
    ops = []
    for l in mld.layers:
        Dl = mld.per_layer[l]
        delta = np.mean([Dl.anchors[t] for t in mld.task_names], axis=0)
        ops.append(make_additive_op(delta, gamma=gamma))
    return ops, list(mld.layers)


def z_list_from_Z(mld: MultiLayerDictionary, Z):
    """Z: [k, L+1, d] tensor -> list of k stacked, deshared vectors."""
    out = []
    for j in range(Z.shape[0]):
        stacked = mld.stack({l: Z[j, l].numpy() for l in mld.layers})
        out.append(mld.project_out_shared(stacked))
    return out
