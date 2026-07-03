"""Skill-subspace dictionary mining.

Steps: (1) shared "task-generic" component U0 from SVD of stacked task means;
(2) project it out of every task's diff activations; (3) truncated SVD per task
-> orthonormal basis U_t (spectral energy >= tau, rank <= r_max) + anchor mu_t.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SkillDictionary:
    task_names: list
    U0: np.ndarray                      # [d, r0] shared component (r0 may be 0)
    bases: dict = field(default_factory=dict)    # name -> U_t [d, r_t]
    anchors: dict = field(default_factory=dict)  # name -> mu_t [d] (deshared)
    spectra: dict = field(default_factory=dict)  # name -> singular values (full)
    raw_means: dict = field(default_factory=dict)  # name -> mean before desharing

    def project_out_shared(self, v: np.ndarray) -> np.ndarray:
        if self.U0.shape[1] == 0:
            return v
        return v - self.U0 @ (self.U0.T @ v)

    def subset(self, names):
        return SkillDictionary(
            task_names=list(names), U0=self.U0,
            bases={n: self.bases[n] for n in names},
            anchors={n: self.anchors[n] for n in names},
            spectra={n: self.spectra[n] for n in names},
            raw_means={n: self.raw_means[n] for n in names},
        )


def shared_component(mean_vectors: np.ndarray, r0: int) -> np.ndarray:
    """mean_vectors: [d, T] stacked task means. Returns U0 [d, r0]."""
    d = mean_vectors.shape[0]
    if r0 == 0:
        return np.zeros((d, 0), dtype=np.float64)
    U, _, _ = np.linalg.svd(mean_vectors, full_matrices=False)
    return U[:, :r0]


def rank_by_energy(s: np.ndarray, tau: float, r_max: int) -> int:
    e = np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-12)
    r = int(np.searchsorted(e, tau) + 1)
    return max(1, min(r, r_max, len(s)))


def build_dictionary(G_by_task: dict, r0=2, tau=0.90, r_max=16) -> SkillDictionary:
    """G_by_task: name -> [n, d] diff activations at the chosen layer."""
    names = list(G_by_task)
    means = np.stack([G_by_task[n].mean(0) for n in names], axis=1).astype(np.float64)
    U0 = shared_component(means, r0)

    D = SkillDictionary(task_names=names, U0=U0)
    for i, name in enumerate(names):
        G = G_by_task[name].astype(np.float64).T          # [d, n]
        G_t = G - U0 @ (U0.T @ G) if r0 > 0 else G
        U, s, _ = np.linalg.svd(G_t, full_matrices=False)
        r = rank_by_energy(s, tau, r_max)
        D.bases[name] = U[:, :r]
        D.anchors[name] = G_t.mean(1)
        D.spectra[name] = s
        D.raw_means[name] = means[:, i]
    return D


class MultiLayerDictionary:
    """Joint dictionary over several layers: per task, the stacked basis is
    block-diagonal over layers ([K*d, sum_l r_tl], orthonormal columns), so the
    group-LASSO solver applies unchanged and the support is shared across
    layers while coefficients stay layer-specific."""

    def __init__(self, layer_dicts: dict):
        self.layers = sorted(layer_dicts)
        self.per_layer = layer_dicts
        first = layer_dicts[self.layers[0]]
        self.task_names = list(first.task_names)
        self.d = first.U0.shape[0]
        K = len(self.layers)
        self.bases, self.anchors = {}, {}
        for n in self.task_names:
            blocks = [layer_dicts[l].bases[n] for l in self.layers]
            rs = [b.shape[1] for b in blocks]
            U = np.zeros((K * self.d, sum(rs)))
            c0 = 0
            for i, b in enumerate(blocks):
                U[i * self.d:(i + 1) * self.d, c0:c0 + b.shape[1]] = b
                c0 += b.shape[1]
            self.bases[n] = U
            self.anchors[n] = np.concatenate(
                [layer_dicts[l].anchors[n] for l in self.layers])

    def stack(self, z_by_layer: dict) -> np.ndarray:
        return np.concatenate([np.asarray(z_by_layer[l], dtype=np.float64)
                               for l in self.layers])

    def split(self, v: np.ndarray) -> dict:
        return {l: v[i * self.d:(i + 1) * self.d]
                for i, l in enumerate(self.layers)}

    def project_out_shared(self, v: np.ndarray) -> np.ndarray:
        parts = [self.per_layer[l].project_out_shared(p)
                 for l, p in self.split(v).items()]
        return np.concatenate(parts)

    def subset(self, names):
        return MultiLayerDictionary(
            {l: D.subset(names) for l, D in self.per_layer.items()})


def build_multilayer_dictionary(G_by_task_by_layer: dict, r0=2, tau=0.90,
                                r_max=16) -> MultiLayerDictionary:
    """G_by_task_by_layer: layer -> {task: [n, d]}."""
    return MultiLayerDictionary(
        {l: build_dictionary(G, r0=r0, tau=tau, r_max=r_max)
         for l, G in G_by_task_by_layer.items()})


# ---------- diagnostics ----------

def pairwise_cosine(vectors: dict) -> np.ndarray:
    names = list(vectors)
    V = np.stack([vectors[n] for n in names])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V @ V.T


def block_coherence(D: SkillDictionary) -> float:
    """mu_B = max over task pairs of largest principal-angle cosine."""
    return subcoherence_matrix(D).max() if len(D.task_names) > 1 else 0.0


def subcoherence_matrix(D: SkillDictionary) -> np.ndarray:
    names = D.task_names
    T = len(names)
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            s = np.linalg.svd(D.bases[names[i]].T @ D.bases[names[j]],
                              compute_uv=False)
            M[i, j] = M[j, i] = s[0]
    return M
