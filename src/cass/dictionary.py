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
