"""Injection operators applied to the last-token residual stream at layer l*.

additive:  h + gamma * delta
affine:    h + alpha(h) * [gamma * delta + P_S (mu_S - h)]
           alpha(h) = min(alpha_max, beta * ||(I-P_S)(h - mu_S)|| / ||h||)
"""
import numpy as np
import torch


def _to_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def make_additive_op(delta, gamma=1.0, device="cuda"):
    dvec = _to_torch(delta, device)

    def op(h):  # h: [B, d]
        return h.float() + gamma * dvec
    return op


def make_affine_op(delta, code_or_basis, anchor, gamma=1.0, beta=4.0,
                   alpha_max=2.0, device="cuda", dictionary=None):
    """code_or_basis: either a SparseCode (uses dictionary bases on its support)
    or a [d, r] ndarray basis. anchor: mu_S [d]."""
    if hasattr(code_or_basis, "support"):
        assert dictionary is not None
        cols = [dictionary.bases[n] for n in code_or_basis.support]
        if not cols:
            return make_additive_op(delta, gamma, device)
        B = np.concatenate(cols, axis=1)
    else:
        B = np.asarray(code_or_basis)
    Q, _ = np.linalg.qr(B)
    Qt = _to_torch(Q, device)                     # [d, r]
    dvec = _to_torch(delta, device)
    mu = _to_torch(anchor, device)

    def op(h):  # h: [B, d]
        h = h.float()
        diff = mu.unsqueeze(0) - h                # [B, d]
        proj = (diff @ Qt) @ Qt.T                 # P_S (mu - h)
        ortho = diff - proj                       # (I-P_S)(h-mu) up to sign
        alpha = beta * ortho.norm(dim=1) / (h.norm(dim=1) + 1e-8)
        alpha = alpha.clamp(max=alpha_max).unsqueeze(1)
        return h + alpha * (gamma * dvec.unsqueeze(0) + proj)
    return op


def make_projection_only_op(code_or_basis, anchor, beta=4.0, alpha_max=2.0,
                            device="cuda", dictionary=None):
    """Ablation: subspace correction without the direction term."""
    d = np.zeros_like(np.asarray(anchor))
    return make_affine_op(d, code_or_basis, anchor, gamma=0.0, beta=beta,
                          alpha_max=alpha_max, device=device, dictionary=dictionary)
