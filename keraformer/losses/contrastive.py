"""Contrastive losses (NT-Xent / SimCLR)."""

from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def nt_xent_loss(z_i: np.ndarray, z_j: np.ndarray, temperature: float = 0.07) -> float:
    """Compute NT-Xent loss for paired embeddings.

    z_i, z_j: (N, D) augmented view embeddings.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    zi = _l2_normalize(np.asarray(z_i, dtype=np.float32))
    zj = _l2_normalize(np.asarray(z_j, dtype=np.float32))
    if zi.shape != zj.shape or zi.ndim != 2:
        raise ValueError("z_i and z_j must have identical shape (N, D)")

    z = np.concatenate([zi, zj], axis=0)
    sim = np.matmul(z, z.T) / temperature

    n = zi.shape[0]
    mask = np.eye(2 * n, dtype=bool)
    sim_masked = np.where(mask, -1e9, sim)

    positives = np.concatenate([np.diag(sim, n), np.diag(sim, -n)], axis=0)

    logits_max = np.max(sim_masked, axis=1, keepdims=True)
    logits = sim_masked - logits_max
    exp_logits = np.exp(logits)
    denom = np.sum(exp_logits, axis=1)

    pos_exp = np.exp(positives - logits_max.squeeze(-1))
    loss = -np.log(pos_exp / (denom + 1e-9) + 1e-9)
    return float(np.mean(loss))
