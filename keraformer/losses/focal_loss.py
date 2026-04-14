"""Focal loss implementation."""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def focal_loss(
    logits: np.ndarray,
    target_ids: np.ndarray,
    gamma: float = 2.0,
    alpha: float | np.ndarray = 1.0,
) -> float:
    """Multi-class focal loss.

    logits: (..., C)
    target_ids: (...,)
    alpha: scalar or per-class array of shape (C,)
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")

    x = np.asarray(logits, dtype=np.float32)
    y = np.asarray(target_ids, dtype=np.int64)
    if x.shape[:-1] != y.shape:
        raise ValueError("target shape must match logits without class dimension")

    probs = _softmax(x, axis=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_targets = y.reshape(-1)

    p_t = flat_probs[np.arange(flat_targets.shape[0]), flat_targets]
    p_t = np.clip(p_t, 1e-9, 1.0)

    if np.isscalar(alpha):
        alpha_t = float(alpha)
    else:
        alpha_arr = np.asarray(alpha, dtype=np.float32)
        if alpha_arr.shape != (probs.shape[-1],):
            raise ValueError("alpha array must have shape (num_classes,)")
        alpha_t = alpha_arr[flat_targets]

    loss = -alpha_t * ((1.0 - p_t) ** gamma) * np.log(p_t)
    return float(np.mean(loss))
