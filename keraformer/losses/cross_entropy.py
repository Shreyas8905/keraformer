"""Cross-entropy losses with optional label smoothing and masking."""

from __future__ import annotations

import numpy as np


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - logsumexp


def label_smoothed_cross_entropy(
    logits: np.ndarray,
    target_ids: np.ndarray,
    epsilon: float = 0.1,
    padding_mask: np.ndarray | None = None,
) -> float:
    """Compute label-smoothed cross-entropy over token logits.

    logits: (B, T, V)
    target_ids: (B, T)
    padding_mask: optional binary mask (B, T) where 1 keeps token and 0 drops it.
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")

    x = np.asarray(logits, dtype=np.float32)
    y = np.asarray(target_ids, dtype=np.int64)
    if x.ndim != 3 or y.ndim != 2:
        raise ValueError("logits must be (B, T, V) and targets must be (B, T)")
    if x.shape[:2] != y.shape:
        raise ValueError("batch/sequence dimensions must match")

    bsz, seq, vocab = x.shape
    log_probs = _log_softmax(x, axis=-1)

    one_hot = np.eye(vocab, dtype=np.float32)[y]
    smooth_labels = (1.0 - epsilon) * one_hot + (epsilon / vocab)
    per_token = -np.sum(smooth_labels * log_probs, axis=-1)

    if padding_mask is not None:
        m = np.asarray(padding_mask, dtype=np.float32)
        if m.shape != (bsz, seq):
            raise ValueError("padding_mask must have shape (B, T)")
        denom = np.sum(m) + 1e-9
        return float(np.sum(per_token * m) / denom)

    return float(np.mean(per_token))
