"""Masked language modeling loss."""

from __future__ import annotations

import numpy as np

from .cross_entropy import label_smoothed_cross_entropy


def masked_lm_loss(
    logits: np.ndarray,
    target_ids: np.ndarray,
    mask_positions: np.ndarray,
    epsilon: float = 0.0,
) -> float:
    """Compute CE over masked positions only.

    logits: (B, T, V)
    target_ids: (B, T)
    mask_positions: boolean or {0,1} mask of shape (B, T)
    """
    m = np.asarray(mask_positions)
    if m.shape != np.asarray(target_ids).shape:
        raise ValueError("mask_positions must match target_ids shape")

    binary = (m > 0).astype(np.float32)
    if np.sum(binary) == 0:
        return 0.0

    return label_smoothed_cross_entropy(
        logits=np.asarray(logits, dtype=np.float32),
        target_ids=np.asarray(target_ids, dtype=np.int64),
        epsilon=epsilon,
        padding_mask=binary,
    )
