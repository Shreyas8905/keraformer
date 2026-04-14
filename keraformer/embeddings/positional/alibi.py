"""ALiBi attention biases."""

from __future__ import annotations

import numpy as np


def alibi_slopes(num_heads: int) -> np.ndarray:
    """Return per-head ALiBi slopes using an exponential schedule."""
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    heads = np.arange(num_heads, dtype=np.float32)
    return np.power(2.0, -8.0 * heads / float(num_heads)).astype(np.float32)


def alibi_bias(num_heads: int, query_len: int, key_len: int) -> np.ndarray:
    """Return ALiBi additive bias with shape (num_heads, query_len, key_len)."""
    if query_len < 0 or key_len < 0:
        raise ValueError("query_len and key_len must be non-negative")

    slopes = alibi_slopes(num_heads)[:, None, None]
    q_pos = np.arange(query_len, dtype=np.float32)[None, :, None]
    k_pos = np.arange(key_len, dtype=np.float32)[None, None, :]
    distance = np.abs(q_pos - k_pos)
    return (-slopes * distance).astype(np.float32)
