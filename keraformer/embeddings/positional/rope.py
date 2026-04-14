"""Rotary positional embeddings (RoPE)."""

from __future__ import annotations

import numpy as np


def _rope_frequencies(seq_len: int, head_dim: int, base: float = 10000.0) -> np.ndarray:
    half_dim = head_dim // 2
    positions = np.arange(seq_len, dtype=np.float32)[:, None]
    freq_indices = np.arange(half_dim, dtype=np.float32)[None, :]
    theta = np.power(base, -2.0 * freq_indices / float(head_dim))
    return positions * theta


def apply_rope(x: np.ndarray, positions: np.ndarray | None = None, base: float = 10000.0) -> np.ndarray:
    """Apply RoPE to the last two dimensions (sequence, head_dim) of ``x``.

    Expected shape: ``(..., seq_len, head_dim)`` with even ``head_dim``.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("x must have at least 2 dimensions")

    seq_len = arr.shape[-2]
    head_dim = arr.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    if positions is None:
        pos = np.arange(seq_len, dtype=np.float32)
    else:
        pos = np.asarray(positions, dtype=np.float32)
        if pos.shape != (seq_len,):
            raise ValueError("positions must have shape (seq_len,)")

    freqs = _rope_frequencies(seq_len, head_dim, base=base)
    if positions is not None:
        freqs = pos[:, None] * np.power(
            base,
            -2.0 * np.arange(head_dim // 2, dtype=np.float32)[None, :] / float(head_dim),
        )

    cos = np.cos(freqs)
    sin = np.sin(freqs)

    x_even = arr[..., 0::2]
    x_odd = arr[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    out = np.empty_like(arr)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out
