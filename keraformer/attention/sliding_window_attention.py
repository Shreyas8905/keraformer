"""Sliding-window attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multi_head_attention import MultiHeadAttention


def _window_mask(query_len: int, key_len: int, window_size: int, causal: bool = False) -> np.ndarray:
    mask = np.zeros((query_len, key_len), dtype=np.float32)
    for i in range(query_len):
        for j in range(key_len):
            too_far = abs(i - j) > window_size
            future = causal and j > i
            if too_far or future:
                mask[i, j] = float("-inf")
    return mask


@dataclass
class SlidingWindowAttention:
    """MHA with a local sliding window mask."""

    d_model: int
    num_heads: int
    window_size: int
    dropout: float = 0.0
    seed: int | None = None
    causal: bool = False

    def __post_init__(self) -> None:
        if self.window_size < 0:
            raise ValueError("window_size must be non-negative")
        self.inner = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            seed=self.seed,
        )

    def call(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_len = query.shape[1]
        k_len = key.shape[1]
        local = _window_mask(q_len, k_len, self.window_size, causal=self.causal)
        local = local[np.newaxis, np.newaxis, :, :]
        final_mask = local if mask is None else (np.asarray(mask, dtype=np.float32) + local)
        return self.inner.call(query, key, value, mask=final_mask, training=training)
