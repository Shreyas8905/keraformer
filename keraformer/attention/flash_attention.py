"""FlashAttention-style wrapper.

This module exposes the same interface as other attention classes. In this
reference implementation it delegates to scaled dot-product attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multi_head_attention import MultiHeadAttention


@dataclass
class FlashAttention:
    """Interface-compatible FlashAttention wrapper."""

    d_model: int
    num_heads: int
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
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
        return self.inner.call(query, key, value, mask=mask, training=training)
