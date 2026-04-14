"""Encoder-decoder cross attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multi_head_attention import MultiHeadAttention


@dataclass
class CrossAttention:
    """Cross attention wrapper over standard MHA."""

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
        decoder_query: np.ndarray,
        encoder_key: np.ndarray,
        encoder_value: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.inner.call(decoder_query, encoder_key, encoder_value, mask=mask, training=training)
