"""Composable transformer encoder block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.attention import MultiHeadAttention
from keraformer.feedforward import FFN
from keraformer.normalization import LayerNorm


@dataclass
class EncoderBlock:
    """Transformer encoder block with swappable attention/ffn/norm."""

    d_model: int
    num_heads: int
    attention_cls: type = MultiHeadAttention
    ffn_cls: type = FFN
    norm_cls: type = LayerNorm
    pre_norm: bool = True
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.self_attention = self.attention_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            seed=self.seed,
        )
        self.ffn = self.ffn_cls(d_model=self.d_model, seed=self.seed)
        self.norm1 = self.norm_cls(d_model=self.d_model)
        self.norm2 = self.norm_cls(d_model=self.d_model)

    def call(self, x: np.ndarray, mask: np.ndarray | None = None, training: bool = False) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if self.pre_norm:
            y, _ = self.self_attention.call(self.norm1(arr), self.norm1(arr), self.norm1(arr), mask=mask, training=training)
            arr = arr + y
            z = self.ffn(self.norm2(arr), training=training)
            return (arr + z).astype(np.float32)

        y, _ = self.self_attention.call(arr, arr, arr, mask=mask, training=training)
        arr = self.norm1(arr + y)
        z = self.ffn(arr, training=training)
        return self.norm2(arr + z).astype(np.float32)
