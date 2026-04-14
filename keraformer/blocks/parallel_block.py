"""PaLM-style parallel attention+FFN block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.attention import MultiHeadAttention
from keraformer.feedforward import FFN
from keraformer.normalization import LayerNorm


@dataclass
class ParallelBlock:
    """Run attention and FFN in parallel on normalized input."""

    d_model: int
    num_heads: int
    attention_cls: type = MultiHeadAttention
    ffn_cls: type = FFN
    norm_cls: type = LayerNorm
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.norm = self.norm_cls(d_model=self.d_model)
        self.attn = self.attention_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            seed=self.seed,
        )
        self.ffn = self.ffn_cls(d_model=self.d_model, seed=self.seed)

    def call(self, x: np.ndarray, mask: np.ndarray | None = None, training: bool = False) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        normed = self.norm(arr)
        attn_out, _ = self.attn.call(normed, normed, normed, mask=mask, training=training)
        ffn_out = self.ffn(normed, training=training)
        return (arr + attn_out + ffn_out).astype(np.float32)
