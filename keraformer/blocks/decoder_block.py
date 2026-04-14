"""Composable transformer decoder block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.attention import CrossAttention, MultiHeadAttention
from keraformer.feedforward import FFN
from keraformer.normalization import LayerNorm


def _causal_mask(seq_len: int) -> np.ndarray:
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[0, 0, i, j] = float("-inf")
    return mask


@dataclass
class DecoderBlock:
    """Decoder block supporting optional cross-attention."""

    d_model: int
    num_heads: int
    attention_cls: type = MultiHeadAttention
    cross_attention_cls: type = CrossAttention
    ffn_cls: type = FFN
    norm_cls: type = LayerNorm
    pre_norm: bool = True
    dropout: float = 0.0
    use_cross_attention: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        self.self_attention = self.attention_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            seed=self.seed,
        )
        self.cross_attention = self.cross_attention_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            seed=self.seed,
        )
        self.ffn = self.ffn_cls(d_model=self.d_model, seed=self.seed)
        self.norm1 = self.norm_cls(d_model=self.d_model)
        self.norm2 = self.norm_cls(d_model=self.d_model)
        self.norm3 = self.norm_cls(d_model=self.d_model)

    def call(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray | None = None,
        self_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
        training: bool = False,
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        seq_len = arr.shape[1]
        causal = _causal_mask(seq_len)
        self_mask = causal if self_mask is None else (causal + np.asarray(self_mask, dtype=np.float32))

        if self.pre_norm:
            y, _ = self.self_attention.call(self.norm1(arr), self.norm1(arr), self.norm1(arr), mask=self_mask, training=training)
            arr = arr + y

            if self.use_cross_attention and encoder_output is not None:
                c, _ = self.cross_attention.call(
                    self.norm2(arr),
                    np.asarray(encoder_output, dtype=np.float32),
                    np.asarray(encoder_output, dtype=np.float32),
                    mask=cross_mask,
                    training=training,
                )
                arr = arr + c

            z = self.ffn(self.norm3(arr), training=training)
            return (arr + z).astype(np.float32)

        y, _ = self.self_attention.call(arr, arr, arr, mask=self_mask, training=training)
        arr = self.norm1(arr + y)

        if self.use_cross_attention and encoder_output is not None:
            c, _ = self.cross_attention.call(arr, encoder_output, encoder_output, mask=cross_mask, training=training)
            arr = self.norm2(arr + c)

        z = self.ffn(arr, training=training)
        return self.norm3(arr + z).astype(np.float32)
