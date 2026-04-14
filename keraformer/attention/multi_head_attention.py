"""Multi-head attention implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaled_dot_product import scaled_dot_product_attention


@dataclass
class MultiHeadAttention:
    """Standard multi-head attention (MHA)."""

    d_model: int
    num_heads: int
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.w_q = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_k = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_v = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_o = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)

    def _project(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.matmul(x, w)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        bsz, seq_len, _ = x.shape
        reshaped = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return np.transpose(reshaped, (0, 2, 1, 3))

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        bsz, _, seq_len, _ = x.shape
        transposed = np.transpose(x, (0, 2, 1, 3))
        return transposed.reshape(bsz, seq_len, self.d_model)

    def call(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = self._split_heads(self._project(np.asarray(query, dtype=np.float32), self.w_q))
        k = self._split_heads(self._project(np.asarray(key, dtype=np.float32), self.w_k))
        v = self._split_heads(self._project(np.asarray(value, dtype=np.float32), self.w_v))

        attn_out, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=self.dropout,
            training=training,
            seed=self.seed,
        )
        merged = self._merge_heads(attn_out)
        out = np.matmul(merged, self.w_o)
        return out.astype(np.float32), attn_weights.astype(np.float32)
