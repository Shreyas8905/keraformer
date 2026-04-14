"""Grouped-query attention (GQA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaled_dot_product import scaled_dot_product_attention


@dataclass
class GroupedQueryAttention:
    """GQA shares K/V heads across groups of query heads."""

    d_model: int
    num_heads: int
    num_kv_groups: int
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0 or self.num_kv_groups <= 0:
            raise ValueError("d_model, num_heads, and num_kv_groups must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.head_dim = self.d_model // self.num_heads
        self.heads_per_group = self.num_heads // self.num_kv_groups

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.w_q = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_k = rng.normal(
            0.0,
            scale,
            size=(self.d_model, self.num_kv_groups * self.head_dim),
        ).astype(np.float32)
        self.w_v = rng.normal(
            0.0,
            scale,
            size=(self.d_model, self.num_kv_groups * self.head_dim),
        ).astype(np.float32)
        self.w_o = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)

    def _split_q(self, x: np.ndarray) -> np.ndarray:
        bsz, seq_len, _ = x.shape
        q = np.matmul(x, self.w_q).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return np.transpose(q, (0, 2, 1, 3))

    def _split_kv(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        bsz, seq_len, _ = x.shape
        kv = np.matmul(x, w).reshape(bsz, seq_len, self.num_kv_groups, self.head_dim)
        kv = np.transpose(kv, (0, 2, 1, 3))
        return np.repeat(kv, repeats=self.heads_per_group, axis=1)

    def call(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = self._split_q(np.asarray(query, dtype=np.float32))
        k = self._split_kv(np.asarray(key, dtype=np.float32), self.w_k)
        v = self._split_kv(np.asarray(value, dtype=np.float32), self.w_v)

        out, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=self.dropout,
            training=training,
            seed=self.seed,
        )
        merged = np.transpose(out, (0, 2, 1, 3)).reshape(query.shape[0], query.shape[1], self.d_model)
        projected = np.matmul(merged, self.w_o)
        return projected.astype(np.float32), weights.astype(np.float32)
