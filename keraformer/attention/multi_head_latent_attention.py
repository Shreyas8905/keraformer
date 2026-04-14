"""Multi-head latent attention (MLA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaled_dot_product import scaled_dot_product_attention


@dataclass
class MultiHeadLatentAttention:
    """MLA compresses KV into a smaller latent cache."""

    d_model: int
    num_heads: int
    kv_latent_dim: int
    q_latent_dim: int | None = None
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0 or self.kv_latent_dim <= 0:
            raise ValueError("d_model, num_heads, and kv_latent_dim must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.head_dim = self.d_model // self.num_heads
        self.q_latent_dim = self.q_latent_dim or self.kv_latent_dim
        if self.q_latent_dim <= 0:
            raise ValueError("q_latent_dim must be positive")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)

        self.w_dkv = rng.normal(0.0, scale, size=(self.d_model, self.kv_latent_dim)).astype(np.float32)
        self.w_uk = rng.normal(0.0, scale, size=(self.kv_latent_dim, self.d_model)).astype(np.float32)
        self.w_uv = rng.normal(0.0, scale, size=(self.kv_latent_dim, self.d_model)).astype(np.float32)

        self.w_dq = rng.normal(0.0, scale, size=(self.d_model, self.q_latent_dim)).astype(np.float32)
        self.w_uq = rng.normal(0.0, scale, size=(self.q_latent_dim, self.d_model)).astype(np.float32)
        self.w_o = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)

        self.last_kv_cache_shape: tuple[int, ...] | None = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        bsz, seq_len, _ = x.shape
        x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def kv_cache_size(self, batch_size: int, seq_len: int) -> tuple[int, int]:
        full = batch_size * seq_len * (2 * self.d_model)
        latent = batch_size * seq_len * self.kv_latent_dim
        return full, latent

    def call(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_in = np.asarray(query, dtype=np.float32)
        k_in = np.asarray(key, dtype=np.float32)
        v_in = np.asarray(value, dtype=np.float32)

        q_latent = np.matmul(q_in, self.w_dq)
        q_full = np.matmul(q_latent, self.w_uq)

        kv_source = 0.5 * (k_in + v_in)
        c_kv = np.matmul(kv_source, self.w_dkv)
        self.last_kv_cache_shape = c_kv.shape
        k_full = np.matmul(c_kv, self.w_uk)
        v_full = np.matmul(c_kv, self.w_uv)

        q = self._split_heads(q_full)
        k = self._split_heads(k_full)
        v = self._split_heads(v_full)

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
