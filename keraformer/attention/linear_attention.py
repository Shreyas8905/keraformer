"""Linear attention approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _phi(x: np.ndarray) -> np.ndarray:
    # Positive feature map for kernelized linear attention.
    return np.maximum(x, 0.0) + 1.0


@dataclass
class LinearAttention:
    """Kernelized linear attention using a positive feature map."""

    d_model: int
    num_heads: int
    eps: float = 1e-6
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.w_q = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_k = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_v = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)
        self.w_o = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        bsz, seq_len, _ = x.shape
        x = x.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def call(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        del training  # Unused in this deterministic approximation.
        del mask  # Masking is not handled in this simplified linear variant.

        q = self._split_heads(np.matmul(np.asarray(query, dtype=np.float32), self.w_q))
        k = self._split_heads(np.matmul(np.asarray(key, dtype=np.float32), self.w_k))
        v = self._split_heads(np.matmul(np.asarray(value, dtype=np.float32), self.w_v))

        qf = _phi(q)
        kf = _phi(k)

        kv = np.einsum("bhtm,bhtd->bhmd", kf, v)
        k_sum = np.sum(kf, axis=2)
        denom = np.einsum("bhtm,bhm->bht", qf, k_sum) + self.eps
        out = np.einsum("bhtm,bhmd->bhtd", qf, kv) / denom[..., np.newaxis]

        merged = np.transpose(out, (0, 2, 1, 3)).reshape(query.shape[0], query.shape[1], self.d_model)
        projected = np.matmul(merged, self.w_o)

        # Approximate weights for interface compatibility.
        scores = np.matmul(qf, np.swapaxes(kf, -1, -2))
        weights = scores / (np.sum(scores, axis=-1, keepdims=True) + self.eps)
        return projected.astype(np.float32), weights.astype(np.float32)
