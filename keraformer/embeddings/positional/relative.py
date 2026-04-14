"""Relative position bias utilities (T5-style buckets)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def t5_relative_position_bucket(
    relative_position: np.ndarray,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    """Map relative positions to T5-style log buckets."""
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")

    half = num_buckets // 2
    n = np.asarray(relative_position)
    sign = (n > 0).astype(np.int32)
    n_abs = np.abs(n)

    max_exact = max(1, half // 2)
    is_small = n_abs < max_exact

    n_abs_float = np.maximum(n_abs, 1).astype(np.float32)
    log_ratio = np.log(n_abs_float / float(max_exact)) / np.log(float(max_distance) / float(max_exact))
    log_bucket = max_exact + (log_ratio * (half - max_exact)).astype(np.int32)
    log_bucket = np.clip(log_bucket, 0, half - 1)

    bucket = np.where(is_small, n_abs, log_bucket)
    return (bucket + sign * half).astype(np.int32)


@dataclass
class RelativePositionBias:
    """Learned relative position bias table indexed by T5 buckets."""

    num_heads: int
    num_buckets: int = 32
    max_distance: int = 128
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        rng = np.random.default_rng(self.seed)
        self.table = rng.normal(0.0, 0.02, size=(self.num_heads, self.num_buckets)).astype(np.float32)

    def __call__(self, query_len: int, key_len: int) -> np.ndarray:
        q_pos = np.arange(query_len, dtype=np.int32)[:, None]
        k_pos = np.arange(key_len, dtype=np.int32)[None, :]
        rel = k_pos - q_pos
        buckets = t5_relative_position_bucket(
            rel,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        bias = self.table[:, buckets]
        return bias.astype(np.float32)
