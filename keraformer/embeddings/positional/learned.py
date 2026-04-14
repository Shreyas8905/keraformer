"""Learned absolute positional encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LearnedPositionalEncoding:
    """Learned absolute position embeddings."""

    max_len: int
    d_model: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_len <= 0:
            raise ValueError("max_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.02, size=(self.max_len, self.d_model)).astype(np.float32)

    def __call__(self, seq_len: int) -> np.ndarray:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if seq_len > self.max_len:
            raise ValueError("seq_len exceeds configured max_len")
        return self.weights[:seq_len]
