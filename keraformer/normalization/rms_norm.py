"""RMS normalization implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RMSNorm:
    """RMS normalization across the final feature axis."""

    d_model: int
    eps: float = 1e-8
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        self.gamma = np.ones((self.d_model,), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.d_model:
            raise ValueError("last dimension of x must equal d_model")

        rms = np.sqrt(np.mean(np.square(arr), axis=-1, keepdims=True) + self.eps)
        y = (arr / rms) * self.gamma
        return y.astype(np.float32)
