"""Layer normalization implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerNorm:
    """Layer normalization across the final feature axis."""

    d_model: int
    eps: float = 1e-5
    affine: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

        if self.affine:
            self.gamma = np.ones((self.d_model,), dtype=np.float32)
            self.beta = np.zeros((self.d_model,), dtype=np.float32)
        else:
            self.gamma = None
            self.beta = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.d_model:
            raise ValueError("last dimension of x must equal d_model")

        mean = np.mean(arr, axis=-1, keepdims=True)
        var = np.var(arr, axis=-1, keepdims=True)
        y = (arr - mean) / np.sqrt(var + self.eps)

        if self.affine:
            y = y * self.gamma + self.beta
        return y.astype(np.float32)
