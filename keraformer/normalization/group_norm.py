"""Group normalization implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroupNorm:
    """Group normalization over the feature axis."""

    d_model: int
    num_groups: int
    eps: float = 1e-5
    affine: bool = True

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if self.d_model % self.num_groups != 0:
            raise ValueError("d_model must be divisible by num_groups")
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

        group_size = self.d_model // self.num_groups
        reshaped = arr.reshape(*arr.shape[:-1], self.num_groups, group_size)
        mean = np.mean(reshaped, axis=-1, keepdims=True)
        var = np.var(reshaped, axis=-1, keepdims=True)
        normalized = (reshaped - mean) / np.sqrt(var + self.eps)
        y = normalized.reshape(arr.shape)

        if self.affine:
            y = y * self.gamma + self.beta
        return y.astype(np.float32)
