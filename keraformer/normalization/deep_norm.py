"""DeepNorm residual scaling wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layer_norm import LayerNorm


@dataclass
class DeepNorm:
    """DeepNorm with residual scaling followed by LayerNorm."""

    d_model: int
    num_layers: int
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.alpha = float((2.0 * self.num_layers) ** 0.25)
        self.norm = LayerNorm(d_model=self.d_model, eps=self.eps)

    def __call__(self, residual: np.ndarray, sublayer_out: np.ndarray) -> np.ndarray:
        res = np.asarray(residual, dtype=np.float32)
        sub = np.asarray(sublayer_out, dtype=np.float32)
        if res.shape != sub.shape:
            raise ValueError("residual and sublayer_out must have identical shapes")
        combined = self.alpha * res + sub
        return self.norm(combined)
