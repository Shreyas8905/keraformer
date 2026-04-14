"""Standard position-wise feed-forward network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@dataclass
class FFN:
    """Two-layer FFN with configurable activation."""

    d_model: int
    d_ff: int | None = None
    activation: str = "gelu"
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_ff = self.d_ff or (4 * self.d_model)
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.activation not in {"relu", "gelu", "silu"}:
            raise ValueError("activation must be one of: relu, gelu, silu")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.w1 = rng.normal(0.0, scale, size=(self.d_model, self.d_ff)).astype(np.float32)
        self.b1 = np.zeros((self.d_ff,), dtype=np.float32)
        self.w2 = rng.normal(0.0, scale, size=(self.d_ff, self.d_model)).astype(np.float32)
        self.b2 = np.zeros((self.d_model,), dtype=np.float32)

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return _relu(x)
        if self.activation == "gelu":
            return _gelu(x)
        return _silu(x)

    def _apply_dropout(self, x: np.ndarray, training: bool) -> np.ndarray:
        if not training or self.dropout == 0.0:
            return x
        rng = np.random.default_rng(self.seed)
        keep = 1.0 - self.dropout
        mask = rng.random(size=x.shape) < keep
        return (x * mask) / keep

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.d_model:
            raise ValueError("last dimension of x must equal d_model")
        hidden = self._act(np.matmul(arr, self.w1) + self.b1)
        hidden = self._apply_dropout(hidden, training=training)
        out = np.matmul(hidden, self.w2) + self.b2
        return out.astype(np.float32)
