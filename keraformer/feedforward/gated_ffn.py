"""Gated feed-forward variants (GLU/SwiGLU/GeGLU)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ffn import _gelu, _silu


@dataclass
class GatedFFN:
    """Gated FFN supporting GLU, SwiGLU, and GeGLU."""

    d_model: int
    d_ff: int | None = None
    variant: str = "swiglu"
    dropout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_ff = self.d_ff or int((8 * self.d_model) / 3)
        if self.variant not in {"glu", "swiglu", "geglu"}:
            raise ValueError("variant must be one of: glu, swiglu, geglu")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.w_gate = rng.normal(0.0, scale, size=(self.d_model, self.d_ff)).astype(np.float32)
        self.w_up = rng.normal(0.0, scale, size=(self.d_model, self.d_ff)).astype(np.float32)
        self.w_down = rng.normal(0.0, scale, size=(self.d_ff, self.d_model)).astype(np.float32)

    def _gate(self, x: np.ndarray) -> np.ndarray:
        if self.variant == "glu":
            return 1.0 / (1.0 + np.exp(-x))
        if self.variant == "swiglu":
            return _silu(x)
        return _gelu(x)

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
        gate = self._gate(np.matmul(arr, self.w_gate))
        up = np.matmul(arr, self.w_up)
        hidden = gate * up
        hidden = self._apply_dropout(hidden, training=training)
        out = np.matmul(hidden, self.w_down)
        return out.astype(np.float32)
