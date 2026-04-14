"""Convolutional feed-forward network variant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvFFN:
    """Depthwise-conv enhanced FFN.

    Uses a depthwise 1D convolution over sequence length followed by a
    point-wise projection back to d_model.
    """

    d_model: int
    kernel_size: int = 3
    activation: str = "gelu"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.activation not in {"relu", "gelu", "silu"}:
            raise ValueError("activation must be one of: relu, gelu, silu")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.depthwise_kernel = rng.normal(
            0.0,
            scale,
            size=(self.kernel_size, self.d_model),
        ).astype(np.float32)
        self.pointwise = rng.normal(0.0, scale, size=(self.d_model, self.d_model)).astype(np.float32)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return x / (1.0 + np.exp(-x))

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        del training
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.d_model:
            raise ValueError("last dimension of x must equal d_model")

        bsz, seq_len, channels = arr.shape
        pad = self.kernel_size // 2
        padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        conv = np.zeros((bsz, seq_len, channels), dtype=np.float32)

        for t in range(seq_len):
            window = padded[:, t : t + self.kernel_size, :]
            conv[:, t, :] = np.sum(window * self.depthwise_kernel[np.newaxis, :, :], axis=1)

        activated = self._activate(conv)
        out = np.matmul(activated, self.pointwise)
        return out.astype(np.float32)
