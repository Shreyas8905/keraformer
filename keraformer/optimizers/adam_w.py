"""AdamW optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdamW:
    """AdamW with decoupled weight decay."""

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2
    m: np.ndarray | None = field(default=None, init=False)
    v: np.ndarray | None = field(default=None, init=False)
    t: int = field(default=0, init=False)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=np.float32)
        g = np.asarray(grads, dtype=np.float32)
        if p.shape != g.shape:
            raise ValueError("params and grads must have identical shapes")

        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        update = m_hat / (np.sqrt(v_hat) + self.eps)
        p_next = p - self.lr * (update + self.weight_decay * p)
        return p_next.astype(np.float32)
