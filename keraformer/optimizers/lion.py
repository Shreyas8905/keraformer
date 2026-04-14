"""Lion optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Lion:
    """Lion optimizer with sign-based updates."""

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 1e-2
    m: np.ndarray | None = field(default=None, init=False)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=np.float32)
        g = np.asarray(grads, dtype=np.float32)
        if p.shape != g.shape:
            raise ValueError("params and grads must have identical shapes")

        if self.m is None:
            self.m = np.zeros_like(p)

        c_t = self.beta1 * self.m + (1.0 - self.beta1) * g
        p_next = p - self.lr * (np.sign(c_t) + self.weight_decay * p)
        self.m = self.beta2 * self.m + (1.0 - self.beta2) * g
        return p_next.astype(np.float32)
