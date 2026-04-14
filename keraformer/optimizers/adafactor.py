"""Adafactor optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Adafactor:
    """Memory-efficient Adafactor for vectors and matrices."""

    lr: float = 1e-3
    beta2: float = 0.999
    eps1: float = 1e-30
    clip_threshold: float = 1.0
    factored: bool = True
    v: np.ndarray | None = field(default=None, init=False)
    vr: np.ndarray | None = field(default=None, init=False)
    vc: np.ndarray | None = field(default=None, init=False)

    def _factor_second_moment(self, g2: np.ndarray) -> np.ndarray:
        assert self.vr is not None and self.vc is not None
        self.vr = self.beta2 * self.vr + (1.0 - self.beta2) * np.mean(g2, axis=1)
        self.vc = self.beta2 * self.vc + (1.0 - self.beta2) * np.mean(g2, axis=0)
        denom = np.mean(self.vr) + self.eps1
        v_hat = np.outer(self.vr, self.vc) / denom
        return v_hat

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=np.float32)
        g = np.asarray(grads, dtype=np.float32)
        if p.shape != g.shape:
            raise ValueError("params and grads must have identical shapes")

        g2 = g * g + self.eps1
        if self.factored and g.ndim == 2:
            if self.vr is None or self.vc is None:
                self.vr = np.zeros((g.shape[0],), dtype=np.float32)
                self.vc = np.zeros((g.shape[1],), dtype=np.float32)
            v_hat = self._factor_second_moment(g2)
        else:
            if self.v is None:
                self.v = np.zeros_like(g)
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * g2
            v_hat = self.v

        update = g / (np.sqrt(v_hat) + self.eps1)
        rms = np.sqrt(np.mean(update * update))
        scale = min(1.0, self.clip_threshold / (rms + self.eps1))
        update = update * scale
        p_next = p - self.lr * update
        return p_next.astype(np.float32)
