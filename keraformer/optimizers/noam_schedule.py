"""Noam learning-rate schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoamSchedule:
    """Vaswani et al. warmup + inverse-sqrt decay schedule."""

    d_model: int
    warmup_steps: int

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")

    def __call__(self, step: int) -> float:
        if step <= 0:
            raise ValueError("step must be >= 1")
        a = step ** (-0.5)
        b = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * min(a, b)
