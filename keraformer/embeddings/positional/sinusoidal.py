"""Fixed sinusoidal positional encoding."""

from __future__ import annotations

import numpy as np


def sinusoidal_positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Return sinusoidal positional encoding with shape (seq_len, d_model)."""
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if d_model <= 0:
        raise ValueError("d_model must be positive")

    positions = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    dims = np.arange(d_model, dtype=np.float32)[np.newaxis, :]
    angle_rates = 1.0 / np.power(base, (2.0 * np.floor(dims / 2.0)) / float(d_model))
    angles = positions * angle_rates

    encoding = np.empty((seq_len, d_model), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding
