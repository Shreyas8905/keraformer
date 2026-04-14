"""No-op positional encoding."""

from __future__ import annotations

import numpy as np


def no_positional_encoding(token_embeddings: np.ndarray) -> np.ndarray:
    """Return token embeddings unchanged."""
    return np.asarray(token_embeddings)
