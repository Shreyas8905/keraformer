"""Token embeddings for learned and pretrained lookup modes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TokenEmbedding:
    """Embed integer token IDs into vectors.

    Parameters
    ----------
    vocab_size:
        Size of the vocabulary for learned lookup mode.
    d_model:
        Embedding dimension.
    mode:
        Either ``"learned"`` (default) or ``"hf"``.
    scale:
        If True, multiply outputs by ``sqrt(d_model)`` as in Vaswani et al.
    seed:
        Optional RNG seed for deterministic initialization.
    hf_model_name:
        Hugging Face model id used in ``"hf"`` mode.
    """

    vocab_size: int
    d_model: int
    mode: str = "learned"
    scale: bool = True
    seed: int | None = None
    hf_model_name: str | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.mode not in {"learned", "hf"}:
            raise ValueError("mode must be either 'learned' or 'hf'")

        self._rng = np.random.default_rng(self.seed)
        self._embedding_matrix: np.ndarray

        if self.mode == "learned":
            self._embedding_matrix = self._rng.normal(
                loc=0.0,
                scale=0.02,
                size=(self.vocab_size, self.d_model),
            ).astype(np.float32)
        else:
            self._embedding_matrix = self._load_hf_embeddings()

    def _load_hf_embeddings(self) -> np.ndarray:
        if not self.hf_model_name:
            raise ValueError("hf_model_name is required when mode='hf'")

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "transformers is required for mode='hf'. Install with pip install transformers."
            ) from exc

        try:
            model = AutoModel.from_pretrained(self.hf_model_name)
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "Failed to load Hugging Face model embeddings. "
                "Ensure a compatible backend (torch/tf) is installed."
            ) from exc

        weight = model.get_input_embeddings().weight
        if hasattr(weight, "detach"):
            weight = weight.detach().cpu().numpy()
        else:
            weight = np.asarray(weight)

        matrix = weight.astype(np.float32)
        if matrix.shape[1] == self.d_model:
            return matrix

        if matrix.shape[1] > self.d_model:
            return matrix[:, : self.d_model]

        pad_width = self.d_model - matrix.shape[1]
        return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant")

    @property
    def embedding_matrix(self) -> np.ndarray:
        return self._embedding_matrix

    def __call__(self, token_ids: np.ndarray | list[list[int]]) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence_length)")
        if ids.min() < 0:
            raise ValueError("token_ids must be non-negative")
        if ids.max() >= self._embedding_matrix.shape[0]:
            raise ValueError("token_ids contain values outside embedding vocabulary")

        output = self._embedding_matrix[ids]
        if self.scale:
            output = output * math.sqrt(float(self.d_model))
        return output.astype(np.float32)
