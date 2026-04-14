"""BERT-style encoder model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.blocks import EncoderBlock


@dataclass
class BERT:
    """Encoder-only model with MLM and NSP heads."""

    vocab_size: int
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    num_classes: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.embed = rng.normal(0.0, scale, size=(self.vocab_size, self.d_model)).astype(np.float32)
        self.encoder = [
            EncoderBlock(d_model=self.d_model, num_heads=self.num_heads, seed=None if self.seed is None else self.seed + i)
            for i in range(self.num_layers)
        ]
        self.mlm_head = rng.normal(0.0, scale, size=(self.d_model, self.vocab_size)).astype(np.float32)
        self.nsp_head = rng.normal(0.0, scale, size=(self.d_model, 2)).astype(np.float32)
        self.cls_head = rng.normal(0.0, scale, size=(self.d_model, self.num_classes)).astype(np.float32)

    def call(self, token_ids: np.ndarray, training: bool = False) -> dict[str, np.ndarray]:
        ids = np.asarray(token_ids, dtype=np.int64)
        hidden = self.embed[ids]
        for block in self.encoder:
            hidden = block.call(hidden, training=training)

        cls = hidden[:, 0, :]
        return {
            "hidden_states": hidden.astype(np.float32),
            "mlm_logits": np.matmul(hidden, self.mlm_head).astype(np.float32),
            "nsp_logits": np.matmul(cls, self.nsp_head).astype(np.float32),
            "class_logits": np.matmul(cls, self.cls_head).astype(np.float32),
        }
