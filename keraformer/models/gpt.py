"""GPT-style decoder-only language model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.blocks import DecoderBlock


@dataclass
class GPT:
    """Autoregressive decoder-only model with optional weight tying."""

    vocab_size: int
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    tie_weights: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.embed = rng.normal(0.0, scale, size=(self.vocab_size, self.d_model)).astype(np.float32)
        self.decoder = [
            DecoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                use_cross_attention=False,
                seed=None if self.seed is None else self.seed + i,
            )
            for i in range(self.num_layers)
        ]
        if self.tie_weights:
            self.lm_head = self.embed.T.copy()
        else:
            self.lm_head = rng.normal(0.0, scale, size=(self.d_model, self.vocab_size)).astype(np.float32)

    def call(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        hidden = self.embed[ids]
        for block in self.decoder:
            hidden = block.call(hidden, encoder_output=None, training=training)
        logits = np.matmul(hidden, self.lm_head)
        return logits.astype(np.float32)
