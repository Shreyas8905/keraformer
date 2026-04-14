"""Sequence-to-sequence transformer model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.blocks import DecoderBlock, EncoderBlock


@dataclass
class Transformer:
    """Encoder-decoder Transformer built from modular blocks."""

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.src_embed = rng.normal(0.0, scale, size=(self.vocab_size, self.d_model)).astype(np.float32)
        self.tgt_embed = rng.normal(0.0, scale, size=(self.vocab_size, self.d_model)).astype(np.float32)
        self.encoder = [
            EncoderBlock(d_model=self.d_model, num_heads=self.num_heads, seed=None if self.seed is None else self.seed + i)
            for i in range(self.num_layers)
        ]
        self.decoder = [
            DecoderBlock(d_model=self.d_model, num_heads=self.num_heads, seed=None if self.seed is None else self.seed + 100 + i)
            for i in range(self.num_layers)
        ]
        self.lm_head = rng.normal(0.0, scale, size=(self.d_model, self.vocab_size)).astype(np.float32)

    def _embed(self, token_ids: np.ndarray, table: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("token ids must have shape (batch, seq_len)")
        return table[ids]

    def call(self, src_ids: np.ndarray, tgt_ids: np.ndarray, training: bool = False) -> np.ndarray:
        enc = self._embed(src_ids, self.src_embed)
        dec = self._embed(tgt_ids, self.tgt_embed)

        for block in self.encoder:
            enc = block.call(enc, training=training)
        for block in self.decoder:
            dec = block.call(dec, encoder_output=enc, training=training)

        logits = np.matmul(dec, self.lm_head)
        return logits.astype(np.float32)
