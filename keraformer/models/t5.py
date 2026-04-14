"""T5-style encoder-decoder model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.blocks import DecoderBlock, EncoderBlock
from keraformer.feedforward import GatedFFN
from keraformer.normalization import RMSNorm


@dataclass
class T5:
    """T5-style text-to-text encoder-decoder model."""

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.embed = rng.normal(0.0, scale, size=(self.vocab_size, self.d_model)).astype(np.float32)
        self.encoder = [
            EncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_cls=GatedFFN,
                norm_cls=RMSNorm,
                pre_norm=True,
                seed=None if self.seed is None else self.seed + i,
            )
            for i in range(self.num_layers)
        ]
        self.decoder = [
            DecoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_cls=GatedFFN,
                norm_cls=RMSNorm,
                pre_norm=True,
                seed=None if self.seed is None else self.seed + 100 + i,
            )
            for i in range(self.num_layers)
        ]
        self.lm_head = rng.normal(0.0, scale, size=(self.d_model, self.vocab_size)).astype(np.float32)

    def call(self, src_ids: np.ndarray, tgt_ids: np.ndarray, training: bool = False) -> np.ndarray:
        src = self.embed[np.asarray(src_ids, dtype=np.int64)]
        tgt = self.embed[np.asarray(tgt_ids, dtype=np.int64)]

        for block in self.encoder:
            src = block.call(src, training=training)
        for block in self.decoder:
            tgt = block.call(tgt, encoder_output=src, training=training)

        return np.matmul(tgt, self.lm_head).astype(np.float32)
