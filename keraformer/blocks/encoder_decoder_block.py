"""Combined encoder-decoder block wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoder_block import DecoderBlock
from .encoder_block import EncoderBlock


@dataclass
class EncoderDecoderBlock:
    """Convenience wrapper that runs one encoder and one decoder block."""

    d_model: int
    num_heads: int
    seed: int | None = None

    def __post_init__(self) -> None:
        self.encoder = EncoderBlock(d_model=self.d_model, num_heads=self.num_heads, seed=self.seed)
        self.decoder = DecoderBlock(d_model=self.d_model, num_heads=self.num_heads, seed=self.seed)

    def call(
        self,
        encoder_input: np.ndarray,
        decoder_input: np.ndarray,
        encoder_mask: np.ndarray | None = None,
        decoder_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        enc = self.encoder.call(encoder_input, mask=encoder_mask, training=training)
        dec = self.decoder.call(
            decoder_input,
            encoder_output=enc,
            self_mask=decoder_mask,
            cross_mask=cross_mask,
            training=training,
        )
        return enc.astype(np.float32), dec.astype(np.float32)
