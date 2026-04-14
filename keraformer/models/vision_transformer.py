"""Vision Transformer (ViT) wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keraformer.blocks import EncoderBlock


@dataclass
class VisionTransformer:
    """Simple ViT with patch embedding and encoder stack."""

    image_size: int
    patch_size: int
    in_channels: int
    num_classes: int
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side * self.num_patches_per_side
        patch_dim = self.patch_size * self.patch_size * self.in_channels

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.patch_proj = rng.normal(0.0, scale, size=(patch_dim, self.d_model)).astype(np.float32)
        self.cls_token = rng.normal(0.0, scale, size=(1, 1, self.d_model)).astype(np.float32)
        self.pos_embed = rng.normal(0.0, scale, size=(1, self.num_patches + 1, self.d_model)).astype(np.float32)
        self.encoder = [
            EncoderBlock(d_model=self.d_model, num_heads=self.num_heads, seed=None if self.seed is None else self.seed + i)
            for i in range(self.num_layers)
        ]
        self.head = rng.normal(0.0, scale, size=(self.d_model, self.num_classes)).astype(np.float32)

    def _patchify(self, images: np.ndarray) -> np.ndarray:
        arr = np.asarray(images, dtype=np.float32)
        bsz, h, w, c = arr.shape
        if h != self.image_size or w != self.image_size or c != self.in_channels:
            raise ValueError("image shape must match configured image_size and in_channels")

        p = self.patch_size
        patches = arr.reshape(
            bsz,
            self.num_patches_per_side,
            p,
            self.num_patches_per_side,
            p,
            c,
        )
        patches = np.transpose(patches, (0, 1, 3, 2, 4, 5)).reshape(bsz, self.num_patches, p * p * c)
        return patches

    def call(self, images: np.ndarray, training: bool = False) -> dict[str, np.ndarray]:
        patches = self._patchify(images)
        tokens = np.matmul(patches, self.patch_proj)
        bsz = tokens.shape[0]
        cls = np.repeat(self.cls_token, repeats=bsz, axis=0)
        hidden = np.concatenate([cls, tokens], axis=1) + self.pos_embed

        for block in self.encoder:
            hidden = block.call(hidden, training=training)

        cls_hidden = hidden[:, 0, :]
        logits = np.matmul(cls_hidden, self.head)
        return {"hidden_states": hidden.astype(np.float32), "logits": logits.astype(np.float32)}
