"""Mixture-of-experts feed-forward layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ffn import FFN


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass
class MoEFFN:
    """Sparse top-k MoE using FFN experts."""

    d_model: int
    d_ff: int | None = None
    num_experts: int = 4
    top_k: int = 2
    aux_loss_alpha: float = 1e-2
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.d_model)
        self.w_router = rng.normal(0.0, scale, size=(self.d_model, self.num_experts)).astype(np.float32)
        self.experts = [
            FFN(d_model=self.d_model, d_ff=self.d_ff, activation="gelu", seed=None if self.seed is None else self.seed + i)
            for i in range(self.num_experts)
        ]
        self.last_aux_loss: float = 0.0

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != self.d_model:
            raise ValueError("last dimension of x must equal d_model")

        bsz, seq_len, _ = arr.shape
        flat = arr.reshape(bsz * seq_len, self.d_model)
        logits = np.matmul(flat, self.w_router)
        probs = _softmax(logits, axis=-1)

        topk_idx = np.argpartition(-probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        topk_probs = np.take_along_axis(probs, topk_idx, axis=-1)
        topk_probs = topk_probs / (np.sum(topk_probs, axis=-1, keepdims=True) + 1e-9)

        out_flat = np.zeros_like(flat)
        usage = np.zeros((self.num_experts,), dtype=np.float32)

        for expert_id, expert in enumerate(self.experts):
            selected_mask = topk_idx == expert_id
            token_rows = np.any(selected_mask, axis=-1)
            if not np.any(token_rows):
                continue

            rows_idx = np.where(token_rows)[0]
            expert_in = flat[rows_idx][np.newaxis, :, :]
            expert_out = expert(expert_in, training=training)[0]

            # Sum probabilities where this expert appears in top-k.
            weights_for_rows = np.sum(
                np.where(selected_mask[rows_idx], topk_probs[rows_idx], 0.0),
                axis=-1,
                keepdims=True,
            )
            out_flat[rows_idx] += expert_out * weights_for_rows
            usage[expert_id] = float(np.mean(token_rows.astype(np.float32)))

        prob_mean = np.mean(probs, axis=0)
        self.last_aux_loss = float(self.aux_loss_alpha * self.num_experts * np.sum(usage * prob_mean))
        return out_flat.reshape(bsz, seq_len, self.d_model).astype(np.float32)
