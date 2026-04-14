"""Scaled dot-product attention core."""

from __future__ import annotations

import math

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _apply_dropout(x: np.ndarray, dropout: float, training: bool, seed: int | None = None) -> np.ndarray:
    if not training or dropout <= 0.0:
        return x
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 - dropout
    mask = rng.random(size=x.shape) < keep_prob
    return (x * mask) / keep_prob


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
    dropout: float = 0.0,
    training: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Expected shapes:
      query: (B, H, T_q, D_k)
      key:   (B, H, T_k, D_k)
      value: (B, H, T_k, D_v)
      mask:  broadcastable to (B, H, T_q, T_k), additive {0, -inf}
    """
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions must match")
    q_heads = q.shape[1]
    k_heads = k.shape[1]
    v_heads = v.shape[1]
    valid_k_heads = k_heads == q_heads or k_heads == 1
    valid_v_heads = v_heads == q_heads or v_heads == 1
    if not (valid_k_heads and valid_v_heads and k_heads == v_heads):
        raise ValueError("head dimensions must match or use shared (1-head) KV")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query/key depth must match")
    if k.shape[2] != v.shape[2]:
        raise ValueError("key and value sequence lengths must match")

    d_k = float(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + np.asarray(mask, dtype=np.float32)

    weights = _softmax(scores, axis=-1)
    weights = _apply_dropout(weights, dropout=dropout, training=training, seed=seed)
    output = np.matmul(weights, v)
    return output.astype(np.float32), weights.astype(np.float32)
