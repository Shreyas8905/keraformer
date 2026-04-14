"""Tests for attention modules."""

from __future__ import annotations

import unittest

import numpy as np

from keraformer.attention import (
    CrossAttention,
    FlashAttention,
    GroupedQueryAttention,
    LinearAttention,
    MultiHeadAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
)


def _causal_additive_mask(seq_len: int) -> np.ndarray:
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[0, 0, i, j] = float("-inf")
    return mask


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 2
        self.t_q = 5
        self.t_k = 6
        self.d_model = 16
        self.heads = 4
        rng = np.random.default_rng(123)
        self.query = rng.normal(size=(self.batch, self.t_q, self.d_model)).astype(np.float32)
        self.key = rng.normal(size=(self.batch, self.t_k, self.d_model)).astype(np.float32)
        self.value = rng.normal(size=(self.batch, self.t_k, self.d_model)).astype(np.float32)

    def test_shape_checks_for_all_variants(self) -> None:
        variants = [
            MultiHeadAttention(self.d_model, self.heads, seed=1),
            MultiQueryAttention(self.d_model, self.heads, seed=2),
            GroupedQueryAttention(self.d_model, self.heads, num_kv_groups=2, seed=3),
            MultiHeadLatentAttention(self.d_model, self.heads, kv_latent_dim=6, seed=4),
            SlidingWindowAttention(self.d_model, self.heads, window_size=2, seed=5),
            LinearAttention(self.d_model, self.heads, seed=6),
            FlashAttention(self.d_model, self.heads, seed=7),
        ]

        for layer in variants:
            out, weights = layer.call(self.query, self.key, self.value)
            self.assertEqual(out.shape, (self.batch, self.t_q, self.d_model))
            self.assertEqual(weights.shape, (self.batch, self.heads, self.t_q, self.t_k))

        cross = CrossAttention(self.d_model, self.heads, seed=8)
        out, weights = cross.call(self.query, self.key, self.value)
        self.assertEqual(out.shape, (self.batch, self.t_q, self.d_model))
        self.assertEqual(weights.shape, (self.batch, self.heads, self.t_q, self.t_k))

    def test_causal_mask_zeros_future_weights(self) -> None:
        seq_len = 6
        x = np.ones((1, seq_len, self.d_model), dtype=np.float32)
        mask = _causal_additive_mask(seq_len)
        attn = MultiHeadAttention(self.d_model, self.heads, seed=11)
        _, weights = attn.call(x, x, x, mask=mask)

        future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        future_weights = weights[0, :, future]
        self.assertTrue(np.allclose(future_weights, 0.0, atol=1e-7))

    def test_mqa_weight_sharing(self) -> None:
        layer = MultiQueryAttention(self.d_model, self.heads, seed=12)
        # Shared K/V means projection width equals one head depth.
        self.assertEqual(layer.w_k.shape[1], self.d_model // self.heads)
        self.assertEqual(layer.w_v.shape[1], self.d_model // self.heads)

    def test_gqa_group_sharing(self) -> None:
        num_groups = 2
        layer = GroupedQueryAttention(self.d_model, self.heads, num_kv_groups=num_groups, seed=13)
        expected_width = num_groups * (self.d_model // self.heads)
        self.assertEqual(layer.w_k.shape[1], expected_width)
        self.assertEqual(layer.w_v.shape[1], expected_width)
        self.assertEqual(layer.heads_per_group, self.heads // num_groups)

    def test_mla_kv_cache_reduction(self) -> None:
        layer = MultiHeadLatentAttention(self.d_model, self.heads, kv_latent_dim=4, seed=14)
        full, latent = layer.kv_cache_size(batch_size=2, seq_len=64)
        self.assertLess(latent, full)

        _out, _weights = layer.call(self.query, self.key, self.value)
        self.assertIsNotNone(layer.last_kv_cache_shape)
        assert layer.last_kv_cache_shape is not None
        self.assertEqual(layer.last_kv_cache_shape[-1], 4)


if __name__ == "__main__":
    unittest.main()
