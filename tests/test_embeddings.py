"""Tests for token and positional embeddings."""

from __future__ import annotations

import math
import unittest

import numpy as np

from keraformer.embeddings import TokenEmbedding
from keraformer.embeddings.positional import (
    LearnedPositionalEncoding,
    RelativePositionBias,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    no_positional_encoding,
    sinusoidal_positional_encoding,
    t5_relative_position_bucket,
)


class EmbeddingTests(unittest.TestCase):
    def test_token_embedding_shape_and_scaling(self) -> None:
        embed = TokenEmbedding(vocab_size=8, d_model=6, seed=123)
        token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        out = embed(token_ids)

        self.assertEqual(out.shape, (2, 3, 6))
        expected = embed.embedding_matrix[token_ids] * math.sqrt(6.0)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_sinusoidal_values_small_case(self) -> None:
        pe = sinusoidal_positional_encoding(seq_len=2, d_model=4)
        self.assertEqual(pe.shape, (2, 4))

        # pos=0 gives sin(0)=0 and cos(0)=1 for all frequencies.
        np.testing.assert_allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32), atol=1e-7)

        # First pair uses frequency 1.0.
        self.assertAlmostEqual(float(pe[1, 0]), math.sin(1.0), places=6)
        self.assertAlmostEqual(float(pe[1, 1]), math.cos(1.0), places=6)

    def test_learned_positional_slice(self) -> None:
        learned = LearnedPositionalEncoding(max_len=10, d_model=7, seed=42)
        pos = learned(4)
        self.assertEqual(pos.shape, (4, 7))
        np.testing.assert_allclose(pos, learned.weights[:4])

    def test_rope_same_position_preserves_dot_product(self) -> None:
        rng = np.random.default_rng(5)
        q = rng.normal(size=(1, 1, 3, 8)).astype(np.float32)
        k = rng.normal(size=(1, 1, 3, 8)).astype(np.float32)

        q_rot = apply_rope(q)
        k_rot = apply_rope(k)

        # Same-position rotation is orthonormal; pairwise dot products are preserved.
        base_dot = np.sum(q * k, axis=-1)
        rot_dot = np.sum(q_rot * k_rot, axis=-1)
        np.testing.assert_allclose(rot_dot, base_dot, rtol=1e-5, atol=1e-5)

    def test_alibi_bias_shape_and_monotonicity(self) -> None:
        slopes = alibi_slopes(4)
        self.assertEqual(slopes.shape, (4,))

        bias = alibi_bias(num_heads=4, query_len=3, key_len=3)
        self.assertEqual(bias.shape, (4, 3, 3))
        # Distance grows from j=0 to j=2 for query i=0, so bias decreases.
        self.assertGreaterEqual(float(bias[0, 0, 0]), float(bias[0, 0, 1]))
        self.assertGreaterEqual(float(bias[0, 0, 1]), float(bias[0, 0, 2]))

    def test_relative_bucket_and_bias_shape(self) -> None:
        rel = np.array([[-2, -1, 0, 1, 2]])
        buckets = t5_relative_position_bucket(rel, num_buckets=16, max_distance=32)
        self.assertEqual(buckets.shape, rel.shape)
        self.assertTrue(np.all(buckets >= 0))
        self.assertTrue(np.all(buckets < 16))

        bias_layer = RelativePositionBias(num_heads=6, num_buckets=16, max_distance=32, seed=9)
        bias = bias_layer(query_len=5, key_len=7)
        self.assertEqual(bias.shape, (6, 5, 7))

    def test_no_positional_encoding_identity(self) -> None:
        x = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        y = no_positional_encoding(x)
        np.testing.assert_array_equal(y, x)


if __name__ == "__main__":
    unittest.main()
