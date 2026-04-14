"""Tests for feed-forward modules."""

from __future__ import annotations

import unittest

import numpy as np

from keraformer.feedforward import ConvFFN, FFN, GatedFFN, MoEFFN


class FeedForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 2
        self.seq = 5
        self.d_model = 12
        rng = np.random.default_rng(1234)
        self.x = rng.normal(size=(self.batch, self.seq, self.d_model)).astype(np.float32)

    def test_ffn_shapes_for_all_activations(self) -> None:
        for act in ("relu", "gelu", "silu"):
            ffn = FFN(d_model=self.d_model, d_ff=24, activation=act, seed=1)
            y = ffn(self.x)
            self.assertEqual(y.shape, self.x.shape)
            self.assertTrue(np.all(np.isfinite(y)))

    def test_gated_ffn_variants_shape(self) -> None:
        for variant in ("glu", "swiglu", "geglu"):
            layer = GatedFFN(d_model=self.d_model, d_ff=16, variant=variant, seed=2)
            y = layer(self.x)
            self.assertEqual(y.shape, self.x.shape)
            self.assertTrue(np.all(np.isfinite(y)))

    def test_moe_ffn_shape_and_aux_loss(self) -> None:
        moe = MoEFFN(d_model=self.d_model, d_ff=24, num_experts=4, top_k=2, seed=3)
        y = moe(self.x)
            
        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreaterEqual(moe.last_aux_loss, 0.0)

    def test_conv_ffn_shape(self) -> None:
        conv = ConvFFN(d_model=self.d_model, kernel_size=3, activation="gelu", seed=4)
        y = conv(self.x)

        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_invalid_configs_raise(self) -> None:
        with self.assertRaises(ValueError):
            FFN(d_model=0)

        with self.assertRaises(ValueError):
            GatedFFN(d_model=8, variant="bad")

        with self.assertRaises(ValueError):
            MoEFFN(d_model=8, num_experts=2, top_k=3)

        with self.assertRaises(ValueError):
            ConvFFN(d_model=8, kernel_size=2)


if __name__ == "__main__":
    unittest.main()
