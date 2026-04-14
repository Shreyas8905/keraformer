"""Tests for normalization layers."""

from __future__ import annotations

import math
import unittest

import numpy as np

from keraformer.normalization import DeepNorm, GroupNorm, LayerNorm, RMSNorm


class NormalizationTests(unittest.TestCase):
    def test_layer_norm_matches_manual_formula(self) -> None:
        x = np.array(
            [[[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]],
            dtype=np.float32,
        )
        norm = LayerNorm(d_model=3, eps=1e-5, affine=False)
        y = norm(x)

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)

    def test_rms_norm_matches_analytic_formula(self) -> None:
        x = np.array([[[-2.0, 0.0, 2.0, 4.0]]], dtype=np.float32)
        norm = RMSNorm(d_model=4, eps=1e-8)
        y = norm(x)

        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1e-8)
        expected = x / rms
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)

    def test_group_norm_shape_and_zero_mean_per_group(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 3, 8)).astype(np.float32)
        norm = GroupNorm(d_model=8, num_groups=4, eps=1e-5, affine=False)
        y = norm(x)

        self.assertEqual(y.shape, x.shape)
        grouped = y.reshape(2, 3, 4, 2)
        means = np.mean(grouped, axis=-1)
        np.testing.assert_allclose(means, np.zeros_like(means), atol=2e-5)

    def test_deep_norm_alpha_and_shape(self) -> None:
        x = np.ones((1, 2, 4), dtype=np.float32)
        sub = np.zeros((1, 2, 4), dtype=np.float32)
        dn = DeepNorm(d_model=4, num_layers=16)
        y = dn(x, sub)

        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(dn.alpha, (2.0 * 16.0) ** 0.25, places=7)

    def test_invalid_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            LayerNorm(d_model=0)

        with self.assertRaises(ValueError):
            RMSNorm(d_model=4, eps=0.0)

        with self.assertRaises(ValueError):
            GroupNorm(d_model=7, num_groups=4)

        with self.assertRaises(ValueError):
            DeepNorm(d_model=4, num_layers=0)

    def test_layer_norm_last_axis_near_unit_variance(self) -> None:
        rng = np.random.default_rng(21)
        x = rng.normal(size=(4, 5, 6)).astype(np.float32)
        y = LayerNorm(d_model=6, affine=False)(x)
        var = np.var(y, axis=-1)
        self.assertTrue(np.all(np.isfinite(var)))
        self.assertTrue(np.all(np.abs(var - 1.0) < 1e-3))

    def test_rms_norm_last_axis_rms_is_one(self) -> None:
        rng = np.random.default_rng(22)
        x = rng.normal(size=(3, 2, 4)).astype(np.float32)
        y = RMSNorm(d_model=4)(x)
        rms = np.sqrt(np.mean(np.square(y), axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
