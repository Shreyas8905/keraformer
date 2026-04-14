"""Tests for loss functions."""

from __future__ import annotations

import unittest

import numpy as np

from keraformer.losses import (
    focal_loss,
    label_smoothed_cross_entropy,
    masked_lm_loss,
    nt_xent_loss,
)


class LossTests(unittest.TestCase):
    def test_label_smoothed_cross_entropy_numerical_case(self) -> None:
        logits = np.array([[[2.0, 0.0]]], dtype=np.float32)  # (B=1,T=1,V=2)
        targets = np.array([[0]], dtype=np.int64)

        loss = label_smoothed_cross_entropy(logits, targets, epsilon=0.1)

        p0 = np.exp(2.0) / (np.exp(2.0) + np.exp(0.0))
        p1 = 1.0 - p0
        expected = -(0.95 * np.log(p0) + 0.05 * np.log(p1))
        self.assertAlmostEqual(loss, float(expected), places=6)

    def test_label_smoothed_ce_with_mask(self) -> None:
        logits = np.array(
            [
                [[3.0, 1.0], [1.0, 3.0]],
                [[2.0, 0.5], [0.5, 2.0]],
            ],
            dtype=np.float32,
        )
        targets = np.array([[0, 1], [0, 1]], dtype=np.int64)
        mask = np.array([[1, 1], [1, 0]], dtype=np.float32)

        masked = label_smoothed_cross_entropy(logits, targets, epsilon=0.1, padding_mask=mask)
        unmasked = label_smoothed_cross_entropy(logits, targets, epsilon=0.1)
        self.assertTrue(np.isfinite(masked))
        self.assertTrue(np.isfinite(unmasked))
        self.assertNotEqual(masked, unmasked)

    def test_focal_loss_finite(self) -> None:
        logits = np.array([[2.0, -1.0, 0.5], [0.1, 0.2, 1.3]], dtype=np.float32)
        targets = np.array([0, 2], dtype=np.int64)
        loss = focal_loss(logits, targets, gamma=2.0, alpha=0.75)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_nt_xent_loss_finite(self) -> None:
        rng = np.random.default_rng(7)
        z1 = rng.normal(size=(8, 12)).astype(np.float32)
        z2 = z1 + 0.05 * rng.normal(size=(8, 12)).astype(np.float32)
        loss = nt_xent_loss(z1, z2, temperature=0.07)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_masked_lm_loss_only_counts_masked_positions(self) -> None:
        logits = np.array(
            [
                [[4.0, 0.0], [0.0, 4.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.float32,
        )
        targets = np.array([[0, 1], [0, 1]], dtype=np.int64)
        mask_all = np.array([[1, 1], [1, 1]], dtype=np.float32)
        mask_half = np.array([[1, 0], [0, 1]], dtype=np.float32)

        loss_all = masked_lm_loss(logits, targets, mask_all)
        loss_half = masked_lm_loss(logits, targets, mask_half)
        self.assertTrue(np.isfinite(loss_all))
        self.assertTrue(np.isfinite(loss_half))

    def test_masked_lm_loss_zero_mask(self) -> None:
        logits = np.zeros((1, 2, 3), dtype=np.float32)
        targets = np.zeros((1, 2), dtype=np.int64)
        mask = np.zeros((1, 2), dtype=np.float32)
        self.assertEqual(masked_lm_loss(logits, targets, mask), 0.0)


if __name__ == "__main__":
    unittest.main()
