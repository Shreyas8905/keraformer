"""Tests for mask helpers."""

from __future__ import annotations

import math
import unittest

from keraformer.masks import causal_mask, padding_mask, prefix_lm_mask


class MaskTests(unittest.TestCase):
    def test_causal_mask_values(self) -> None:
        mask = causal_mask(4)

        self.assertEqual(len(mask), 4)
        self.assertEqual(mask[0], [0.0, float("-inf"), float("-inf"), float("-inf")])
        self.assertEqual(mask[1], [0.0, 0.0, float("-inf"), float("-inf")])
        self.assertEqual(mask[2], [0.0, 0.0, 0.0, float("-inf")])
        self.assertEqual(mask[3], [0.0, 0.0, 0.0, 0.0])

    def test_padding_mask_shape_and_values(self) -> None:
        mask = padding_mask([[1, 0, 2], [0, 0, 3]], pad_token_id=0)

        self.assertEqual(len(mask), 2)
        self.assertEqual(len(mask[0]), 1)
        self.assertEqual(len(mask[0][0]), 1)
        self.assertEqual(mask[0][0][0], [0.0, float("-inf"), 0.0])
        self.assertEqual(mask[1][0][0], [float("-inf"), float("-inf"), 0.0])

    def test_prefix_lm_mask_values(self) -> None:
        mask = prefix_lm_mask(seq_len=5, prefix_len=2)

        self.assertEqual(mask[0], [0.0, 0.0, float("-inf"), float("-inf"), float("-inf")])
        self.assertEqual(mask[1], [0.0, 0.0, float("-inf"), float("-inf"), float("-inf")])
        self.assertEqual(mask[2], [0.0, 0.0, 0.0, float("-inf"), float("-inf")])
        self.assertEqual(mask[3], [0.0, 0.0, 0.0, 0.0, float("-inf")])
        self.assertEqual(mask[4], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_masks_reject_invalid_lengths(self) -> None:
        with self.assertRaises(ValueError):
            causal_mask(-1)

        with self.assertRaises(ValueError):
            prefix_lm_mask(3, -1)

        with self.assertRaises(ValueError):
            prefix_lm_mask(2, 3)

    def test_mask_values_are_finite_or_negative_infinity(self) -> None:
        mask = causal_mask(3)
        for row in mask:
            for value in row:
                self.assertTrue(value == 0.0 or math.isinf(value))


if __name__ == "__main__":
    unittest.main()
