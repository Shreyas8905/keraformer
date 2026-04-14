"""Tests for transformer block compositions."""

from __future__ import annotations

import unittest

import numpy as np

from keraformer.blocks import DecoderBlock, EncoderBlock, EncoderDecoderBlock, ParallelBlock


def _finite_difference_grad_norm(fn, x: np.ndarray, eps: float = 1e-3) -> float:
    """Approximate gradient norm of sum(fn(x)) w.r.t. x via finite differences."""
    grad = np.zeros_like(x, dtype=np.float32)
    flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)

    for idx in range(flat.shape[0]):
        orig = flat[idx]
        flat[idx] = orig + eps
        y_pos = float(np.sum(fn(x)))
        flat[idx] = orig - eps
        y_neg = float(np.sum(fn(x)))
        flat[idx] = orig
        grad_flat[idx] = (y_pos - y_neg) / (2.0 * eps)

    return float(np.linalg.norm(grad_flat))


class BlockTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(77)
        self.batch = 2
        self.src_len = 6
        self.tgt_len = 5
        self.d_model = 16
        self.heads = 4
        self.src = rng.normal(size=(self.batch, self.src_len, self.d_model)).astype(np.float32)
        self.tgt = rng.normal(size=(self.batch, self.tgt_len, self.d_model)).astype(np.float32)

    def test_encoder_block_shape(self) -> None:
        block = EncoderBlock(d_model=self.d_model, num_heads=self.heads, pre_norm=True, seed=1)
        out = block.call(self.src)
        self.assertEqual(out.shape, self.src.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_decoder_block_shape(self) -> None:
        block = DecoderBlock(d_model=self.d_model, num_heads=self.heads, pre_norm=True, seed=2)
        out = block.call(self.tgt, encoder_output=self.src)
        self.assertEqual(out.shape, self.tgt.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_encoder_decoder_block_shapes(self) -> None:
        block = EncoderDecoderBlock(d_model=self.d_model, num_heads=self.heads, seed=3)
        enc_out, dec_out = block.call(self.src, self.tgt)
        self.assertEqual(enc_out.shape, self.src.shape)
        self.assertEqual(dec_out.shape, self.tgt.shape)

    def test_parallel_block_shape(self) -> None:
        block = ParallelBlock(d_model=self.d_model, num_heads=self.heads, seed=4)
        out = block.call(self.src)
        self.assertEqual(out.shape, self.src.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_encoder_gradient_flow_non_zero(self) -> None:
        block = EncoderBlock(d_model=self.d_model, num_heads=self.heads, pre_norm=True, seed=5)

        x = self.src[:1, :2, :4].copy()

        def fn(inp: np.ndarray) -> np.ndarray:
            padded = np.zeros((1, 2, self.d_model), dtype=np.float32)
            padded[..., :4] = inp
            y = block.call(padded)
            return y[..., :4]

        norm = _finite_difference_grad_norm(fn, x)
        self.assertGreater(norm, 0.0)

    def test_parallel_gradient_flow_non_zero(self) -> None:
        block = ParallelBlock(d_model=self.d_model, num_heads=self.heads, seed=6)

        x = self.src[:1, :2, :4].copy()

        def fn(inp: np.ndarray) -> np.ndarray:
            padded = np.zeros((1, 2, self.d_model), dtype=np.float32)
            padded[..., :4] = inp
            y = block.call(padded)
            return y[..., :4]

        norm = _finite_difference_grad_norm(fn, x)
        self.assertGreater(norm, 0.0)


if __name__ == "__main__":
    unittest.main()
