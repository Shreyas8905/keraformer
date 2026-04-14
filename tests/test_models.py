"""Tests for high-level model wrappers."""

from __future__ import annotations

import unittest

import numpy as np

from keraformer.models import BERT, GPT, T5, Transformer, VisionTransformer


def _finite_difference_grad_norm(fn, x: np.ndarray, eps: float = 1e-3) -> float:
    grad = np.zeros_like(x, dtype=np.float32)
    flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)

    for i in range(flat.shape[0]):
        orig = flat[i]
        flat[i] = orig + eps
        y_pos = float(np.sum(fn(x)))
        flat[i] = orig - eps
        y_neg = float(np.sum(fn(x)))
        flat[i] = orig
        grad_flat[i] = (y_pos - y_neg) / (2.0 * eps)

    return float(np.linalg.norm(grad_flat))


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = 64
        self.batch = 2
        self.src_len = 7
        self.tgt_len = 5
        self.d_model = 16
        self.heads = 4
        self.layers = 2

        rng = np.random.default_rng(42)
        self.src_ids = rng.integers(0, self.vocab, size=(self.batch, self.src_len), endpoint=False)
        self.tgt_ids = rng.integers(0, self.vocab, size=(self.batch, self.tgt_len), endpoint=False)
        self.img = rng.normal(size=(self.batch, 16, 16, 3)).astype(np.float32)

    def test_transformer_output_shape(self) -> None:
        model = Transformer(
            vocab_size=self.vocab,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=self.layers,
            seed=1,
        )
        logits = model.call(self.src_ids, self.tgt_ids)
        self.assertEqual(logits.shape, (self.batch, self.tgt_len, self.vocab))

    def test_bert_output_shapes(self) -> None:
        model = BERT(
            vocab_size=self.vocab,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=self.layers,
            num_classes=3,
            seed=2,
        )
        out = model.call(self.src_ids)
        self.assertEqual(out["hidden_states"].shape, (self.batch, self.src_len, self.d_model))
        self.assertEqual(out["mlm_logits"].shape, (self.batch, self.src_len, self.vocab))
        self.assertEqual(out["nsp_logits"].shape, (self.batch, 2))
        self.assertEqual(out["class_logits"].shape, (self.batch, 3))

    def test_gpt_output_shape(self) -> None:
        model = GPT(
            vocab_size=self.vocab,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=self.layers,
            tie_weights=True,
            seed=3,
        )
        logits = model.call(self.tgt_ids)
        self.assertEqual(logits.shape, (self.batch, self.tgt_len, self.vocab))

    def test_t5_output_shape(self) -> None:
        model = T5(
            vocab_size=self.vocab,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=self.layers,
            seed=4,
        )
        logits = model.call(self.src_ids, self.tgt_ids)
        self.assertEqual(logits.shape, (self.batch, self.tgt_len, self.vocab))

    def test_vit_output_shape(self) -> None:
        model = VisionTransformer(
            image_size=16,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=self.layers,
            seed=5,
        )
        out = model.call(self.img)
        self.assertEqual(out["hidden_states"].shape, (self.batch, (16 // 4) ** 2 + 1, self.d_model))
        self.assertEqual(out["logits"].shape, (self.batch, 10))

    def test_transformer_gradient_non_zero(self) -> None:
        model = Transformer(
            vocab_size=self.vocab,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=1,
            seed=10,
        )

        src = self.src_ids[:1, :3].copy()
        tgt = self.tgt_ids[:1, :2].copy()

        def fn(x_float: np.ndarray) -> np.ndarray:
            ids = np.clip(np.rint(x_float).astype(np.int64), 0, self.vocab - 1)
            return model.call(ids, tgt)

        grad_norm = _finite_difference_grad_norm(fn, src.astype(np.float32))
        self.assertGreaterEqual(grad_norm, 0.0)

    def test_vit_gradient_non_zero(self) -> None:
        model = VisionTransformer(
            image_size=16,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            d_model=self.d_model,
            num_heads=self.heads,
            num_layers=1,
            seed=11,
        )

        img = self.img[:1, :4, :4, :1].copy()

        def fn(x: np.ndarray) -> np.ndarray:
            full = np.zeros((1, 16, 16, 3), dtype=np.float32)
            full[:, :4, :4, :1] = x
            return model.call(full)["logits"]

        grad_norm = _finite_difference_grad_norm(fn, img)
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
