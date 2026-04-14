"""Tests for optimizer implementations and schedules."""

from __future__ import annotations

import unittest

import numpy as np

from keraformer.optimizers import Adafactor, AdamW, Lion, NoamSchedule


class OptimizerTests(unittest.TestCase):
    def test_adamw_state_shapes_and_step(self) -> None:
        opt = AdamW(lr=1e-2)
        params = np.ones((3, 4), dtype=np.float32)
        grads = np.full((3, 4), 0.5, dtype=np.float32)

        updated = opt.step(params, grads)
        self.assertEqual(updated.shape, params.shape)
        self.assertIsNotNone(opt.m)
        self.assertIsNotNone(opt.v)
        assert opt.m is not None and opt.v is not None
        self.assertEqual(opt.m.shape, params.shape)
        self.assertEqual(opt.v.shape, params.shape)

    def test_lion_state_shapes_and_step(self) -> None:
        opt = Lion(lr=1e-3)
        params = np.ones((5,), dtype=np.float32)
        grads = np.linspace(-1.0, 1.0, 5).astype(np.float32)

        updated = opt.step(params, grads)
        self.assertEqual(updated.shape, params.shape)
        self.assertIsNotNone(opt.m)
        assert opt.m is not None
        self.assertEqual(opt.m.shape, params.shape)

    def test_adafactor_factored_state_shapes(self) -> None:
        opt = Adafactor(lr=1e-2, factored=True)
        params = np.ones((6, 4), dtype=np.float32)
        grads = np.full((6, 4), 0.1, dtype=np.float32)

        updated = opt.step(params, grads)
        self.assertEqual(updated.shape, params.shape)
        self.assertIsNotNone(opt.vr)
        self.assertIsNotNone(opt.vc)
        assert opt.vr is not None and opt.vc is not None
        self.assertEqual(opt.vr.shape, (6,))
        self.assertEqual(opt.vc.shape, (4,))

    def test_adafactor_unfactored_state_shapes(self) -> None:
        opt = Adafactor(lr=1e-2, factored=False)
        params = np.ones((7,), dtype=np.float32)
        grads = np.full((7,), 0.2, dtype=np.float32)

        updated = opt.step(params, grads)
        self.assertEqual(updated.shape, params.shape)
        self.assertIsNotNone(opt.v)
        assert opt.v is not None
        self.assertEqual(opt.v.shape, params.shape)

    def test_noam_schedule_reference_values(self) -> None:
        schedule = NoamSchedule(d_model=512, warmup_steps=4000)

        v1 = schedule(1)
        vw = schedule(4000)
        v2w = schedule(8000)

        expected_1 = (512 ** -0.5) * min(1.0, 1.0 / (4000 ** 1.5))
        expected_w = (512 ** -0.5) * min(4000 ** -0.5, 4000 * (4000 ** -1.5))
        expected_2w = (512 ** -0.5) * min(8000 ** -0.5, 8000 * (4000 ** -1.5))

        self.assertAlmostEqual(v1, expected_1, places=12)
        self.assertAlmostEqual(vw, expected_w, places=12)
        self.assertAlmostEqual(v2w, expected_2w, places=12)
        self.assertGreater(vw, v1)
        self.assertLess(v2w, vw)


if __name__ == "__main__":
    unittest.main()
