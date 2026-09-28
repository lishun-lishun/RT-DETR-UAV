"""Focused numerical tests for Background-Orthogonal Residual Neck."""

import unittest

import torch
import torch.nn.functional as F

from tests._support import prepare_imports

prepare_imports()

from src.zoo.rtdetr.bor_neck import (  # noqa: E402
    BackgroundOrthogonalResidual,
)


class BORNeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ring_prototype_is_prescribed_pooling_formula(self):
        module = BackgroundOrthogonalResidual(4)
        features = torch.randn(2, 4, 13, 17)
        actual = module.ring_prototype(features)
        expected = (
            49 * F.avg_pool2d(features, 7, stride=1, padding=3,
                              count_include_pad=True)
            - 9 * F.avg_pool2d(features, 3, stride=1, padding=1,
                               count_include_pad=True)
        ) / 40
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_parallel_input_has_near_zero_orthogonal_component(self):
        module = BackgroundOrthogonalResidual(4)
        features = torch.ones(2, 4, 5, 7)
        _, orthogonal = module.orthogonal_decompose(features, features)
        self.assertLess(orthogonal.abs().max().item(), 1e-6)

    def test_orthogonal_input_is_preserved(self):
        module = BackgroundOrthogonalResidual(2)
        features = torch.zeros(1, 2, 3, 4)
        background = torch.zeros_like(features)
        features[:, 0] = 2.0
        background[:, 1] = 3.0
        parallel, orthogonal = module.orthogonal_decompose(
            features, background)
        self.assertTrue(torch.equal(parallel, torch.zeros_like(parallel)))
        self.assertTrue(torch.equal(orthogonal, features))

    def test_alpha_dynamic_shapes_backward_and_debug(self):
        module = BackgroundOrthogonalResidual(8, debug=True)
        self.assertAlmostEqual(module.effective_alpha().item(), 0.05, places=7)
        for height, width in ((60, 60), (80, 72), (100, 96)):
            features = torch.randn(
                1, 8, height, width, requires_grad=True)
            enhanced, aux = module(features, return_aux=True)
            self.assertEqual(enhanced.shape, features.shape)
            self.assertEqual(aux['novelty_ratio'].shape,
                             (1, 1, height, width))
            self.assertEqual(aux['gate'].shape, (1, 1, height, width))
            self.assertTrue(torch.isfinite(enhanced).all())
            self.assertTrue(torch.isfinite(aux['novelty_ratio']).all())
            enhanced.square().mean().backward()
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertTrue(all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()))
            module.zero_grad(set_to_none=True)

        self.assertEqual(set(module.last_debug_stats), {
            'novelty_ratio_mean', 'novelty_ratio_std',
            'gate_mean', 'gate_std', 'alpha_eff',
            'orthogonal_residual_norm', 'base_n3_norm',
        })
        self.assertTrue(all(torch.isfinite(value)
                            for value in module.last_debug_stats.values()))

    def test_validation_rejects_invalid_kernel_or_channel(self):
        with self.assertRaises(ValueError):
            BackgroundOrthogonalResidual(8, outer_kernel=6)
        with self.assertRaises(ValueError):
            BackgroundOrthogonalResidual(8, outer_kernel=3, inner_kernel=3)
        module = BackgroundOrthogonalResidual(8)
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 7, 8, 8))


if __name__ == '__main__':
    unittest.main()
