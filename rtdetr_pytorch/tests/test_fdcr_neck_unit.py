"""Focused unit tests for the FDCR post-fusion refinement block."""

import math
import unittest

import torch
import torch.nn as nn

from tests._support import prepare_imports

prepare_imports()

from src.zoo.rtdetr.fdcr_neck import (  # noqa: E402
    FDCRNeck,
    FrequencyDecoupledContextResidual,
)


class FDCRNeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_frequency_decomposition_and_branch_contract(self):
        module = FrequencyDecoupledContextResidual(
            hidden_dim=8).eval()
        features = torch.randn(2, 8, 15, 19)

        with torch.inference_mode():
            output, aux = module(features, return_aux=True)

        self.assertEqual(tuple(output.shape), tuple(features.shape))
        for name in (
                'low_frequency', 'high_frequency', 'low_context',
                'high_detail', 'residual'):
            self.assertEqual(tuple(aux[name].shape), tuple(features.shape))
        self.assertTrue(torch.allclose(
            aux['low_frequency'] + aux['high_frequency'], features,
            atol=1e-7, rtol=1e-7))

        self.assertEqual(module.low_pass.kernel_size, 3)
        self.assertEqual(module.low_pass.stride, 1)
        self.assertEqual(module.low_pass.padding, 1)
        self.assertEqual(module.low_depthwise.kernel_size, (5, 5))
        self.assertEqual(module.low_depthwise.padding, (2, 2))
        self.assertEqual(module.low_depthwise.groups, 8)
        self.assertEqual(module.high_depthwise.kernel_size, (3, 3))
        self.assertEqual(module.high_depthwise.padding, (1, 1))
        self.assertEqual(module.high_depthwise.groups, 8)
        self.assertEqual(module.fuse.in_channels, 16)
        self.assertEqual(module.fuse.out_channels, 8)

    def test_dynamic_shapes_preserve_resolution(self):
        module = FDCRNeck(hidden_dim=4).eval()
        for height, width in ((1, 1), (7, 11), (30, 38), (80, 80)):
            with self.subTest(shape=(height, width)), torch.inference_mode():
                features = torch.randn(2, 4, height, width)
                output = module(features)
                self.assertEqual(tuple(output.shape), tuple(features.shape))

    def test_channelwise_inverse_tanh_initialization_and_bounds(self):
        module = FrequencyDecoupledContextResidual(
            hidden_dim=7, gamma_max=0.30, gamma_init=0.05)
        expected_raw = math.atanh(0.05 / 0.30)

        self.assertEqual(tuple(module.raw_gamma.shape), (1, 7, 1, 1))
        self.assertTrue(torch.allclose(
            module.raw_gamma,
            torch.full_like(module.raw_gamma, expected_raw),
            atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(
            module.effective_gamma(),
            torch.full_like(module.raw_gamma, 0.05),
            atol=1e-7, rtol=1e-7))

        with torch.no_grad():
            module.raw_gamma[0, :3].fill_(-100.0)
            module.raw_gamma[0, 3:].fill_(100.0)
        gamma = module.effective_gamma()
        self.assertTrue((gamma[0, :3] < 0).all())
        self.assertTrue((gamma[0, 3:] > 0).all())
        self.assertTrue((gamma.abs() <= 0.30).all())

    def test_zero_gamma_strictly_restores_input(self):
        module = FrequencyDecoupledContextResidual(
            hidden_dim=8).eval()
        features = torch.randn(2, 8, 15, 19)
        with torch.no_grad():
            module.raw_gamma.zero_()
            output = module(features)
        self.assertTrue(torch.equal(output, features))

    def test_forward_backward_reaches_every_parameter(self):
        module = FrequencyDecoupledContextResidual(
            hidden_dim=8).train()
        features = torch.randn(2, 8, 13, 17, requires_grad=True)
        output = module(features)

        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0.0)

        problems = []
        for name, parameter in module.named_parameters():
            if (parameter.grad is None
                    or not torch.isfinite(parameter.grad).all()
                    or parameter.grad.abs().sum().item() == 0.0):
                problems.append(name)
        self.assertEqual(problems, [])

    def test_no_normalization_attention_or_gate(self):
        module = FrequencyDecoupledContextResidual(hidden_dim=8)
        forbidden = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.LayerNorm, nn.MultiheadAttention,
        )
        self.assertFalse(any(
            isinstance(child, forbidden) for child in module.modules()))
        names = ' '.join(name.lower() for name, _ in module.named_modules())
        self.assertNotIn('attention', names)
        self.assertNotIn('gate', names)

    def test_configuration_and_input_validation(self):
        for kwargs in (
                {'hidden_dim': 0},
                {'gamma_max': 0.0},
                {'gamma_max': 0.30, 'gamma_init': 0.30},
                {'gamma_max': 0.30, 'gamma_init': -0.30},
                {'gamma_max': float('nan')},
                {'gamma_init': float('inf')}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    FrequencyDecoupledContextResidual(**kwargs)

        module = FrequencyDecoupledContextResidual(hidden_dim=4)
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 3, 8, 8))
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 4, 8))
        with self.assertRaises(RuntimeError):
            module(torch.ones(1, 4, 8, 8, dtype=torch.int64))


if __name__ == '__main__':
    unittest.main()
