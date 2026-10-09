"""Focused unit tests for the Extrema-Sensitive Downsampling Residual."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'src' / 'zoo' / 'rtdetr' / 'esdr_neck.py'
SPEC = importlib.util.spec_from_file_location('esdr_neck_unit', MODULE_PATH)
ESDR_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ESDR_MODULE)
ExtremaSensitiveDownsample = ESDR_MODULE.ExtremaSensitiveDownsample


class ExtremaSensitiveDownsampleTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(37)

    def test_exact_max_pool_and_residual_formula(self):
        block = ExtremaSensitiveDownsample(
            hidden_dim=2, beta_max=0.20, beta_init=0.02)
        with torch.no_grad():
            block.project.weight.zero_()
            block.project.bias.copy_(torch.tensor([0.5, -0.25]))
            block.project.weight[0, 0, 0, 0] = 2.0
            block.project.weight[1, 1, 0, 0] = -3.0

        source = torch.tensor([
            [
                [[1., 4., 2., 0.], [3., -1., 5., 2.],
                 [0., 6., 1., 7.], [2., 3., 4., 1.]],
                [[-1., 2., 4., 3.], [5., 0., -2., 6.],
                 [1., 8., 3., 0.], [7., 2., 9., 4.]],
            ]
        ])
        base = torch.randn(1, 2, 2, 2)

        output, aux = block(source, base, return_aux=True)
        expected_pool = torch.tensor([
            [[[4., 5.], [6., 7.]], [[5., 6.], [8., 9.]]]
        ])
        expected_extrema = torch.stack((
            2.0 * expected_pool[:, 0] + 0.5,
            -3.0 * expected_pool[:, 1] - 0.25,
        ), dim=1)
        expected = base + block.effective_beta() * expected_extrema

        self.assertTrue(torch.equal(aux['max_pooled'], expected_pool))
        self.assertTrue(torch.equal(aux['extrema'], expected_extrema))
        self.assertTrue(torch.allclose(output, expected, atol=0, rtol=0))
        self.assertEqual(tuple(output.shape), tuple(base.shape))

    def test_beta_shape_inverse_initialization_and_bound(self):
        block = ExtremaSensitiveDownsample(
            hidden_dim=7, beta_max=0.20, beta_init=0.02)
        expected_raw = math.atanh(0.02 / 0.20)

        self.assertEqual(tuple(block.raw_beta.shape), (1, 7, 1, 1))
        self.assertTrue(torch.allclose(
            block.raw_beta,
            torch.full_like(block.raw_beta, expected_raw),
            atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(
            block.effective_beta(),
            torch.full_like(block.raw_beta, 0.02),
            atol=1e-7, rtol=0))
        with torch.no_grad():
            block.raw_beta.fill_(100.0)
        self.assertTrue(bool(torch.all(block.effective_beta() <= 0.20)))

    def test_zero_beta_is_bitwise_equal_to_original_base(self):
        block = ExtremaSensitiveDownsample(hidden_dim=4)
        with torch.no_grad():
            block.raw_beta.zero_()
        source = torch.randn(2, 4, 12, 16)
        base = torch.randn(2, 4, 6, 8)
        output = block(source, base)
        self.assertTrue(torch.equal(output, base))

    def test_dynamic_feature_shapes_for_480_640_800_inputs(self):
        block34 = ExtremaSensitiveDownsample(hidden_dim=3)
        block45 = ExtremaSensitiveDownsample(hidden_dim=3)
        self.assertIsNot(block34.raw_beta, block45.raw_beta)
        self.assertIsNot(block34.project.weight, block45.project.weight)

        for image_size in (480, 640, 800):
            with self.subTest(image_size=image_size):
                n3_size = image_size // 8
                n4_size = image_size // 16
                n5_size = image_size // 32
                n3 = torch.randn(1, 3, n3_size, n3_size)
                base34 = torch.randn(1, 3, n4_size, n4_size)
                down34 = block34(n3, base34)
                base45 = torch.randn(1, 3, n5_size, n5_size)
                down45 = block45(down34, base45)
                self.assertEqual(
                    tuple(down34.shape), (1, 3, n4_size, n4_size))
                self.assertEqual(
                    tuple(down45.shape), (1, 3, n5_size, n5_size))

    def test_backward_reaches_every_parameter_with_finite_nonzero_grad(self):
        block = ExtremaSensitiveDownsample(hidden_dim=5)
        source = torch.randn(2, 5, 10, 14, requires_grad=True)
        base = torch.randn(2, 5, 5, 7, requires_grad=True)
        output = block(source, base)
        output.square().mean().backward()

        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(base.grad)
        self.assertTrue(bool(torch.isfinite(source.grad).all()))
        self.assertTrue(bool(torch.isfinite(base.grad).all()))
        for name, parameter in block.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_strict_input_validation(self):
        block = ExtremaSensitiveDownsample(hidden_dim=4)
        valid_source = torch.randn(2, 4, 10, 12)
        valid_base = torch.randn(2, 4, 5, 6)

        invalid_cases = (
            (torch.randn(2, 4, 9, 12), torch.randn(2, 4, 4, 6)),
            (torch.randn(2, 4, 10, 11), torch.randn(2, 4, 5, 5)),
            (valid_source, torch.randn(2, 4, 5, 5)),
            (valid_source, torch.randn(1, 4, 5, 6)),
            (valid_source, valid_base.double()),
            (torch.randn(2, 3, 10, 12), torch.randn(2, 4, 5, 6)),
        )
        for source, base in invalid_cases:
            with self.subTest(source=tuple(source.shape), base=tuple(base.shape)):
                with self.assertRaises(RuntimeError):
                    block(source, base)

        with self.assertRaises(RuntimeError):
            block([valid_source], valid_base)
        with self.assertRaises(RuntimeError):
            block(valid_source, [valid_base])

    def test_configuration_validation(self):
        invalid_kwargs = (
            {'hidden_dim': 0},
            {'hidden_dim': True},
            {'beta_max': 0.0},
            {'beta_max': float('nan')},
            {'beta_init': 0.20},
            {'beta_init': -0.20},
            {'beta_init': float('inf')},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ExtremaSensitiveDownsample(**kwargs)

    def test_contains_only_required_pool_convolution_and_layerscale(self):
        block = ExtremaSensitiveDownsample(hidden_dim=4)
        modules = list(block.modules())[1:]
        self.assertEqual(len(modules), 2)
        self.assertIsInstance(block.max_pool, nn.MaxPool2d)
        self.assertIsInstance(block.project, nn.Conv2d)
        self.assertEqual(block.project.kernel_size, (1, 1))
        self.assertFalse(any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                             for module in modules))
        forbidden_names = ('attention', 'attn', 'gate', 'norm')
        for name, _ in block.named_modules():
            lowered = name.lower()
            self.assertFalse(any(token in lowered for token in forbidden_names))


if __name__ == '__main__':
    unittest.main()
