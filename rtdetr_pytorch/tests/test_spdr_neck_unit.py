"""Regression tests for the retained SPDR implementation.

These tests intentionally exercise the retained SPDR block in isolation so
cleanup of an unrelated experiment cannot alter its PixelUnshuffle path.
"""

import math
import unittest

import torch
import torch.nn as nn

from tests._support import prepare_imports


prepare_imports()

from src.zoo.rtdetr.resample_neck import (  # noqa: E402
    SubpixelPreservingDownsample,
)


class SPDRNeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(211)

    def test_pixel_unshuffle_formula_and_shapes(self):
        block = SubpixelPreservingDownsample(channels=8).eval()
        source = torch.randn(2, 8, 14, 18)
        base = torch.randn(2, 8, 7, 9)

        with torch.inference_mode():
            output, aux = block(source, base, return_aux=True)

        self.assertEqual(tuple(aux['rearranged'].shape), (2, 32, 7, 9))
        self.assertEqual(aux['rearranged'].numel(), source.numel())
        self.assertEqual(aux['preserved'].shape, base.shape)
        self.assertTrue(torch.equal(aux['base'], base))
        self.assertTrue(torch.allclose(
            output, base + aux['beta'] * (aux['preserved'] - base)))
        self.assertTrue(torch.isfinite(output).all())

    def test_inverse_tanh_initialization_bounds_and_exact_zero(self):
        block = SubpixelPreservingDownsample(
            channels=8, beta_max=0.5, beta_init=0.05).eval()
        expected_raw = math.atanh(0.05 / 0.5)
        self.assertTrue(torch.allclose(
            block.raw_beta, torch.full_like(block.raw_beta, expected_raw)))
        self.assertTrue(torch.allclose(
            block.effective_beta(),
            torch.full_like(block.raw_beta, 0.05), atol=1e-7, rtol=1e-6))

        source = torch.randn(1, 8, 10, 12)
        base = torch.randn(1, 8, 5, 6)
        with torch.no_grad():
            block.raw_beta.zero_()
        self.assertTrue(torch.equal(block(source, base), base))
        with torch.no_grad():
            block.raw_beta.fill_(100.0)
        self.assertTrue((block.effective_beta() <= 0.5).all())

    def test_backward_reaches_every_parameter(self):
        block = SubpixelPreservingDownsample(channels=8)
        source = torch.randn(2, 8, 14, 18, requires_grad=True)
        base = torch.randn(2, 8, 7, 9, requires_grad=True)
        block(source, base).square().mean().backward()

        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_contract_validation_and_no_forbidden_layers(self):
        block = SubpixelPreservingDownsample(channels=8)
        with self.assertRaisesRegex(RuntimeError, 'must be even'):
            block(torch.randn(1, 8, 11, 12), torch.randn(1, 8, 5, 6))
        with self.assertRaisesRegex(RuntimeError, 'spatial size'):
            block(torch.randn(1, 8, 10, 12), torch.randn(1, 8, 6, 6))

        forbidden = (
            nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm,
            nn.MultiheadAttention, nn.Sigmoid, nn.Softmax,
        )
        self.assertFalse(any(
            isinstance(module, forbidden) for module in block.modules()))
        self.assertFalse(any(
            'gate' in name.lower() for name, _ in block.named_modules()))


if __name__ == '__main__':
    unittest.main()
