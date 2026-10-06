"""Focused unit tests for the LPRU and SPDR resampling blocks."""

import contextlib
import io
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests._support import prepare_imports

prepare_imports()

from src.zoo.rtdetr.resample_neck import (  # noqa: E402
    LearnablePixelReassemblyUpsample,
    SubpixelPreservingDownsample,
)


class ResampleNeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_lpru_pixel_reassembly_and_shapes(self):
        module = LearnablePixelReassemblyUpsample(channels=8).eval()
        source = torch.randn(2, 8, 15, 19)
        base = F.interpolate(source, scale_factor=2.0, mode='nearest')

        with torch.inference_mode():
            output, aux = module(source, base, return_aux=True)

        self.assertEqual(module.expand.out_channels, 4 * 8)
        self.assertEqual(module.pixel_shuffle.upscale_factor, 2)
        self.assertEqual(tuple(aux['rearranged'].shape), (2, 8, 30, 38))
        self.assertEqual(tuple(aux['learned'].shape), tuple(base.shape))
        self.assertEqual(tuple(output.shape), tuple(base.shape))
        self.assertEqual(module.depthwise.groups, 8)
        self.assertEqual(module.depthwise.kernel_size, (3, 3))
        self.assertEqual(module.depthwise.stride, (1, 1))
        self.assertEqual(module.depthwise.padding, (1, 1))

    def test_lpru_implicit_base_matches_original_nearest(self):
        module = LearnablePixelReassemblyUpsample(channels=4).eval()
        source = torch.randn(1, 4, 7, 11)
        explicit_base = F.interpolate(
            source, scale_factor=2.0, mode='nearest')
        with torch.inference_mode():
            implicit, implicit_aux = module(
                source, return_aux=True)
            explicit = module(source, explicit_base)
        self.assertTrue(torch.equal(implicit_aux['base'], explicit_base))
        self.assertTrue(torch.equal(implicit, explicit))

    def test_lpru_zero_alpha_strictly_restores_base(self):
        module = LearnablePixelReassemblyUpsample(channels=8).eval()
        source = torch.randn(2, 8, 10, 12)
        base = F.interpolate(source, scale_factor=2.0, mode='nearest')
        with torch.no_grad():
            module.raw_alpha.zero_()
            output = module(source, base)
        self.assertTrue(torch.equal(output, base))

    def test_spdr_pixel_unshuffle_is_lossless_and_shapes_match(self):
        module = SubpixelPreservingDownsample(channels=8).eval()
        source = torch.randn(2, 8, 30, 38)
        base_conv = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        base = base_conv(source)

        with torch.inference_mode():
            output, aux = module(source, base, return_aux=True)

        self.assertEqual(module.pixel_unshuffle.downscale_factor, 2)
        self.assertEqual(tuple(aux['rearranged'].shape), (2, 32, 15, 19))
        self.assertTrue(torch.equal(
            F.pixel_shuffle(aux['rearranged'], upscale_factor=2), source))
        self.assertEqual(tuple(aux['preserved'].shape), tuple(base.shape))
        self.assertEqual(tuple(output.shape), tuple(base.shape))
        self.assertEqual(module.compress.in_channels, 4 * 8)
        self.assertEqual(module.compress.out_channels, 8)
        self.assertEqual(module.depthwise.groups, 8)

    def test_spdr_zero_beta_strictly_restores_base(self):
        module = SubpixelPreservingDownsample(channels=8).eval()
        source = torch.randn(2, 8, 20, 24)
        base = torch.randn(2, 8, 10, 12)
        with torch.no_grad():
            module.raw_beta.zero_()
            output = module(source, base)
        self.assertTrue(torch.equal(output, base))

    def test_channelwise_initialization_and_signed_bounds(self):
        lpru = LearnablePixelReassemblyUpsample(
            channels=7, alpha_max=0.5, alpha_init=0.05)
        spdr = SubpixelPreservingDownsample(
            channels=7, beta_max=0.5, beta_init=0.05)
        expected_raw = math.atanh(0.05 / 0.5)

        for raw, effective in (
                (lpru.raw_alpha, lpru.effective_alpha()),
                (spdr.raw_beta, spdr.effective_beta())):
            self.assertEqual(tuple(raw.shape), (1, 7, 1, 1))
            self.assertTrue(torch.allclose(
                raw, torch.full_like(raw, expected_raw),
                atol=1e-7, rtol=1e-7))
            self.assertTrue(torch.allclose(
                effective, torch.full_like(effective, 0.05),
                atol=1e-7, rtol=1e-7))

        with torch.no_grad():
            lpru.raw_alpha.fill_(-100.0)
            spdr.raw_beta.fill_(100.0)
        self.assertTrue((lpru.effective_alpha() < 0).all())
        self.assertTrue((lpru.effective_alpha().abs() <= 0.5).all())
        self.assertTrue((spdr.effective_beta() > 0).all())
        self.assertTrue((spdr.effective_beta().abs() <= 0.5).all())

    def test_forward_backward_is_finite_and_reaches_every_parameter(self):
        cases = []

        lpru = LearnablePixelReassemblyUpsample(channels=8).train()
        lpru_source = torch.randn(2, 8, 10, 12, requires_grad=True)
        lpru_base = F.interpolate(
            lpru_source, scale_factor=2.0, mode='nearest')
        cases.append(('LPRU', lpru, lpru_source,
                      lpru(lpru_source, lpru_base)))

        spdr = SubpixelPreservingDownsample(channels=8).train()
        spdr_source = torch.randn(2, 8, 20, 24, requires_grad=True)
        spdr_base = F.avg_pool2d(spdr_source, kernel_size=2, stride=2)
        cases.append(('SPDR', spdr, spdr_source,
                      spdr(spdr_source, spdr_base)))

        for name, module, source, output in cases:
            with self.subTest(module=name):
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(source.grad)
                self.assertTrue(torch.isfinite(source.grad).all())
                self.assertGreater(source.grad.abs().sum().item(), 0.0)
                problems = []
                for parameter_name, parameter in module.named_parameters():
                    if (parameter.grad is None
                            or not torch.isfinite(parameter.grad).all()
                            or parameter.grad.abs().sum().item() == 0.0):
                        problems.append(parameter_name)
                self.assertEqual(problems, [])

    def test_cpu_autocast_preserves_feature_dtype_and_finite_gradients(self):
        # In the real encoder, the preceding autocast convolution already
        # supplies a reduced-precision feature to the resampling block.
        source = torch.randn(2, 8, 16, 20).to(
            dtype=torch.bfloat16).requires_grad_()
        lpru = LearnablePixelReassemblyUpsample(channels=8).train()
        spdr = SubpixelPreservingDownsample(channels=8).train()

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            up = lpru(source)
            down = spdr(up, F.avg_pool2d(up, kernel_size=2, stride=2))
            loss = down.square().mean()

        self.assertEqual(up.dtype, torch.bfloat16)
        self.assertEqual(down.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(up).all())
        self.assertTrue(torch.isfinite(down).all())
        loss.backward()
        for module in (lpru, spdr):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_no_normalization_attention_or_gate_and_instances_are_independent(self):
        modules = (
            LearnablePixelReassemblyUpsample(channels=8),
            SubpixelPreservingDownsample(channels=8),
        )
        for module in modules:
            self.assertFalse(any(
                isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d,
                                   nn.BatchNorm3d, nn.GroupNorm,
                                   nn.LayerNorm, nn.MultiheadAttention))
                for child in module.modules()))
            names = ' '.join(name.lower() for name, _ in module.named_modules())
            self.assertNotIn('attention', names)
            self.assertNotIn('gate', names)

        first = LearnablePixelReassemblyUpsample(channels=8)
        second = LearnablePixelReassemblyUpsample(channels=8)
        self.assertNotEqual(id(first.expand.weight), id(second.expand.weight))
        first_down = SubpixelPreservingDownsample(channels=8)
        second_down = SubpixelPreservingDownsample(channels=8)
        self.assertNotEqual(
            id(first_down.compress.weight), id(second_down.compress.weight))

    def test_debug_stats_are_detached_finite_and_silent(self):
        source = torch.randn(1, 4, 8, 10)
        lpru = LearnablePixelReassemblyUpsample(
            channels=4, debug=True).eval()
        spdr = SubpixelPreservingDownsample(
            channels=4, debug=True).eval()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), torch.inference_mode():
            up = lpru(source)
            spdr(up, source)
        self.assertEqual(captured.getvalue(), '')
        for module in (lpru, spdr):
            self.assertTrue(module.last_debug_stats)
            self.assertTrue(all(
                torch.is_tensor(value)
                and not value.requires_grad
                and torch.isfinite(value).all()
                for value in module.last_debug_stats.values()))

    def test_contract_and_configuration_validation(self):
        for kwargs in (
                {'channels': 0},
                {'alpha_max': 0.0},
                {'alpha_max': 0.5, 'alpha_init': 0.5},
                {'alpha_max': 0.5, 'alpha_init': -0.5},
                {'debug': 1}):
            with self.subTest(kind='LPRU', kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LearnablePixelReassemblyUpsample(**kwargs)
        for kwargs in (
                {'channels': 0},
                {'beta_max': 0.0},
                {'beta_max': 0.5, 'beta_init': 0.5},
                {'beta_max': 0.5, 'beta_init': -0.5},
                {'debug': 1}):
            with self.subTest(kind='SPDR', kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SubpixelPreservingDownsample(**kwargs)

        lpru = LearnablePixelReassemblyUpsample(channels=4)
        with self.assertRaises(RuntimeError):
            lpru(torch.randn(1, 3, 8, 8))
        with self.assertRaises(RuntimeError):
            lpru(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 15, 16))

        spdr = SubpixelPreservingDownsample(channels=4)
        with self.assertRaises(RuntimeError):
            spdr(torch.randn(1, 4, 15, 16), torch.randn(1, 4, 8, 8))
        with self.assertRaises(RuntimeError):
            spdr(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 7, 8))


if __name__ == '__main__':
    unittest.main()
