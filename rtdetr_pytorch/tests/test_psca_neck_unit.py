"""Focused unit tests for Partial Spatial Context Attention."""

import math
import unittest

import torch
import torch.nn as nn

from tests._support import prepare_imports

prepare_imports()

from src.zoo.rtdetr.psca_neck import (  # noqa: E402
    PSCANeck,
    PartialSpatialContextAttention,
)


class PSCANeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2718)

    def test_contiguous_75_25_split_and_projection_contract(self):
        module = PartialSpatialContextAttention(
            hidden_dim=8, attention_dim=3, pool_stride=2,
            channel_shuffle=False).eval()
        features = torch.arange(
            1 * 8 * 6 * 10, dtype=torch.float32).reshape(1, 8, 6, 10)
        with torch.inference_mode():
            output, aux = module(features, return_aux=True)

        self.assertEqual(module.partial_channels, 6)
        self.assertEqual(module.context_channels, 2)
        self.assertTrue(torch.equal(aux['partial'], features[:, :6]))
        self.assertTrue(torch.equal(aux['context'], features[:, 6:]))
        self.assertEqual(module.query.in_channels, 2)
        self.assertEqual(module.query.out_channels, 3)
        self.assertEqual(module.key.in_channels, 2)
        self.assertEqual(module.key.out_channels, 3)
        self.assertEqual(module.value.in_channels, 2)
        self.assertEqual(module.value.out_channels, 2)
        self.assertEqual(module.output.in_channels, 2)
        self.assertEqual(module.output.out_channels, 2)
        self.assertEqual(tuple(output.shape), tuple(features.shape))
        self.assertTrue(torch.equal(output[:, :6], features[:, :6]))

    def test_dynamic_kv_compression_and_attention_rows(self):
        module = PSCANeck(
            hidden_dim=8, attention_dim=4, pool_stride=4,
            channel_shuffle=False).eval()
        features = torch.randn(2, 8, 20, 28)
        with torch.inference_mode():
            _, aux = module(features, return_aux=True)

        self.assertEqual(tuple(aux['pooled_context'].shape), (2, 2, 5, 7))
        self.assertEqual(tuple(aux['query'].shape), (2, 560, 4))
        self.assertEqual(tuple(aux['key'].shape), (2, 4, 35))
        self.assertEqual(tuple(aux['value'].shape), (2, 35, 2))
        self.assertEqual(tuple(aux['attention'].shape), (2, 560, 35))
        row_sums = aux['attention'].sum(dim=-1)
        self.assertTrue(torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6))
        self.assertEqual(aux['attention'].dtype, torch.float32)

        # The pooled grid changes with the input rather than being fixed.
        with torch.inference_mode():
            _, dynamic = module(
                torch.randn(1, 8, 24, 36), return_aux=True)
        self.assertEqual(tuple(dynamic['pooled_context'].shape[-2:]), (6, 9))
        self.assertEqual(dynamic['attention'].shape[-1], 54)

    def test_inverse_tanh_alpha_initialization_bounds_and_zero(self):
        module = PartialSpatialContextAttention(
            hidden_dim=8, alpha_max=0.20, alpha_init=0.02,
            channel_shuffle=False).eval()
        expected_raw = math.atanh(0.02 / 0.20)
        self.assertEqual(tuple(module.raw_alpha.shape), (1, 2, 1, 1))
        self.assertTrue(torch.allclose(
            module.raw_alpha,
            torch.full_like(module.raw_alpha, expected_raw),
            atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(
            module.effective_alpha(),
            torch.full_like(module.raw_alpha, 0.02),
            atol=1e-7, rtol=1e-7))

        features = torch.randn(2, 8, 12, 16)
        with torch.no_grad():
            module.raw_alpha.zero_()
            output = module(features)
        self.assertTrue(torch.equal(output, features))

        with torch.no_grad():
            module.raw_alpha.fill_(100.0)
        self.assertTrue((module.effective_alpha().abs() <= 0.20 + 1e-7).all())
        with torch.no_grad():
            module.raw_alpha.fill_(-100.0)
        self.assertTrue((module.effective_alpha() < 0.0).all())
        self.assertTrue((module.effective_alpha().abs() <= 0.20 + 1e-7).all())

    def test_fixed_channel_shuffle_preserves_value_multiset(self):
        plain = PartialSpatialContextAttention(
            hidden_dim=8, pool_stride=2, channel_shuffle=False).eval()
        shuffled = PartialSpatialContextAttention(
            hidden_dim=8, pool_stride=2, channel_shuffle=True,
            shuffle_groups=2).eval()
        shuffled.load_state_dict(plain.state_dict())
        features = torch.arange(
            8 * 4 * 4, dtype=torch.float32).reshape(1, 8, 4, 4)
        with torch.no_grad():
            plain.raw_alpha.zero_()
            shuffled.raw_alpha.zero_()
            expected = plain(features)
            actual = shuffled(features)
        self.assertTrue(torch.equal(expected, features))
        self.assertFalse(torch.equal(actual, features))
        self.assertTrue(torch.equal(
            actual.flatten().sort().values,
            features.flatten().sort().values))
        self.assertEqual(sum(p.numel() for p in shuffled.parameters()),
                         sum(p.numel() for p in plain.parameters()))

    def test_n3_n4_feature_sizes_for_480_640_800(self):
        n3_module = PSCANeck(
            hidden_dim=8, attention_dim=4, pool_stride=4).eval()
        n4_module = PSCANeck(
            hidden_dim=8, attention_dim=4, pool_stride=2).eval()
        with torch.inference_mode():
            for image_size in (480, 640, 800):
                n3 = torch.randn(1, 8, image_size // 8, image_size // 8)
                n4 = torch.randn(1, 8, image_size // 16, image_size // 16)
                out3, aux3 = n3_module(n3, return_aux=True)
                out4, aux4 = n4_module(n4, return_aux=True)
                with self.subTest(image_size=image_size):
                    self.assertEqual(tuple(out3.shape), tuple(n3.shape))
                    self.assertEqual(tuple(out4.shape), tuple(n4.shape))
                    expected_kv = image_size // 32
                    self.assertEqual(
                        tuple(aux3['pooled_context'].shape[-2:]),
                        (expected_kv, expected_kv))
                    self.assertEqual(
                        tuple(aux4['pooled_context'].shape[-2:]),
                        (expected_kv, expected_kv))

    def test_backward_reaches_every_parameter(self):
        module = PartialSpatialContextAttention(
            hidden_dim=8, attention_dim=4, pool_stride=2).train()
        features = torch.randn(2, 8, 12, 16, requires_grad=True)
        output = module(features)
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

    def test_cpu_autocast_is_finite_and_local_fp32_attention(self):
        if not hasattr(torch, 'autocast'):
            self.skipTest('torch.autocast is unavailable')
        module = PartialSpatialContextAttention(
            hidden_dim=8, attention_dim=4, pool_stride=2).eval()
        features = torch.randn(1, 8, 12, 16)
        try:
            with torch.inference_mode(), torch.autocast(
                    device_type='cpu', dtype=torch.bfloat16):
                output, aux = module(features, return_aux=True)
        except RuntimeError as error:
            self.skipTest(f'CPU bfloat16 autocast unavailable: {error}')
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(aux['attention']).all())
        self.assertEqual(aux['attention'].dtype, torch.float32)
        self.assertEqual(aux['attended'].dtype, torch.bfloat16)
        self.assertIn(output.dtype, (torch.float32, torch.bfloat16))

    def test_no_forbidden_layers_or_extra_gates(self):
        module = PartialSpatialContextAttention(hidden_dim=8)
        forbidden = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.LayerNorm, nn.MultiheadAttention,
        )
        self.assertFalse(any(
            isinstance(child, forbidden) for child in module.modules()))
        module_names = ' '.join(
            type(child).__name__.lower() for child in module.modules())
        for name in ('batchnorm', 'groupnorm', 'layernorm', 'seblock',
                     'squeezeexcitation', 'cbam', 'gate'):
            self.assertNotIn(name, module_names)
        parameter_names = set(dict(module.named_parameters()))
        self.assertEqual(
            {name for name in parameter_names if 'alpha' in name},
            {'raw_alpha'})

    def test_validation(self):
        invalid_kwargs = (
            {'hidden_dim': 0},
            {'hidden_dim': 2, 'context_ratio': 0.1},
            {'context_ratio': 0.0},
            {'context_ratio': 1.0},
            {'context_ratio': float('nan')},
            {'attention_dim': 0},
            {'pool_stride': 0},
            {'alpha_max': 0.0},
            {'alpha_max': 0.2, 'alpha_init': 0.2},
            {'channel_shuffle': 1},
            {'hidden_dim': 7, 'channel_shuffle': True,
             'shuffle_groups': 2},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PartialSpatialContextAttention(**kwargs)

        module = PartialSpatialContextAttention(
            hidden_dim=8, pool_stride=4)
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 7, 8, 8))
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 8, 8))
        with self.assertRaises(RuntimeError):
            module(torch.ones(1, 8, 8, 8, dtype=torch.int64))
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 8, 3, 8))


if __name__ == '__main__':
    unittest.main()
