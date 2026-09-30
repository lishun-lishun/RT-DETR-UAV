"""Focused unit tests for the Direct Global Fusion Residual neck."""

import contextlib
import io
import math
import unittest

import torch
import torch.nn as nn

from tests._support import prepare_imports

prepare_imports()

from src.zoo.rtdetr.dgfr_neck import (  # noqa: E402
    DGFRNeck,
    DirectGlobalFusion,
    DirectScaleAdapter,
)
from src.zoo.rtdetr.hybrid_encoder import (  # noqa: E402
    CSPRepLayer,
    ConvNormLayer,
)


def build_neck(hidden_dim=16, fusion_channels=4, **kwargs):
    return DGFRNeck(
        hidden_dim=hidden_dim,
        fusion_channels=fusion_channels,
        conv_norm_factory=ConvNormLayer,
        fusion_factory=CSPRepLayer,
        **kwargs)


def make_features(base_size, hidden_dim=16, requires_grad=False):
    sizes = (base_size // 8, base_size // 16, base_size // 32)
    projected = [
        torch.randn(2, hidden_dim, size, size, requires_grad=requires_grad)
        for size in sizes
    ]
    original = [
        torch.randn(2, hidden_dim, size, size, requires_grad=requires_grad)
        for size in sizes
    ]
    return projected, original


class DGFRNeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_factories_and_fully_independent_routes(self):
        module = build_neck()
        fusions = (module.fusion3, module.fusion4, module.fusion5)
        self.assertTrue(all(isinstance(value, DirectGlobalFusion)
                            for value in fusions))
        self.assertEqual(len({id(value.fusion) for value in fusions}), 3)
        self.assertTrue(all(isinstance(value.fusion, CSPRepLayer)
                            for value in fusions))

        adapters = [fusion.adapters[key]
                    for fusion in fusions for key in ('x3', 'x4', 'x5')]
        self.assertEqual(len({id(value) for value in adapters}), 9)
        self.assertTrue(all(isinstance(value, DirectScaleAdapter)
                            for value in adapters))

        # X3 -> target 5 is exactly two consecutive native 3x3 stride-2
        # ConvNormLayer blocks: hidden_dim -> fusion_channels -> fusion_channels.
        x3_to_5 = module.fusion5.adapters['x3']
        self.assertIsInstance(x3_to_5.layers, nn.Sequential)
        self.assertEqual(len(x3_to_5.layers), 2)
        first, second = x3_to_5.layers
        self.assertTrue(all(isinstance(value, ConvNormLayer)
                            for value in (first, second)))
        self.assertEqual((first.conv.kernel_size, first.conv.stride,
                          first.conv.padding), ((3, 3), (2, 2), (1, 1)))
        self.assertEqual((second.conv.kernel_size, second.conv.stride,
                          second.conv.padding), ((3, 3), (2, 2), (1, 1)))
        self.assertEqual((first.conv.in_channels, first.conv.out_channels),
                         (16, 4))
        self.assertEqual((second.conv.in_channels, second.conv.out_channels),
                         (4, 4))

    def test_each_fusion_uses_192_channels_at_production_width(self):
        module = DGFRNeck(
            hidden_dim=256,
            fusion_channels=64,
            conv_norm_factory=ConvNormLayer,
            fusion_factory=CSPRepLayer)
        for fusion in (module.fusion3, module.fusion4, module.fusion5):
            self.assertEqual(fusion.fusion.conv1.conv.in_channels, 192)
            self.assertEqual(fusion.fusion.conv2.conv.in_channels, 192)
            self.assertEqual(fusion.fusion.conv3.conv.out_channels, 256)

    def test_gamma_initialization_is_channelwise_and_correct(self):
        module = build_neck(gamma_max=0.25, gamma_init=0.05)
        expected_raw = math.atanh(0.05 / 0.25)
        for level, gamma in zip((3, 4, 5), module.effective_gamma()):
            raw = getattr(module, f'raw_gamma{level}')
            self.assertEqual(tuple(raw.shape), (1, 16, 1, 1))
            self.assertTrue(torch.allclose(
                raw, torch.full_like(raw, expected_raw),
                atol=1e-7, rtol=1e-7))
            self.assertTrue(torch.allclose(
                gamma, torch.full_like(gamma, 0.05),
                atol=1e-7, rtol=1e-7))

        with torch.no_grad():
            module.raw_gamma3.fill_(-1.0)
        self.assertTrue((module.effective_gamma(3) < 0).all())
        self.assertTrue((module.effective_gamma(3).abs() < 0.25).all())

    def test_dynamic_480_640_800_shapes(self):
        module = build_neck().eval()
        with torch.inference_mode():
            for size in (480, 640, 800):
                projected, original = make_features(size)
                outputs, aux = module(
                    projected, original, return_aux=True)
                expected_sizes = (
                    (size // 8, size // 8),
                    (size // 16, size // 16),
                    (size // 32, size // 32),
                )
                self.assertEqual(
                    [tuple(value.shape) for value in outputs],
                    [(2, 16, *spatial) for spatial in expected_sizes])
                self.assertEqual(
                    [tuple(value.shape) for value in aux['evidences']],
                    [(2, 16, *spatial) for spatial in expected_sizes])

    def test_nearest_upsample_uses_exact_target_size(self):
        adapter = DirectScaleAdapter(
            source_level=5,
            target_level=3,
            hidden_dim=16,
            fusion_channels=4,
            conv_norm_factory=ConvNormLayer).eval()
        source = torch.randn(1, 16, 3, 5)
        with torch.inference_mode():
            output = adapter(source, (13, 21))
        self.assertEqual(tuple(output.shape), (1, 4, 13, 21))

    def test_backward_reaches_every_adapter_fusion_and_gamma(self):
        module = build_neck().train()
        projected, original = make_features(
            160, requires_grad=True)
        outputs = module(projected, original)
        loss = sum(value.square().mean() for value in outputs)
        loss.backward()

        problems = []
        for name, parameter in module.named_parameters():
            if (parameter.grad is None
                    or not torch.isfinite(parameter.grad).all()
                    or parameter.grad.abs().sum().item() == 0.0):
                problems.append(name)
        self.assertEqual(problems, [])
        for feature in projected:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0.0)

    def test_debug_stats_are_detached_and_never_printed(self):
        module = build_neck(debug=True, debug_interval=2).eval()
        projected, original = make_features(160)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), torch.inference_mode():
            module(projected, original)
        self.assertEqual(captured.getvalue(), '')
        expected = {
            *(f'gamma{level}_{stat}' for level in (3, 4, 5)
              for stat in ('mean', 'min', 'max')),
            *(f'e{level}_to_o{level}_norm_ratio' for level in (3, 4, 5)),
            *(f'y{level}_norm' for level in (3, 4, 5)),
        }
        self.assertEqual(set(module.last_debug_stats), expected)
        self.assertTrue(all(not value.requires_grad
                            and torch.isfinite(value).all()
                            for value in module.last_debug_stats.values()))

    def test_configuration_and_contract_validation(self):
        invalid_kwargs = (
            {'hidden_dim': 0},
            {'fusion_channels': 0},
            {'gamma_max': 0.0},
            {'gamma_max': 0.25, 'gamma_init': 0.25},
            {'gamma_max': 0.25, 'gamma_init': -0.25},
            {'debug': 1},
            {'debug_interval': 0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_neck(**kwargs)

        module = build_neck().eval()
        projected, original = make_features(160)
        with self.assertRaises(RuntimeError):
            module(projected[:2], original)
        with self.assertRaises(RuntimeError):
            module(projected, original[:2])
        bad = list(original)
        bad[1] = torch.randn(2, 16, 11, 10)
        with self.assertRaises(RuntimeError):
            module(projected, bad)


if __name__ == '__main__':
    unittest.main()
