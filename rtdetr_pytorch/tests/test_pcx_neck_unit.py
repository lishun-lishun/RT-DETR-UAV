"""Focused unit tests for partial-channel cross-scale exchange."""

import inspect
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'src' / 'zoo' / 'rtdetr' / 'pcx_neck.py')
SPEC = importlib.util.spec_from_file_location('pcx_neck_unit', MODULE_PATH)
PCX_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PCX_MODULE
SPEC.loader.exec_module(PCX_MODULE)
PCXNeck = PCX_MODULE.PCXNeck
PartialChannelCrossScaleExchange = (
    PCX_MODULE.PartialChannelCrossScaleExchange)


def _pyramid(batch=2, channels=8, sizes=((15, 19), (8, 10), (4, 5))):
    return [
        torch.randn(batch, channels, height, width)
        for height, width in sizes
    ]


class PCXNeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_contiguous_channel_split_and_preservation_route(self):
        module = PartialChannelCrossScaleExchange(
            hidden_dim=8, exchange_ratio=0.25,
            channel_shuffle=False).eval()
        features = _pyramid()

        with torch.inference_mode():
            outputs, aux = module(features, return_aux=True)

        self.assertEqual(module.preserved_channels, 6)
        self.assertEqual(module.exchange_channels, 2)
        for source, output, preserved, exchanged in zip(
                features, outputs, aux['preserved'], aux['exchange']):
            self.assertTrue(torch.equal(preserved, source[:, :6]))
            self.assertTrue(torch.equal(exchanged, source[:, 6:]))
            self.assertTrue(torch.equal(output[:, :6], source[:, :6]))
            self.assertEqual(tuple(output.shape), tuple(source.shape))

    def test_exchange_formula_uses_explicit_target_sizes(self):
        module = PCXNeck(
            hidden_dim=8, exchange_ratio=0.25,
            channel_shuffle=False).eval()
        features = _pyramid(sizes=((17, 23), (9, 12), (5, 6)))
        with torch.inference_mode():
            _, aux = module(features, return_aux=True)

            e3, e4, e5 = aux['exchange']
            expected3 = e3 + module._resize(e4, e3.shape[-2:])
            expected4 = (
                e4
                + module._resize(module.down_3_to_4(e3), e4.shape[-2:])
                + module._resize(e5, e4.shape[-2:]))
            expected5 = (
                e5
                + module._resize(module.down_4_to_5(e4), e5.shape[-2:]))

        for actual, expected in zip(
                aux['exchange_sums'], (expected3, expected4, expected5)):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(module.down_3_to_4.kernel_size, (3, 3))
        self.assertEqual(module.down_3_to_4.stride, (2, 2))
        self.assertEqual(module.down_4_to_5.kernel_size, (3, 3))
        self.assertEqual(module.down_4_to_5.stride, (2, 2))

        for refinement in (module.refine3, module.refine4, module.refine5):
            self.assertEqual(refinement.depthwise.kernel_size, (3, 3))
            self.assertEqual(refinement.depthwise.groups, 2)
            self.assertEqual(refinement.pointwise.kernel_size, (1, 1))

    def test_480_640_800_pyramid_shapes(self):
        module = PCXNeck(hidden_dim=8).eval()
        for image_size in (480, 640, 800):
            sizes = tuple(
                (image_size // stride, image_size // stride)
                for stride in (8, 16, 32))
            with self.subTest(image_size=image_size), torch.inference_mode():
                features = _pyramid(batch=1, sizes=sizes)
                outputs = module(features)
            self.assertEqual(
                [tuple(output.shape) for output in outputs],
                [tuple(feature.shape) for feature in features])

    def test_shuffle_is_exact_parameter_free_permutation(self):
        feature = torch.arange(2 * 8 * 3 * 5, dtype=torch.float32).reshape(
            2, 8, 3, 5)
        shuffled = PCXNeck._shuffle_two_groups(feature)
        expected = feature.reshape(2, 2, 4, 3, 5).transpose(
            1, 2).contiguous().reshape_as(feature)

        self.assertTrue(torch.equal(shuffled, expected))
        for batch in range(feature.shape[0]):
            self.assertTrue(torch.equal(
                torch.sort(shuffled[batch].flatten()).values,
                torch.sort(feature[batch].flatten()).values))

    def test_forward_backward_reaches_every_parameter(self):
        module = PartialChannelCrossScaleExchange(
            hidden_dim=8, exchange_ratio=0.25).train()
        features = [
            feature.requires_grad_() for feature in _pyramid(
                sizes=((15, 19), (8, 10), (4, 5)))
        ]
        outputs = module(features)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()

        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0.0)

        problems = []
        for name, parameter in module.named_parameters():
            if (parameter.grad is None
                    or not torch.isfinite(parameter.grad).all()
                    or parameter.grad.abs().sum().item() == 0.0):
                problems.append(name)
        self.assertEqual(problems, [])

    def test_all_level_parameters_are_independent(self):
        module = PCXNeck(hidden_dim=8)
        self.assertIsNot(module.down_3_to_4, module.down_4_to_5)
        self.assertIsNot(module.refine3, module.refine4)
        self.assertIsNot(module.refine4, module.refine5)
        pointers = [parameter.data_ptr() for parameter in module.parameters()]
        self.assertEqual(len(pointers), len(set(pointers)))

    def test_no_disallowed_operators_or_identifiers(self):
        module = PCXNeck(hidden_dim=8)
        forbidden_types = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.LayerNorm, nn.MultiheadAttention,
            nn.Sigmoid, nn.Softmax,
        )
        self.assertFalse(any(
            isinstance(child, forbidden_types) for child in module.modules()))
        source = inspect.getsource(PartialChannelCrossScaleExchange).lower()
        module_names = ' '.join(
            name.lower() for name, _ in module.named_modules())
        for token in ('sigmoid', 'softmax', 'cosine', 'attention', 'gate'):
            self.assertNotIn(token, source)
            self.assertNotIn(token, module_names)

    def test_configuration_and_input_validation(self):
        for kwargs in (
                {'hidden_dim': 0},
                {'hidden_dim': 3},
                {'exchange_ratio': 0.0},
                {'exchange_ratio': 1.0},
                {'exchange_ratio': float('nan')},
                {'hidden_dim': 4, 'exchange_ratio': 0.1},
                {'channel_shuffle': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PCXNeck(**kwargs)

        module = PCXNeck(hidden_dim=8)
        valid = _pyramid()
        invalid_inputs = (
            valid[:2],
            [valid[0], valid[1], torch.randn(2, 7, 4, 5)],
            [valid[0], valid[1], torch.randn(1, 8, 4, 5)],
            [valid[0], valid[1], torch.ones(2, 8, 4, 5, dtype=torch.int64)],
            [valid[1], valid[0], valid[2]],
        )
        for features in invalid_inputs:
            with self.subTest(shapes=[tuple(x.shape) for x in features]):
                with self.assertRaises(RuntimeError):
                    module(features)


if __name__ == '__main__':
    unittest.main()
