"""Focused unit tests for the RDCF neck and deploy conversion."""

import copy
import math
import unittest

import torch
import torch.nn as nn

from tests._support import prepare_imports

prepare_imports()

from src.zoo.rtdetr.rdcf_neck import (  # noqa: E402
    RDCFNeck,
    ReparamDirectionalContextBlock,
)


def make_features(image_size, channels=8, requires_grad=False):
    return [
        torch.randn(
            1, channels, image_size // stride, image_size // stride,
            requires_grad=requires_grad)
        for stride in (8, 16, 32)
    ]


class RDCFNeckUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(413)

    def test_training_structure_and_channelwise_eta(self):
        module = ReparamDirectionalContextBlock(
            channels=7, eta_max=0.30, eta_init=0.05)
        self.assertEqual(module.dw_3x3.kernel_size, (3, 3))
        self.assertEqual(module.dw_3x3.padding, (1, 1))
        self.assertEqual(module.dw_1x9.kernel_size, (1, 9))
        self.assertEqual(module.dw_1x9.padding, (0, 4))
        self.assertEqual(module.dw_9x1.kernel_size, (9, 1))
        self.assertEqual(module.dw_9x1.padding, (4, 0))
        self.assertTrue(all(
            branch.groups == 7
            for branch in (module.dw_3x3, module.dw_1x9, module.dw_9x1)))
        self.assertEqual(module.project.kernel_size, (1, 1))

        expected_raw = math.atanh(0.05 / 0.30)
        self.assertEqual(tuple(module.raw_eta.shape), (1, 7, 1, 1))
        self.assertTrue(torch.allclose(
            module.raw_eta, torch.full_like(module.raw_eta, expected_raw),
            atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(
            module.effective_eta(),
            torch.full_like(module.raw_eta, 0.05),
            atol=1e-7, rtol=1e-7))
        with torch.no_grad():
            module.raw_eta.fill_(-100.0)
        self.assertTrue((module.effective_eta() < 0).all())
        self.assertTrue((module.effective_eta().abs() <= 0.30).all())

    def test_no_normalization_attention_or_gate(self):
        module = RDCFNeck(hidden_dim=8)
        forbidden = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                     nn.GroupNorm, nn.LayerNorm, nn.MultiheadAttention)
        self.assertFalse(any(
            isinstance(child, forbidden) for child in module.modules()))
        names = ' '.join(name.lower() for name, _ in module.named_modules())
        self.assertNotIn('attention', names)
        self.assertNotIn('gate', names)
        self.assertNotEqual(
            id(module.rdcf3.dw_3x3.weight), id(module.rdcf4.dw_3x3.weight))

    def test_dynamic_480_640_800_and_n5_identity(self):
        module = RDCFNeck(hidden_dim=8).eval()
        with torch.inference_mode():
            for image_size in (480, 640, 800):
                features = make_features(image_size)
                outputs = module(features)
                self.assertEqual(
                    [tuple(value.shape) for value in outputs],
                    [tuple(value.shape) for value in features])
                self.assertIs(outputs[2], features[2])
                self.assertTrue(torch.equal(outputs[2], features[2]))

    def test_backward_reaches_all_parameters_and_inputs(self):
        module = RDCFNeck(hidden_dim=8).train()
        features = make_features(160, requires_grad=True)
        outputs = module(features)
        sum(output.square().mean() for output in outputs).backward()

        problems = []
        for name, parameter in module.named_parameters():
            if (parameter.grad is None
                    or not torch.isfinite(parameter.grad).all()
                    or parameter.grad.abs().sum().item() == 0.0):
                problems.append(name)
        self.assertEqual(problems, [])
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0.0)

    def test_branch_fusion_kernel_layout_and_bias(self):
        block = ReparamDirectionalContextBlock(channels=1).eval()
        with torch.no_grad():
            block.dw_3x3.weight.fill_(1.0)
            block.dw_1x9.weight.fill_(2.0)
            block.dw_9x1.weight.fill_(3.0)
            block.dw_3x3.bias.fill_(5.0)
            block.dw_1x9.bias.fill_(7.0)
            block.dw_9x1.bias.fill_(11.0)
        kernel, bias = block.get_equivalent_kernel_bias()
        expected = torch.zeros_like(kernel)
        expected[:, :, 3:6, 3:6] += 1.0
        expected[:, :, 4:5, :] += 2.0
        expected[:, :, :, 4:5] += 3.0
        self.assertTrue(torch.equal(kernel, expected))
        self.assertTrue(torch.equal(bias, torch.tensor([23.0])))

    def test_train_and_deploy_are_numerically_equivalent(self):
        module = RDCFNeck(hidden_dim=8).eval()
        features = make_features(160)
        with torch.inference_mode():
            training_outputs = module(features)

        module.switch_to_deploy()
        self.assertTrue(module.deploy)
        for block in (module.rdcf3, module.rdcf4):
            self.assertTrue(block.deploy)
            self.assertIsInstance(block.reparam_conv, nn.Conv2d)
            self.assertEqual(block.reparam_conv.kernel_size, (9, 9))
            self.assertEqual(block.reparam_conv.padding, (4, 4))
            self.assertEqual(block.reparam_conv.groups, 8)
            self.assertIsNotNone(block.reparam_conv.bias)
            self.assertFalse(hasattr(block, 'dw_3x3'))
            self.assertFalse(hasattr(block, 'dw_1x9'))
            self.assertFalse(hasattr(block, 'dw_9x1'))

        with torch.inference_mode():
            deployed_outputs = module(features)
        for training, deployed in zip(training_outputs, deployed_outputs):
            self.assertTrue(torch.allclose(
                training, deployed, atol=1e-5, rtol=1e-4))

        # Repeated conversion must be harmless.
        state_before = copy.deepcopy(module.state_dict())
        self.assertIs(module.switch_to_deploy(), module)
        for key, value in state_before.items():
            self.assertTrue(torch.equal(value, module.state_dict()[key]))

    def test_converted_state_loads_into_direct_deploy_constructor(self):
        training = RDCFNeck(hidden_dim=8).eval()
        features = make_features(160)
        with torch.inference_mode():
            expected = training(features)

        converted = copy.deepcopy(training).switch_to_deploy()
        deploy_state = copy.deepcopy(converted.state_dict())
        rebuilt = RDCFNeck(hidden_dim=8, deploy=True).eval()
        incompatible = rebuilt.load_state_dict(deploy_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertFalse(any(
            'dw_3x3' in key or 'dw_1x9' in key or 'dw_9x1' in key
            for key in deploy_state))
        self.assertTrue(any('reparam_conv.weight' in key
                            for key in deploy_state))

        with torch.inference_mode():
            actual = rebuilt(features)
        for reference, output in zip(expected, actual):
            self.assertTrue(torch.allclose(
                reference, output, atol=1e-5, rtol=1e-4))

    def test_direct_deploy_construction_and_validation(self):
        block = ReparamDirectionalContextBlock(
            channels=4, deploy=True).eval()
        self.assertTrue(block.deploy)
        self.assertTrue(hasattr(block, 'reparam_conv'))
        self.assertFalse(hasattr(block, 'dw_3x3'))
        with torch.inference_mode():
            output = block(torch.randn(2, 4, 13, 17))
        self.assertEqual(tuple(output.shape), (2, 4, 13, 17))

        invalid_kwargs = (
            {'hidden_dim': 0},
            {'eta_max': 0.0},
            {'eta_max': 0.3, 'eta_init': 0.3},
            {'eta_max': 0.3, 'eta_init': -0.3},
            {'deploy': 1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RDCFNeck(**kwargs)

        neck = RDCFNeck(hidden_dim=4)
        with self.assertRaises(RuntimeError):
            neck([torch.randn(1, 4, 8, 8)] * 2)
        with self.assertRaises(RuntimeError):
            neck([
                torch.randn(1, 4, 8, 8),
                torch.randn(1, 3, 4, 4),
                torch.randn(1, 4, 2, 2),
            ])


if __name__ == '__main__':
    unittest.main()
