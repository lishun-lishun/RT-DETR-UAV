"""Tests for the independent LPRU and SPDR HybridEncoder experiments."""

from pathlib import Path
import runpy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests._support import PROJECT_DIR, prepare_imports


prepare_imports()

from src.core import YAMLConfig  # noqa: E402
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder  # noqa: E402
from src.zoo.rtdetr.resample_neck import (  # noqa: E402
    LearnablePixelReassemblyUpsample,
    SubpixelPreservingDownsample,
)


ROOT = PROJECT_DIR
CONFIG_DIR = ROOT / 'configs/rtdetr'
PRES_BASE = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav.yml'
HR_BASE = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav.yml'
PRES_LPRU = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_lpru.yml'
HR_LPRU = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_lpru.yml'
PRES_SPDR = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_spdr.yml'
HR_SPDR = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_spdr.yml'
CANDIDATES = (PRES_LPRU, HR_LPRU, PRES_SPDR, HR_SPDR)
REFERENCE_FOR = {
    PRES_LPRU: PRES_BASE,
    PRES_SPDR: PRES_BASE,
    HR_LPRU: HR_BASE,
    HR_SPDR: HR_BASE,
}


def is_hrnet(path):
    return 'hrnetv2' in Path(path).name


def method_for(path):
    name = Path(path).stem
    return 'LPRU' if name.endswith('_lpru') else 'SPDR'


def build(path, **overrides):
    pretrained = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
        if is_hrnet(path) else {'PResNet': {'pretrained': False}})
    pretrained.update(overrides)
    return YAMLConfig(str(path), **pretrained)


def fresh_config(path):
    loader = runpy.run_path(str(ROOT / 'src/core/yaml_utils.py'))
    return loader['load_config'](str(path), {})


def flatten(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        if not value:
            result[prefix] = {}
        for key, child in value.items():
            name = f'{prefix}.{key}' if prefix else key
            result.update(flatten(child, name))
    else:
        result[prefix] = value
    return result


def differences(left, right):
    left, right = flatten(left), flatten(right)
    missing = '<absent>'
    return {key: (left.get(key, missing), right.get(key, missing))
            for key in sorted(set(left) | set(right))
            if left.get(key, missing) != right.get(key, missing)}


def new_parameter_prefix(method):
    return ('encoder.lpru54.', 'encoder.lpru43.') if method == 'LPRU' else (
        'encoder.spdr34.', 'encoder.spdr45.')


def assert_tensor_lists_close(case, expected, actual):
    case.assertEqual(len(expected), len(actual))
    for index, (left, right) in enumerate(zip(expected, actual)):
        case.assertTrue(torch.allclose(
            left, right, atol=1e-6, rtol=1e-5), f'level {index}')


class ResampleBlockUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_lpru_pixel_reassembly_formula_and_shapes(self):
        block = LearnablePixelReassemblyUpsample(channels=8)
        x = torch.randn(2, 8, 7, 9)
        base = F.interpolate(x, scale_factor=2.0, mode='nearest')
        output, aux = block(x, base, return_aux=True)
        self.assertEqual(tuple(block.expand.weight.shape), (32, 8, 1, 1))
        self.assertEqual(tuple(aux['rearranged'].shape), (2, 8, 14, 18))
        self.assertEqual(aux['base'].shape, aux['learned'].shape)
        self.assertEqual(output.shape, base.shape)
        self.assertTrue(torch.allclose(
            output, base + aux['alpha'] * (aux['learned'] - base)))
        self.assertTrue(torch.isfinite(output).all())

    def test_lpru_channel_scale_initialization_bounds_and_exact_zero(self):
        block = LearnablePixelReassemblyUpsample(
            channels=8, alpha_max=0.5, alpha_init=0.05)
        self.assertEqual(tuple(block.raw_alpha.shape), (1, 8, 1, 1))
        self.assertTrue(torch.allclose(
            block.effective_alpha(),
            torch.full_like(block.raw_alpha, 0.05), atol=1e-7, rtol=1e-6))
        x = torch.randn(1, 8, 5, 6)
        base = F.interpolate(x, scale_factor=2.0, mode='nearest')
        with torch.no_grad():
            block.raw_alpha.zero_()
        self.assertTrue(torch.equal(block(x, base), base))
        with torch.no_grad():
            block.raw_alpha.fill_(-100.0)
        self.assertTrue((block.effective_alpha() < 0).all())
        self.assertTrue((block.effective_alpha() >= -0.5).all())

    def test_spdr_lossless_rearrangement_formula_and_shapes(self):
        block = SubpixelPreservingDownsample(channels=8)
        x = torch.randn(2, 8, 14, 18)
        base = torch.randn(2, 8, 7, 9)
        output, aux = block(x, base, return_aux=True)
        self.assertEqual(tuple(aux['rearranged'].shape), (2, 32, 7, 9))
        self.assertEqual(aux['rearranged'].numel(), x.numel())
        self.assertEqual(aux['base'].shape, aux['preserved'].shape)
        self.assertEqual(output.shape, base.shape)
        self.assertTrue(torch.allclose(
            output, base + aux['beta'] * (aux['preserved'] - base)))
        self.assertTrue(torch.isfinite(output).all())

    def test_spdr_channel_scale_initialization_bounds_and_exact_zero(self):
        block = SubpixelPreservingDownsample(
            channels=8, beta_max=0.5, beta_init=0.05)
        self.assertEqual(tuple(block.raw_beta.shape), (1, 8, 1, 1))
        self.assertTrue(torch.allclose(
            block.effective_beta(),
            torch.full_like(block.raw_beta, 0.05), atol=1e-7, rtol=1e-6))
        x = torch.randn(1, 8, 10, 12)
        base = torch.randn(1, 8, 5, 6)
        with torch.no_grad():
            block.raw_beta.zero_()
        self.assertTrue(torch.equal(block(x, base), base))
        with self.assertRaisesRegex(RuntimeError, 'must be even'):
            block(torch.randn(1, 8, 11, 12), base)

    def test_blocks_have_no_norm_attention_or_spatial_gate(self):
        blocks = (LearnablePixelReassemblyUpsample(8),
                  SubpixelPreservingDownsample(8))
        forbidden = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm,
                     nn.GroupNorm, nn.MultiheadAttention, nn.Sigmoid,
                     nn.Softmax)
        for block in blocks:
            self.assertFalse(any(isinstance(module, forbidden)
                                 for module in block.modules()))
            self.assertFalse(any('gate' in name.lower()
                                 for name, _ in block.named_modules()))


class ResampleNeckIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_final_top_level_dut_config_set_is_exactly_six(self):
        actual = {path.name for path in CONFIG_DIR.glob(
            'rtdetr*_dut_anti_uav*.yml')}
        expected = {path.name for path in (
            PRES_BASE, HR_BASE, PRES_LPRU, HR_LPRU, PRES_SPDR, HR_SPDR)}
        self.assertEqual(actual, expected)

    def test_candidates_enable_exactly_one_independent_neck(self):
        for path in CANDIDATES:
            model = build(path).model
            method = method_for(path)
            self.assertEqual(model.encoder.lpru_enabled, method == 'LPRU')
            self.assertEqual(model.encoder.spdr_enabled, method == 'SPDR')
            for attribute in ('lpru54', 'lpru43'):
                self.assertEqual(hasattr(model.encoder, attribute),
                                 method == 'LPRU')
            for attribute in ('spdr34', 'spdr45'):
                self.assertEqual(hasattr(model.encoder, attribute),
                                 method == 'SPDR')
            if method == 'LPRU':
                self.assertIsNot(
                    model.encoder.lpru54.expand.weight,
                    model.encoder.lpru43.expand.weight)
            else:
                self.assertIsNot(
                    model.encoder.spdr34.compress.weight,
                    model.encoder.spdr45.compress.weight)

    def test_disabled_hybrid_encoder_is_original_for_both_channel_contracts(self):
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = dict(
                in_channels=channels, hidden_dim=32, nhead=8,
                dim_feedforward=64, expansion=0.5,
                num_encoder_layers=1, eval_spatial_size=None)
            torch.manual_seed(7)
            reference = HybridEncoder(**kwargs).eval()
            torch.manual_seed(7)
            disabled = HybridEncoder(
                **kwargs, LPRU={'enabled': False},
                SPDR={'enabled': False}).eval()
            self.assertFalse(disabled.lpru_enabled)
            self.assertFalse(disabled.spdr_enabled)
            self.assertEqual(reference.state_dict().keys(),
                             disabled.state_dict().keys())
            for key, value in reference.state_dict().items():
                self.assertTrue(torch.equal(
                    value, disabled.state_dict()[key]), key)
            features = [torch.randn(1, channels[0], 16, 16),
                        torch.randn(1, channels[1], 8, 8),
                        torch.randn(1, channels[2], 4, 4)]
            with torch.inference_mode():
                expected = reference(features)
                actual = disabled(features)
            assert_tensor_lists_close(self, expected, actual)

    def test_disabled_full_candidates_match_baseline_features_and_predictions(self):
        image = torch.randn(1, 3, 128, 128)
        for candidate_path in CANDIDATES:
            baseline_path = REFERENCE_FOR[candidate_path]
            method = method_for(candidate_path)
            torch.manual_seed(13)
            reference = build(baseline_path).model.eval()
            torch.manual_seed(13)
            disabled = build(candidate_path, **{
                method: {'enabled': False}}).model.eval()
            self.assertEqual(reference.state_dict().keys(),
                             disabled.state_dict().keys())
            disabled.load_state_dict(reference.state_dict(), strict=True)
            for model in (reference, disabled):
                model.multi_scale = None
                model.encoder.eval_spatial_size = None
                model.decoder.eval_spatial_size = None
            with torch.inference_mode():
                reference_backbone = reference.backbone(image)
                disabled_backbone = disabled.backbone(image)
                reference_neck = reference.encoder(reference_backbone)
                disabled_neck = disabled.encoder(disabled_backbone)
                reference_output = reference(image)
                disabled_output = disabled(image)
            assert_tensor_lists_close(
                self, reference_backbone, disabled_backbone)
            assert_tensor_lists_close(self, reference_neck, disabled_neck)
            for key in ('pred_logits', 'pred_boxes'):
                self.assertTrue(torch.allclose(
                    reference_output[key], disabled_output[key],
                    atol=1e-6, rtol=1e-5), f'{candidate_path.name}: {key}')

    def test_all_four_candidates_support_480_640_800_detector_forward(self):
        for path in CANDIDATES:
            model = build(path).model.eval()
            model.multi_scale = None
            model.encoder.eval_spatial_size = None
            model.decoder.eval_spatial_size = None
            captured = []
            handle = model.encoder.register_forward_hook(
                lambda _module, _inputs, output: captured.append(
                    [tuple(value.shape) for value in output]))
            try:
                for size in (480, 640, 800):
                    with torch.inference_mode():
                        output = model(torch.randn(1, 3, size, size))
                    self.assertEqual(captured[-1], [
                        (1, 256, size // 8, size // 8),
                        (1, 256, size // 16, size // 16),
                        (1, 256, size // 32, size // 32),
                    ], f'{path.name}@{size}')
                    self.assertEqual(tuple(output['pred_logits'].shape),
                                     (1, 300, 1))
                    self.assertEqual(tuple(output['pred_boxes'].shape),
                                     (1, 300, 4))
                    self.assertTrue(all(torch.isfinite(value).all()
                                        for value in output.values()
                                        if torch.is_tensor(value)))
            finally:
                handle.remove()

    def test_backward_reaches_every_new_parameter(self):
        for path in CANDIDATES:
            channels = [36, 72, 144] if is_hrnet(path) else [128, 256, 512]
            model = build(path).model
            encoder = model.encoder.train()
            encoder.eval_spatial_size = None
            features = [
                torch.randn(2, channels[0], 16, 16, requires_grad=True),
                torch.randn(2, channels[1], 8, 8, requires_grad=True),
                torch.randn(2, channels[2], 4, 4, requires_grad=True),
            ]
            output = encoder(features)
            loss = sum(value.float().square().mean() for value in output)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            prefixes = new_parameter_prefix(method_for(path))
            parameters = [(name, parameter)
                          for name, parameter in model.named_parameters()
                          if name.startswith(prefixes)]
            self.assertTrue(parameters)
            problems = [name for name, parameter in parameters
                        if parameter.grad is None
                        or not torch.isfinite(parameter.grad).all()
                        or parameter.grad.abs().sum().item() == 0.0]
            self.assertEqual(problems, [], path.name)

    def test_optimizer_and_resolved_config_fairness(self):
        expected_options = {
            'LPRU': {'enabled': True, 'alpha_max': 0.5,
                     'alpha_init': 0.05},
            'SPDR': {'enabled': True, 'beta_max': 0.5,
                     'beta_init': 0.05},
        }
        for path in CANDIDATES:
            method = method_for(path)
            reference = REFERENCE_FOR[path]
            diff = differences(fresh_config(reference), fresh_config(path))
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', method)
                or key.startswith(method + '.'))]
            self.assertEqual(unexpected, [], f'{path.name}: {diff}')
            configured = fresh_config(path)[method]
            self.assertEqual(configured, expected_options[method])

            config = build(path)
            model, optimizer = config.model, config.optimizer
            assignment = {}
            for group_index, group in enumerate(optimizer.param_groups):
                for parameter in group['params']:
                    assignment[id(parameter)] = (
                        group_index, group['lr'], group['weight_decay'])
            prefixes = new_parameter_prefix(method)
            rows = [(name, parameter) for name, parameter
                    in model.named_parameters() if name.startswith(prefixes)]
            self.assertTrue(rows)
            for name, parameter in rows:
                self.assertIn(id(parameter), assignment, name)
                _group, lr, weight_decay = assignment[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                expected_decay = 0.0 if name.endswith('.bias') else 1e-4
                self.assertAlmostEqual(
                    weight_decay, expected_decay, places=12, msg=name)

    def test_enabled_candidate_preserves_every_common_seeded_weight(self):
        for path in CANDIDATES:
            torch.manual_seed(31)
            reference = build(REFERENCE_FOR[path]).model.state_dict()
            torch.manual_seed(31)
            candidate = build(path).model.state_dict()
            common = set(reference).intersection(candidate)
            self.assertEqual(common, set(reference))
            changed = [key for key in sorted(common)
                       if not torch.equal(reference[key], candidate[key])]
            self.assertEqual(changed, [], path.name)

    def test_lpru_and_spdr_cannot_be_enabled_together(self):
        with self.assertRaisesRegex(ValueError, 'independently'):
            HybridEncoder(
                in_channels=[16, 32, 64], hidden_dim=32,
                nhead=8, dim_feedforward=64,
                LPRU={'enabled': True}, SPDR={'enabled': True})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for AMP')
    def test_cuda_amp_forward_backward_is_finite(self):
        for path in CANDIDATES:
            channels = [36, 72, 144] if is_hrnet(path) else [128, 256, 512]
            encoder = build(path).model.encoder.cuda().train()
            encoder.eval_spatial_size = None
            features = [
                torch.randn(2, channels[0], 16, 16, device='cuda'),
                torch.randn(2, channels[1], 8, 8, device='cuda'),
                torch.randn(2, channels[2], 4, 4, device='cuda'),
            ]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = encoder(features)
                loss = sum(value.float().square().mean() for value in output)
            loss.backward()
            prefixes = new_parameter_prefix(method_for(path))
            bad = [name for name, parameter in encoder.named_parameters()
                   if ('lpru' in name or 'spdr' in name)
                   and (parameter.grad is None
                        or not torch.isfinite(parameter.grad).all())]
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in output))
            self.assertEqual(bad, [], str(prefixes))


if __name__ == '__main__':
    unittest.main()
