"""Unit, integration, and fairness tests for the optional DGFR-Neck."""

from pathlib import Path
import unittest

import torch
import torch.nn as nn

from tests._support import PROJECT_DIR, prepare_imports


prepare_imports()

from src.core import YAMLConfig  # noqa: E402
from src.zoo.rtdetr.dgfr_neck import (  # noqa: E402
    DGFRNeck,
    DirectGlobalFusion,
    DirectScaleAdapter,
)
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder  # noqa: E402
from tools.analyze_dut_models import differences, fresh_config  # noqa: E402


ROOT = PROJECT_DIR
CONFIG_DIR = ROOT / 'configs/rtdetr'
PRES_BASE = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav.yml'
HR_BASE = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav.yml'
PRES_DGFR = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_dgfr.yml'
HR_DGFR = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml'


def is_hrnet(path):
    return 'hrnetv2' in Path(path).name


def build(path, **extra_overrides):
    overrides = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
                 if is_hrnet(path) else {'PResNet': {'pretrained': False}})
    overrides.update(extra_overrides)
    return YAMLConfig(str(path), **overrides)


def assert_tensor_lists_close(test_case, expected, actual):
    test_case.assertEqual(len(expected), len(actual))
    for index, (left, right) in enumerate(zip(expected, actual)):
        test_case.assertTrue(
            torch.allclose(left, right, atol=1e-6, rtol=1e-5),
            f'feature level {index} differs')


class DGFRNeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gamma_is_channelwise_correctly_initialized_bounded_and_signed(self):
        neck = DGFRNeck(
            hidden_dim=32, fusion_channels=8,
            gamma_max=0.25, gamma_init=0.05)
        for level, raw in zip((3, 4, 5), (
                neck.raw_gamma3, neck.raw_gamma4, neck.raw_gamma5)):
            self.assertEqual(tuple(raw.shape), (1, 32, 1, 1))
            gamma = neck.effective_gamma(level)
            self.assertTrue(torch.allclose(
                gamma, torch.full_like(gamma, 0.05),
                atol=1e-7, rtol=1e-6), str(level))

        with torch.no_grad():
            neck.raw_gamma3.fill_(-100.0)
            neck.raw_gamma4.zero_()
            neck.raw_gamma5.fill_(100.0)
        gamma3, gamma4, gamma5 = neck.effective_gamma()
        self.assertTrue((gamma3 < 0).all())
        self.assertTrue((gamma5 > 0).all())
        self.assertTrue((gamma3 >= -0.25).all())
        self.assertTrue((gamma5 <= 0.25).all())
        self.assertTrue(torch.equal(gamma4, torch.zeros_like(gamma4)))

    def test_three_independent_experts_each_receive_all_three_scales(self):
        neck = DGFRNeck(hidden_dim=32, fusion_channels=8).eval()
        experts = [neck.fusion3, neck.fusion4, neck.fusion5]
        self.assertTrue(all(isinstance(value, DirectGlobalFusion)
                            for value in experts))
        self.assertEqual(len({id(value) for value in experts}), 3)

        parameter_sets = [set(map(id, value.parameters())) for value in experts]
        self.assertTrue(parameter_sets[0].isdisjoint(parameter_sets[1]))
        self.assertTrue(parameter_sets[0].isdisjoint(parameter_sets[2]))
        self.assertTrue(parameter_sets[1].isdisjoint(parameter_sets[2]))

        calls = {}
        handles = []
        for target, expert in zip((3, 4, 5), experts):
            self.assertEqual(set(expert.adapters), {'x3', 'x4', 'x5'})
            for source, adapter in expert.adapters.items():
                self.assertIsInstance(adapter, DirectScaleAdapter)
                key = f'{source}_to_{target}'
                calls[key] = 0
                handles.append(adapter.register_forward_hook(
                    lambda _module, _inputs, _output, key=key:
                    calls.__setitem__(key, calls[key] + 1)))

        projected = [torch.randn(1, 32, 17, 19),
                     torch.randn(1, 32, 9, 10),
                     torch.randn(1, 32, 5, 5)]
        original = [torch.randn_like(value) for value in projected]
        try:
            with torch.inference_mode():
                output = neck(projected, original)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, {f'x{source}_to_{target}': 1
                                 for target in (3, 4, 5)
                                 for source in (3, 4, 5)})
        self.assertEqual([tuple(value.shape) for value in output],
                         [tuple(value.shape) for value in original])

    def test_adapter_routes_use_prescribed_convolutions_without_pooling(self):
        expected = {
            (3, 3): [((1, 1), (1, 1))],
            (4, 3): [((1, 1), (1, 1))],
            (5, 3): [((1, 1), (1, 1))],
            (3, 4): [((3, 3), (2, 2))],
            (4, 4): [((1, 1), (1, 1))],
            (5, 4): [((1, 1), (1, 1))],
            (3, 5): [((3, 3), (2, 2)), ((3, 3), (2, 2))],
            (4, 5): [((3, 3), (2, 2))],
            (5, 5): [((1, 1), (1, 1))],
        }
        for (source, target), route in expected.items():
            adapter = DirectScaleAdapter(
                source_level=source, target_level=target,
                hidden_dim=32, fusion_channels=8)
            convolutions = [module for module in adapter.modules()
                            if isinstance(module, nn.Conv2d)]
            actual = [(module.kernel_size, module.stride)
                      for module in convolutions]
            self.assertEqual(actual, route, f'P{source}->P{target}')
            self.assertFalse(any(isinstance(module, (
                nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                nn.AdaptiveMaxPool2d)) for module in adapter.modules()))

    def test_no_dynamic_gate_or_attention_and_debug_statistics_are_finite(self):
        neck = DGFRNeck(
            hidden_dim=32, fusion_channels=8,
            debug=True, debug_interval=1).eval()
        forbidden_types = (nn.Sigmoid, nn.Softmax, nn.MultiheadAttention)
        self.assertFalse(any(isinstance(module, forbidden_types)
                             for module in neck.modules()))
        self.assertFalse(any('gate' in name.lower()
                             for name, _module in neck.named_modules()))

        projected = [torch.randn(1, 32, 16, 16),
                     torch.randn(1, 32, 8, 8),
                     torch.randn(1, 32, 4, 4)]
        original = [torch.randn_like(value) for value in projected]
        with torch.inference_mode():
            output = neck(projected, original)
        self.assertEqual([tuple(value.shape) for value in output],
                         [tuple(value.shape) for value in original])
        expected_keys = {
            *(f'gamma{level}_{stat}' for level in (3, 4, 5)
              for stat in ('mean', 'min', 'max')),
            *(f'e{level}_to_o{level}_norm_ratio' for level in (3, 4, 5)),
            *(f'y{level}_norm' for level in (3, 4, 5)),
        }
        self.assertEqual(set(neck.last_debug_stats), expected_keys)
        for value in neck.last_debug_stats.values():
            self.assertTrue(torch.is_tensor(value))
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value).all())

    def test_disabled_hybrid_encoder_is_the_original_path(self):
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = dict(
                in_channels=channels, hidden_dim=32, nhead=8,
                dim_feedforward=64, expansion=0.5,
                num_encoder_layers=1, eval_spatial_size=None)
            torch.manual_seed(7)
            reference = HybridEncoder(**kwargs, DGFR=None).eval()
            torch.manual_seed(7)
            disabled = HybridEncoder(
                **kwargs, DGFR={'enabled': False}).eval()
            self.assertFalse(disabled.dgfr_enabled)
            self.assertFalse(hasattr(disabled, 'dgfr'))
            self.assertEqual(reference.state_dict().keys(),
                             disabled.state_dict().keys())
            for key in reference.state_dict():
                self.assertTrue(torch.equal(
                    reference.state_dict()[key],
                    disabled.state_dict()[key]), key)
            features = [torch.randn(1, channels[0], 16, 16),
                        torch.randn(1, channels[1], 8, 8),
                        torch.randn(1, channels[2], 4, 4)]
            with torch.inference_mode():
                expected = reference(features)
                actual = disabled(features)
            assert_tensor_lists_close(self, expected, actual)

    def test_disabled_full_models_match_features_and_predictions(self):
        image = torch.randn(1, 3, 128, 128)
        for baseline_path, candidate_path in (
                (PRES_BASE, PRES_DGFR), (HR_BASE, HR_DGFR)):
            torch.manual_seed(13)
            reference = build(baseline_path).model.eval()
            torch.manual_seed(13)
            disabled = build(
                candidate_path, DGFR={'enabled': False}).model.eval()
            self.assertFalse(disabled.encoder.dgfr_enabled)
            self.assertFalse(hasattr(disabled.encoder, 'dgfr'))
            disabled.load_state_dict(reference.state_dict(), strict=True)
            for model in (reference, disabled):
                model.multi_scale = None
                model.encoder.eval_spatial_size = None
                model.decoder.eval_spatial_size = None
            with torch.inference_mode():
                expected_backbone = reference.backbone(image)
                actual_backbone = disabled.backbone(image)
                expected_neck = reference.encoder(expected_backbone)
                actual_neck = disabled.encoder(actual_backbone)
                expected_prediction = reference(image)
                actual_prediction = disabled(image)
            assert_tensor_lists_close(
                self, expected_backbone, actual_backbone)
            assert_tensor_lists_close(self, expected_neck, actual_neck)
            for key in ('pred_logits', 'pred_boxes'):
                self.assertTrue(torch.allclose(
                    expected_prediction[key], actual_prediction[key],
                    atol=1e-6, rtol=1e-5), key)

    def test_dynamic_backbone_encoder_shapes_for_both_candidates(self):
        for path in (PRES_DGFR, HR_DGFR):
            model = build(path).model
            model.backbone.eval()
            model.encoder.eval()
            model.encoder.eval_spatial_size = None
            for size in (480, 640, 800):
                image = torch.randn(1, 3, size, size)
                with torch.inference_mode():
                    backbone_output = model.backbone(image)
                    encoded = model.encoder(backbone_output)
                self.assertEqual(
                    [tuple(value.shape) for value in encoded],
                    [(1, 256, size // 8, size // 8),
                     (1, 256, size // 16, size // 16),
                     (1, 256, size // 32, size // 32)])
                self.assertTrue(all(torch.isfinite(value).all()
                                    for value in encoded))

    def test_full_detector_640_shapes_for_both_candidates(self):
        image = torch.randn(1, 3, 640, 640)
        for path in (PRES_DGFR, HR_DGFR):
            model = build(path).model.eval()
            model.multi_scale = None
            with torch.inference_mode():
                output = model(image)
            self.assertEqual(tuple(output['pred_logits'].shape), (1, 300, 1))
            self.assertEqual(tuple(output['pred_boxes'].shape), (1, 300, 4))
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in output.values()
                                if torch.is_tensor(value)))

    def test_backward_reaches_every_dgfr_parameter_for_both_candidates(self):
        for path, channels in (
                (PRES_DGFR, [128, 256, 512]),
                (HR_DGFR, [36, 72, 144])):
            encoder = build(path).model.encoder.train()
            encoder.eval_spatial_size = None
            features = [
                torch.randn(2, channels[0], 16, 16, requires_grad=True),
                torch.randn(2, channels[1], 8, 8, requires_grad=True),
                torch.randn(2, channels[2], 4, 4, requires_grad=True),
            ]
            output = encoder(features)
            loss = sum(value.float().square().mean() for value in output)
            loss.backward()
            names = []
            problems = []
            for name, parameter in encoder.dgfr.named_parameters():
                names.append(name)
                if parameter.grad is None:
                    problems.append(name + ': missing')
                elif not torch.isfinite(parameter.grad).all():
                    problems.append(name + ': nonfinite')
                elif parameter.grad.abs().sum().item() == 0.0:
                    problems.append(name + ': zero')
            self.assertTrue(names)
            self.assertEqual(problems, [])
            for required in ('raw_gamma3', 'raw_gamma4', 'raw_gamma5'):
                self.assertIn(required, names)

    def test_optimizer_groups_and_resolved_config_fairness(self):
        for reference, candidate in (
                (PRES_BASE, PRES_DGFR), (HR_BASE, HR_DGFR)):
            diff = differences(fresh_config(reference), fresh_config(candidate))
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', 'DGFR')
                or key.startswith('DGFR.'))]
            self.assertEqual(unexpected, [], str(diff))

            config = build(candidate)
            model, optimizer = config.model, config.optimizer
            assignments = {}
            for group_index, group in enumerate(optimizer.param_groups):
                for parameter in group['params']:
                    assignments[id(parameter)] = (
                        group_index, group['lr'], group['weight_decay'])
            names = []
            for name, parameter in model.named_parameters():
                if not name.startswith('encoder.dgfr.'):
                    continue
                names.append(name)
                self.assertIn(id(parameter), assignments, name)
                _group, lr, weight_decay = assignments[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                expected_decay = (0.0 if name.endswith('.bias')
                                  or '.norm.' in name else 1e-4)
                self.assertAlmostEqual(
                    weight_decay, expected_decay, places=12, msg=name)
            self.assertTrue(names)

    def test_config_does_not_leak_and_other_modules_cannot_mix(self):
        candidate = build(PRES_DGFR).model
        baseline = build(PRES_BASE).model
        self.assertTrue(candidate.encoder.dgfr_enabled)
        self.assertFalse(baseline.encoder.dgfr_enabled)
        self.assertFalse(hasattr(baseline.encoder, 'dgfr'))

        common = dict(in_channels=[16, 32, 64], hidden_dim=32,
                      nhead=8, dim_feedforward=64)
        for namespace in ('ACR', 'SLR', 'PAF', 'BOR'):
            options = {'enabled': True}
            if namespace == 'SLR':
                options['detail_source_channels'] = 8
            with self.subTest(namespace=namespace):
                with self.assertRaises(ValueError):
                    HybridEncoder(
                        **common, DGFR={'enabled': True},
                        **{namespace: options})

    def test_enabled_dgfr_preserves_every_common_seeded_weight(self):
        for reference_path, candidate_path in (
                (PRES_BASE, PRES_DGFR), (HR_BASE, HR_DGFR)):
            torch.manual_seed(31)
            reference = build(reference_path).model.state_dict()
            torch.manual_seed(31)
            candidate = build(candidate_path).model.state_dict()
            common = set(reference).intersection(candidate)
            self.assertEqual(common, set(reference))
            changed = [key for key in sorted(common)
                       if not torch.equal(reference[key], candidate[key])]
            self.assertEqual(changed, [])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for AMP')
    def test_cuda_amp_is_finite_for_both_candidates(self):
        for path, channels in (
                (PRES_DGFR, [128, 256, 512]),
                (HR_DGFR, [36, 72, 144])):
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
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in output))
            bad = [name for name, parameter
                   in encoder.dgfr.named_parameters()
                   if parameter.grad is None
                   or not torch.isfinite(parameter.grad).all()]
            self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
