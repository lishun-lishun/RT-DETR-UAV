"""Acceptance tests for the optional ACR-Neck routes."""

import copy
from pathlib import Path
import unittest

import torch

from tests._support import PROJECT_DIR, prepare_imports
prepare_imports()

from src.core import YAMLConfig  # noqa: E402
from src.zoo.rtdetr.acr_neck import (  # noqa: E402
    CrossScaleEnergyCalibration, ScaleExclusiveResidualRouter)
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder  # noqa: E402
from tools.analyze_dut_models import differences, fresh_config  # noqa: E402


ROOT = PROJECT_DIR
BASELINE = ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml'
BASELINE_ACR = ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_acr.yml'
HRNET = ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml'
HRNET_ACR = ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_acr.yml'


def build(path):
    overrides = ({'PResNet': {'pretrained': False}}
                 if path in (BASELINE, BASELINE_ACR) else
                 {'HRNetV2W18': {'pretrained': False,
                                 'pretrained_path': None}})
    return YAMLConfig(str(path), **overrides)


class ACRNeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disabled_is_original_ccff_exactly(self):
        kwargs = dict(in_channels=[128, 256, 512], hidden_dim=256,
                      expansion=0.5, num_encoder_layers=1,
                      eval_spatial_size=[128, 128])
        torch.manual_seed(7)
        original = HybridEncoder(**kwargs, ACR=None).eval()
        torch.manual_seed(7)
        disabled = HybridEncoder(**kwargs, ACR={'enabled': False}).eval()
        self.assertEqual(original.state_dict().keys(), disabled.state_dict().keys())
        for key in original.state_dict():
            self.assertTrue(torch.equal(original.state_dict()[key],
                                        disabled.state_dict()[key]), key)
        features = [torch.randn(1, 128, 16, 16),
                    torch.randn(1, 256, 8, 8),
                    torch.randn(1, 512, 4, 4)]
        with torch.inference_mode():
            expected, actual = original(features), disabled(features)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-5))

    def test_candidate_yaml_changes_only_acr_and_output(self):
        for reference, candidate in ((BASELINE, BASELINE_ACR),
                                     (HRNET, HRNET_ACR)):
            diff = differences(fresh_config(reference), fresh_config(candidate))
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', 'ACR')
                or key.startswith('ACR.'))]
            self.assertEqual(unexpected, [])

    def test_enabled_config_does_not_leak_into_later_baseline(self):
        enabled = build(BASELINE_ACR).model
        baseline = build(BASELINE).model
        self.assertTrue(enabled.encoder.acr_enabled)
        self.assertFalse(baseline.encoder.acr_enabled)
        self.assertFalse(hasattr(baseline.encoder, 'acr_43'))
        self.assertFalse(hasattr(baseline.encoder, 'acr_54'))

    def test_beta_parameterization_starts_at_point_one(self):
        for route in ('34', '45'):
            router = ScaleExclusiveResidualRouter(16, route)
            self.assertAlmostEqual(router.effective_beta().item(), 0.1, places=7)

    def test_energy_scale_detaches_and_is_clipped(self):
        layer = CrossScaleEnergyCalibration(
            scale_min=0.5, scale_max=2.0, detach_scale=True)
        shallow = torch.full((1, 2, 4, 4), 10.0, requires_grad=True)
        deep = torch.ones(1, 2, 4, 4, requires_grad=True)
        calibrated, scale = layer(shallow, deep)
        self.assertFalse(scale.requires_grad)
        self.assertTrue(torch.allclose(scale, torch.full_like(scale, 2.0)))
        calibrated.sum().backward()
        self.assertIsNotNone(deep.grad)

    def test_full_detector_shapes_for_both_backbones(self):
        image = torch.randn(1, 3, 640, 640)
        for path in (BASELINE_ACR, HRNET_ACR):
            model = build(path).model.eval()
            captured = {}

            def hook(module, inputs, output):
                captured['features'] = [tuple(value.shape) for value in output]

            handle = model.encoder.register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    output = model(image)
            finally:
                handle.remove()
            self.assertEqual(captured['features'], [
                (1, 256, 80, 80), (1, 256, 40, 40), (1, 256, 20, 20)])
            self.assertEqual(tuple(output['pred_logits'].shape), (1, 300, 1))
            self.assertEqual(tuple(output['pred_boxes'].shape), (1, 300, 4))
            self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))

    def test_acr_trainable_parameters_receive_gradients(self):
        encoder = build(BASELINE_ACR).model.encoder.train()
        features = [torch.randn(1, 128, 16, 16, requires_grad=True),
                    torch.randn(1, 256, 8, 8, requires_grad=True),
                    torch.randn(1, 512, 4, 4, requires_grad=True)]
        output = encoder(features)
        loss = sum(value.square().mean() for value in output)
        loss.backward()
        missing = [name for name, parameter in encoder.named_parameters()
                   if name.startswith('acr_') and parameter.requires_grad
                   and parameter.grad is None]
        nonfinite = [name for name, parameter in encoder.named_parameters()
                     if name.startswith('acr_') and parameter.grad is not None
                     and not torch.isfinite(parameter.grad).all()]
        self.assertEqual(missing, [])
        self.assertEqual(nonfinite, [])
        self.assertIsNotNone(features[0].grad)
        self.assertIsNotNone(features[1].grad)
        self.assertIsNotNone(features[2].grad)

    def test_optimizer_groups_use_main_lr_and_encoder_norm_policy(self):
        for path in (BASELINE_ACR, HRNET_ACR):
            cfg = build(path)
            model, optimizer = cfg.model, cfg.optimizer
            groups = {}
            for index, group in enumerate(optimizer.param_groups):
                for parameter in group['params']:
                    groups[id(parameter)] = (index, group['lr'], group['weight_decay'])
            acr_names = []
            for name, parameter in model.named_parameters():
                if not name.startswith('encoder.acr_'):
                    continue
                acr_names.append(name)
                _, lr, weight_decay = groups[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                if '.norm.' in name:
                    self.assertEqual(weight_decay, 0.0, name)
                else:
                    self.assertAlmostEqual(weight_decay, 1e-4, places=12,
                                           msg=name)
            self.assertTrue(acr_names)

    def test_ablation_switches_construct_only_requested_routes(self):
        base = dict(in_channels=[8, 16, 32], hidden_dim=16,
                    nhead=4, dim_feedforward=32, num_encoder_layers=0,
                    depth_mult=0.34, eval_spatial_size=[64, 64])
        energy_only = HybridEncoder(**base, ACR={
            'enabled': True, 'energy_calibration': True,
            'semantic_routing': False, 'detail_routing': False})
        self.assertIsNone(energy_only.acr_43.detail_router)
        semantic = HybridEncoder(**base, ACR={
            'enabled': True, 'energy_calibration': True,
            'semantic_routing': True, 'detail_routing': False})
        self.assertIsNone(semantic.acr_54.detail_router)
        full = HybridEncoder(**base, ACR={
            'enabled': True, 'energy_calibration': True,
            'semantic_routing': True, 'detail_routing': True})
        self.assertIsNotNone(full.acr_43.detail_router)
        self.assertIsNotNone(full.acr_54.detail_router)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for AMP')
    def test_cuda_amp_forward_backward_is_finite(self):
        encoder = build(BASELINE_ACR).model.encoder.cuda().train()
        features = [torch.randn(1, 128, 32, 32, device='cuda'),
                    torch.randn(1, 256, 16, 16, device='cuda'),
                    torch.randn(1, 512, 8, 8, device='cuda')]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = encoder(features)
            loss = sum(value.square().mean() for value in output)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value).all() for value in output))


if __name__ == '__main__':
    unittest.main()
