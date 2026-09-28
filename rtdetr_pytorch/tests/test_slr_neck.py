"""Acceptance tests for Sub-cell Localization Relay Neck."""

from pathlib import Path
import unittest

import torch

from tests._support import PROJECT_DIR, prepare_imports

prepare_imports()

from src.core import YAMLConfig  # noqa: E402
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder  # noqa: E402
from src.zoo.rtdetr.slr_neck import SLRNeck  # noqa: E402
from tools.analyze_dut_models import differences, fresh_config  # noqa: E402


ROOT = PROJECT_DIR
PRES_BASE = ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml'
PRES_SLR = ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_slr.yml'
HR_BASE = ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml'
HR_SLR = ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_slr.yml'


def build(path):
    if path in (HR_BASE, HR_SLR):
        overrides = {'HRNetV2W18': {
            'pretrained': False, 'pretrained_path': None}}
    else:
        overrides = {'PResNet': {'pretrained': False}}
    return YAMLConfig(str(path), **overrides)


class SLRNeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pixel_unshuffle_phase_order_residual_and_attention(self):
        module = SLRNeck(4, 2, query_dim=2, position_dim=2)
        detail = torch.tensor([[[[0., 1.], [2., 3.]],
                                [[10., 11.], [12., 13.]]]])
        phases = module._phase_features(detail)
        expected = torch.tensor([[[[[0.]], [[10.]]],
                                  [[[1.]], [[11.]]],
                                  [[[2.]], [[12.]]],
                                  [[[3.]], [[13.]]]]])
        self.assertTrue(torch.equal(phases, expected))
        residuals = phases - phases.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(
            residuals.sum(dim=1), torch.zeros_like(residuals[:, 0])))
        _, aux = module(torch.randn(1, 4, 1, 1), detail, return_aux=True)
        self.assertEqual(tuple(aux['attention'].shape), (1, 4, 1, 1))
        self.assertTrue(torch.allclose(
            aux['attention'].sum(dim=1), torch.ones(1, 1, 1),
            atol=1e-6, rtol=1e-6))
        self.assertTrue((aux['delta_position_xy'].abs() <= 0.25 + 1e-7).all())
        expected_positions = torch.tensor([
            [-0.25, -0.25], [0.25, -0.25],
            [-0.25, 0.25], [0.25, 0.25]])
        self.assertTrue(torch.equal(
            module.positions_xy.view(4, 2), expected_positions))

    def test_alpha_initialization(self):
        module = SLRNeck(8, 4, alpha_max=0.3, alpha_init=0.05)
        self.assertAlmostEqual(module.effective_alpha().item(), 0.05, places=7)

    def test_disabled_hybrid_encoder_is_exact_original(self):
        kwargs = dict(in_channels=[128, 256, 512], hidden_dim=256,
                      expansion=0.5, num_encoder_layers=1,
                      eval_spatial_size=[128, 128])
        torch.manual_seed(11)
        original = HybridEncoder(**kwargs, ACR=None, SLR=None).eval()
        torch.manual_seed(11)
        disabled = HybridEncoder(
            **kwargs, ACR={'enabled': False}, SLR={'enabled': False}).eval()
        self.assertEqual(original.state_dict().keys(), disabled.state_dict().keys())
        for key in original.state_dict():
            self.assertTrue(torch.equal(
                original.state_dict()[key], disabled.state_dict()[key]), key)
        features = [torch.randn(1, 128, 16, 16),
                    torch.randn(1, 256, 8, 8),
                    torch.randn(1, 512, 4, 4)]
        with torch.inference_mode():
            expected, actual = original(features), disabled(features)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-5))

    def test_disabled_full_detector_predictions_are_exact(self):
        torch.manual_seed(19)
        original = build(PRES_BASE).model.eval()
        torch.manual_seed(19)
        disabled = YAMLConfig(
            str(PRES_BASE), PResNet={'pretrained': False},
            SLR={'enabled': False}).model.eval()
        disabled.load_state_dict(original.state_dict(), strict=True)
        image = torch.randn(1, 3, 640, 640)
        with torch.inference_mode():
            expected, actual = original(image), disabled(image)
        for key in ('pred_logits', 'pred_boxes'):
            self.assertTrue(torch.allclose(
                expected[key], actual[key], atol=1e-6, rtol=1e-5), key)

    def test_backbone_detail_sources_and_public_contract(self):
        image = torch.randn(1, 3, 640, 640)
        cases = (
            (PRES_BASE, PRES_SLR, 64, [128, 256, 512]),
            (HR_BASE, HR_SLR, 18, [36, 72, 144]),
        )
        for base_path, slr_path, detail_channels, public_channels in cases:
            base = build(base_path).model.backbone.eval()
            candidate = build(slr_path).model.backbone.eval()
            with torch.inference_mode():
                original = base(image)
                exposed = candidate(image)
            self.assertIsInstance(original, list)
            self.assertEqual(set(exposed), {'features', 'detail'})
            self.assertEqual(tuple(exposed['detail'].shape),
                             (1, detail_channels, 160, 160))
            self.assertEqual(
                [tuple(value.shape) for value in exposed['features']],
                [(1, public_channels[0], 80, 80),
                 (1, public_channels[1], 40, 40),
                 (1, public_channels[2], 20, 20)])
            self.assertEqual(candidate.out_strides, [8, 16, 32])

    def test_only_n3_is_modified_after_original_ccff(self):
        model = build(PRES_SLR).model.eval()
        model.encoder.eval_spatial_size = None
        features = [torch.randn(1, 128, 16, 16),
                    torch.randn(1, 256, 8, 8),
                    torch.randn(1, 512, 4, 4)]
        detail = torch.randn(1, 64, 32, 32)
        with torch.inference_mode():
            model.encoder.slr_enabled = False
            original = model.encoder(features)
            model.encoder.slr_enabled = True
            enhanced = model.encoder({'features': features, 'detail': detail})
        self.assertFalse(torch.equal(original[0], enhanced[0]))
        self.assertTrue(torch.equal(original[1], enhanced[1]))
        self.assertTrue(torch.equal(original[2], enhanced[2]))

    def test_dynamic_training_shapes_for_both_backbones(self):
        for path, channels in ((PRES_SLR, 64), (HR_SLR, 18)):
            model = build(path).model
            model.backbone.eval()
            model.encoder.train()
            for size in (480, 640, 800):
                image = torch.randn(1, 3, size, size)
                with torch.no_grad():
                    backbone_output = model.backbone(image)
                    encoded = model.encoder(backbone_output)
                self.assertEqual(tuple(backbone_output['detail'].shape[-2:]),
                                 (size // 4, size // 4))
                self.assertEqual(backbone_output['detail'].shape[1], channels)
                self.assertEqual(
                    [tuple(value.shape[-2:]) for value in encoded],
                    [(size // 8, size // 8),
                     (size // 16, size // 16),
                     (size // 32, size // 32)])

    def test_full_detector_640_shapes_for_both_candidates(self):
        image = torch.randn(1, 3, 640, 640)
        for path in (PRES_SLR, HR_SLR):
            model = build(path).model.eval()
            with torch.inference_mode():
                output = model(image)
            self.assertEqual(tuple(output['pred_logits'].shape), (1, 300, 1))
            self.assertEqual(tuple(output['pred_boxes'].shape), (1, 300, 4))
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in output.values()))

    def test_backward_reaches_every_slr_parameter_for_both_candidates(self):
        cases = ((PRES_SLR, [128, 256, 512], 64),
                 (HR_SLR, [36, 72, 144], 18))
        for path, channels, detail_channels in cases:
            encoder = build(path).model.encoder.train()
            features = [torch.randn(1, channels[0], 16, 16, requires_grad=True),
                        torch.randn(1, channels[1], 8, 8, requires_grad=True),
                        torch.randn(1, channels[2], 4, 4, requires_grad=True)]
            detail = torch.randn(
                1, detail_channels, 32, 32, requires_grad=True)
            outputs = encoder({'features': features, 'detail': detail})
            loss = sum(value.square().mean() for value in outputs)
            loss.backward()
            problems = []
            for name, parameter in encoder.slr.named_parameters():
                if (parameter.grad is None
                        or not torch.isfinite(parameter.grad).all()
                        or parameter.grad.abs().sum() == 0):
                    problems.append(name)
            self.assertEqual(problems, [])
            self.assertTrue(torch.isfinite(detail.grad).all())
            self.assertGreater(detail.grad.abs().sum().item(), 0.0)

    def test_optimizer_groups_and_config_fairness(self):
        pairs = ((PRES_BASE, PRES_SLR), (HR_BASE, HR_SLR))
        for reference, candidate in pairs:
            diff = differences(fresh_config(reference), fresh_config(candidate))
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', 'SLR')
                or key.startswith('SLR.'))]
            self.assertEqual(unexpected, [])

            cfg = build(candidate)
            model, optimizer = cfg.model, cfg.optimizer
            assignments = {}
            for group_index, group in enumerate(optimizer.param_groups):
                for parameter in group['params']:
                    assignments[id(parameter)] = (
                        group_index, group['lr'], group['weight_decay'])
            names = []
            for name, parameter in model.named_parameters():
                if not name.startswith('encoder.slr.'):
                    continue
                names.append(name)
                _, lr, weight_decay = assignments[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                expected_decay = 0.0 if 'bias' in name or '.norm.' in name else 1e-4
                self.assertAlmostEqual(
                    weight_decay, expected_decay, places=12, msg=name)
            self.assertTrue(names)

    def test_no_shared_config_leak_and_exact_final_yaml_set(self):
        candidate = build(PRES_SLR).model
        baseline = build(PRES_BASE).model
        self.assertTrue(candidate.encoder.slr_enabled)
        self.assertFalse(baseline.encoder.slr_enabled)
        self.assertFalse(hasattr(baseline.encoder, 'slr'))
        actual = sorted(path.name for path in
                        (ROOT / 'configs/rtdetr').glob('*dut_anti_uav*.yml'))
        self.assertEqual(actual, sorted([
            PRES_BASE.name, HR_BASE.name, PRES_SLR.name, HR_SLR.name,
            'rtdetr_r18vd_dut_anti_uav_paf.yml',
            'rtdetr_hrnetv2_w18_dut_anti_uav_paf.yml',
            'rtdetr_r18vd_dut_anti_uav_bor.yml',
            'rtdetr_hrnetv2_w18_dut_anti_uav_bor.yml']))

    def test_enabled_slr_preserves_all_common_seeded_weights(self):
        for baseline_path, candidate_path in (
                (PRES_BASE, PRES_SLR), (HR_BASE, HR_SLR)):
            torch.manual_seed(31)
            baseline = build(baseline_path).model.state_dict()
            torch.manual_seed(31)
            candidate = build(candidate_path).model.state_dict()
            common = set(baseline).intersection(candidate)
            self.assertEqual(common, set(baseline))
            changed = [key for key in common
                       if not torch.equal(baseline[key], candidate[key])]
            self.assertEqual(changed, [])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for AMP')
    def test_cuda_amp_is_finite_for_both_candidates(self):
        cases = ((PRES_SLR, [128, 256, 512], 64),
                 (HR_SLR, [36, 72, 144], 18))
        for path, channels, detail_channels in cases:
            encoder = build(path).model.encoder.cuda().train()
            features = [torch.randn(1, channels[0], 32, 32, device='cuda'),
                        torch.randn(1, channels[1], 16, 16, device='cuda'),
                        torch.randn(1, channels[2], 8, 8, device='cuda')]
            detail = torch.randn(
                1, detail_channels, 64, 64, device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = encoder({'features': features, 'detail': detail})
                loss = sum(value.square().mean() for value in outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in outputs))
            self.assertTrue(all(parameter.grad is not None
                                and torch.isfinite(parameter.grad).all()
                                for parameter in encoder.slr.parameters()))


if __name__ == '__main__':
    unittest.main()
