"""Integration and fairness tests for the optional PAF/BOR necks."""

from pathlib import Path
import unittest

import torch

from tests._support import PROJECT_DIR, prepare_imports


prepare_imports()

from src.core import YAMLConfig  # noqa: E402
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder  # noqa: E402
from tools.analyze_dut_models import differences, fresh_config  # noqa: E402


ROOT = PROJECT_DIR
CONFIG_DIR = ROOT / 'configs/rtdetr'
PRES_BASE = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav.yml'
HR_BASE = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav.yml'
PRES_PAF = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_paf.yml'
HR_PAF = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_paf.yml'
PRES_BOR = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_bor.yml'
HR_BOR = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_bor.yml'


def is_hrnet(path):
    return 'hrnetv2' in path.name


def build(path):
    overrides = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
                 if is_hrnet(path) else {'PResNet': {'pretrained': False}})
    return YAMLConfig(str(path), **overrides)


class PAFBORIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_disabled_is_original_hybrid_encoder(self):
        kwargs = dict(in_channels=[16, 32, 64], hidden_dim=32,
                      nhead=8, dim_feedforward=64, expansion=0.5,
                      num_encoder_layers=1, eval_spatial_size=None)
        torch.manual_seed(7)
        original = HybridEncoder(**kwargs, PAF=None, BOR=None).eval()
        torch.manual_seed(7)
        disabled = HybridEncoder(
            **kwargs, PAF={'enabled': False}, BOR={'enabled': False}).eval()
        self.assertEqual(original.state_dict().keys(), disabled.state_dict().keys())
        for key in original.state_dict():
            self.assertTrue(torch.equal(
                original.state_dict()[key], disabled.state_dict()[key]), key)
        features = [torch.randn(1, 16, 16, 16),
                    torch.randn(1, 32, 8, 8),
                    torch.randn(1, 64, 4, 4)]
        with torch.inference_mode():
            expected = original(features)
            actual = disabled(features)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.allclose(
                left, right, atol=1e-6, rtol=1e-5))

    def test_paf_routes_and_bor_only_n3(self):
        paf = build(PRES_PAF).model.encoder.eval()
        paf.eval_spatial_size = None
        calls = {'54': 0, '43': 0}
        handles = [
            paf.paf54.register_forward_hook(
                lambda *args: calls.__setitem__('54', calls['54'] + 1)),
            paf.paf43.register_forward_hook(
                lambda *args: calls.__setitem__('43', calls['43'] + 1)),
        ]
        features = [torch.randn(1, 128, 16, 16),
                    torch.randn(1, 256, 8, 8),
                    torch.randn(1, 512, 4, 4)]
        try:
            with torch.inference_mode():
                paf_outputs = paf(features)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, {'54': 1, '43': 1})
        self.assertEqual([tuple(value.shape[-2:]) for value in paf_outputs],
                         [(16, 16), (8, 8), (4, 4)])

        bor = build(PRES_BOR).model.encoder.eval()
        bor.eval_spatial_size = None
        with torch.inference_mode():
            bor.bor_enabled = False
            original = bor(features)
            bor.bor_enabled = True
            enhanced = bor(features)
        self.assertFalse(torch.equal(original[0], enhanced[0]))
        self.assertTrue(torch.equal(original[1], enhanced[1]))
        self.assertTrue(torch.equal(original[2], enhanced[2]))

    def test_mutual_exclusion_and_config_leak_protection(self):
        with self.assertRaisesRegex(
                ValueError, 'PAF and BOR cannot be enabled simultaneously'):
            HybridEncoder(
                in_channels=[16, 32, 64], hidden_dim=32,
                PAF={'enabled': True}, BOR={'enabled': True})
        with self.assertRaisesRegex(ValueError, 'cannot mix with ACR or SLR'):
            HybridEncoder(
                in_channels=[16, 32, 64], hidden_dim=32,
                PAF={'enabled': True}, SLR={
                    'enabled': True, 'detail_source_channels': 8})

        candidate = build(PRES_PAF).model
        baseline = build(PRES_BASE).model
        self.assertTrue(candidate.encoder.paf_enabled)
        self.assertFalse(baseline.encoder.paf_enabled)
        self.assertFalse(baseline.encoder.bor_enabled)
        self.assertFalse(hasattr(baseline.encoder, 'paf54'))
        self.assertFalse(hasattr(baseline.encoder, 'bor'))

    def test_four_resolved_configs_are_fair(self):
        for reference, candidate, namespace in (
                (PRES_BASE, PRES_PAF, 'PAF'),
                (HR_BASE, HR_PAF, 'PAF'),
                (PRES_BASE, PRES_BOR, 'BOR'),
                (HR_BASE, HR_BOR, 'BOR')):
            diff = differences(fresh_config(reference), fresh_config(candidate))
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', namespace)
                or key.startswith(namespace + '.'))]
            self.assertEqual(unexpected, [], str(diff))

    def test_backward_and_optimizer_groups_for_all_four_models(self):
        cases = (
            (PRES_PAF, [128, 256, 512], 'paf'),
            (HR_PAF, [36, 72, 144], 'paf'),
            (PRES_BOR, [128, 256, 512], 'bor'),
            (HR_BOR, [36, 72, 144], 'bor'),
        )
        for path, channels, module_name in cases:
            config = build(path)
            model = config.model
            encoder = model.encoder.train()
            encoder.eval_spatial_size = None
            features = [
                torch.randn(1, channels[0], 16, 16),
                torch.randn(1, channels[1], 8, 8),
                torch.randn(1, channels[2], 4, 4),
            ]
            outputs = encoder(features)
            sum(value.square().mean() for value in outputs).backward()

            prefix = 'encoder.paf' if module_name == 'paf' else 'encoder.bor.'
            named = [(name, parameter) for name, parameter
                     in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(named)
            bad = [name for name, parameter in named
                   if parameter.grad is None
                   or not torch.isfinite(parameter.grad).all()
                   or parameter.grad.abs().sum() == 0]
            self.assertEqual(bad, [])

            assignments = {}
            for group_index, group in enumerate(config.optimizer.param_groups):
                for parameter in group['params']:
                    assignments[id(parameter)] = (
                        group_index, group['lr'], group['weight_decay'])
            for name, parameter in named:
                _, lr, decay = assignments[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                expected_decay = 0.0 if 'bias' in name else 1e-4
                self.assertAlmostEqual(
                    decay, expected_decay, places=12, msg=name)

    def test_enabled_necks_preserve_every_common_seeded_weight(self):
        for reference, candidate in (
                (PRES_BASE, PRES_PAF), (HR_BASE, HR_PAF),
                (PRES_BASE, PRES_BOR), (HR_BASE, HR_BOR)):
            torch.manual_seed(31)
            original = build(reference).model.state_dict()
            torch.manual_seed(31)
            enhanced = build(candidate).model.state_dict()
            common = set(original).intersection(enhanced)
            self.assertEqual(common, set(original))
            changed = [key for key in common
                       if not torch.equal(original[key], enhanced[key])]
            self.assertEqual(changed, [])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for AMP')
    def test_cuda_amp_for_all_four_models(self):
        for path, channels, prefix in (
                (PRES_PAF, [128, 256, 512], 'encoder.paf'),
                (HR_PAF, [36, 72, 144], 'encoder.paf'),
                (PRES_BOR, [128, 256, 512], 'encoder.bor.'),
                (HR_BOR, [36, 72, 144], 'encoder.bor.')):
            model = build(path).model.cuda().train()
            model.encoder.eval_spatial_size = None
            features = [
                torch.randn(1, channels[0], 16, 16, device='cuda'),
                torch.randn(1, channels[1], 8, 8, device='cuda'),
                torch.randn(1, channels[2], 4, 4, device='cuda'),
            ]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.encoder(features)
                loss = sum(value.float().square().mean() for value in outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for name, parameter in model.named_parameters()
                if name.startswith(prefix)))


if __name__ == '__main__':
    unittest.main()
