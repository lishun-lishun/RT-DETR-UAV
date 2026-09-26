"""Unit and integration acceptance tests for Persistent Detail Relay."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_pdr_audit', ROOT / 'tools' / 'analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
core = audit.import_model_source(selective=True)

from src.nn.backbone.backbone_modules.pdr import (  # noqa: E402
    DetailRelay, PersistentDetailRelay)
from src.nn.backbone.presnet import PResNet  # noqa: E402


CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'pdr3': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr3.yml',
    'pdr34': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr34.yml',
    'pdr34_nogate': (ROOT / 'configs/rtdetr/'
                     'rtdetr_r18vd_dut_anti_uav_pdr34_nogate.yml'),
}


def build(name, seed=0):
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(CONFIGS[name]), PResNet={'pretrained': False})
    model = config.model.eval()
    model.multi_scale = None
    return model


class PDRUnitTests(unittest.TestCase):
    def test_relay_uses_exact_space_to_depth_and_rejects_odd_shapes(self):
        relay = DetailRelay(5, 7).eval()
        captured = []
        handle = relay.reduce.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        with torch.no_grad():
            result = relay(torch.randn(2, 5, 16, 18))
        handle.remove()
        self.assertEqual(tuple(captured[0].shape), (2, 20, 8, 9))
        self.assertEqual(tuple(result.shape), (2, 7, 8, 9))
        with self.assertRaisesRegex(ValueError, 'even'):
            relay(torch.randn(1, 5, 15, 18))

    def test_gate_bounds_alpha_initialization_and_debug_names(self):
        module = PersistentDetailRelay(
            [64, 128, 256], debug=True,
            gate={'rho': 0.25, 'tau': 0.2, 'learnable_theta': True,
                  'theta_init': 0.0, 'eps': 1e-6},
            fusion={'alpha_max': 0.5, 'alpha_init': 0.1}).eval()
        c2 = torch.randn(2, 64, 32, 32)
        with torch.no_grad():
            _, d3, d4 = module.make_details(c2)
            c3 = module.inject3(torch.randn(2, 128, 16, 16), d3)
            module.inject4(torch.randn(2, 256, 8, 8), d4)
        self.assertEqual(tuple(c3.shape), (2, 128, 16, 16))
        for injection in (module.injection3, module.injection4):
            self.assertAlmostEqual(injection.alpha.item(), 0.1, places=6)
            gate = injection.last_debug_tensors['gate']
            self.assertGreaterEqual(gate.min().item(), 0.25)
            self.assertLessEqual(gate.max().item(), 1.0)
        expected = {'raw_alpha3', 'alpha3_eff', 'theta3', 'gate3_mean',
                    'raw_alpha4', 'alpha4_eff', 'theta4', 'gate4_mean',
                    'detail3_norm', 'main3_norm', 'injection3_norm',
                    'detail4_norm', 'main4_norm', 'injection4_norm'}
        self.assertTrue(expected.issubset(module.last_debug_stats))
        self.assertTrue(all(torch.is_tensor(value)
                            for value in module.last_debug_stats.values()))

    def test_all_pdr34_core_parameters_receive_gradients(self):
        torch.manual_seed(4)
        module = PersistentDetailRelay([64, 128, 256]).train()
        c2 = torch.randn(2, 64, 32, 32, requires_grad=True)
        _, d3, d4 = module.make_details(c2)
        c3 = module.inject3(torch.randn(2, 128, 16, 16), d3)
        c4 = module.inject4(torch.randn(2, 256, 8, 8), d4)
        loss = ((c3 * torch.randn_like(c3)).mean()
                + (c4 * torch.randn_like(c4)).mean())
        loss.backward()
        required = {
            'detail_memory': module.detail_memory.projection.conv.weight,
            'relay3_reduce': module.relay3.reduce.conv.weight,
            'relay3_depthwise': module.relay3.dwconv.conv.weight,
            'projection3': module.injection3.projection.conv.weight,
            'raw_alpha3': module.injection3.raw_alpha,
            'theta3': module.injection3.theta,
            'relay4_reduce': module.relay4.reduce.conv.weight,
            'relay4_depthwise': module.relay4.dwconv.conv.weight,
            'projection4': module.injection4.projection.conv.weight,
            'raw_alpha4': module.injection4.raw_alpha,
            'theta4': module.injection4.theta,
        }
        for name, parameter in required.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.norm().item(), 0, name)
        self.assertGreater(c2.grad.norm().item(), 0)


class PDRIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disabled_is_exact_original_baseline(self):
        torch.manual_seed(9)
        baseline = PResNet(18, variant='d', return_idx=[1, 2, 3],
                           freeze_norm=False, pretrained=False).eval()
        torch.manual_seed(9)
        disabled = PResNet(
            18, variant='d', return_idx=[1, 2, 3], freeze_norm=False,
            pretrained=False, PDR={'enabled': False}).eval()
        self.assertEqual(list(baseline.state_dict()), list(disabled.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            expected, actual = baseline(image), disabled(image)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_configs_fair_shapes_and_no_mert_secd(self):
        baseline = audit.fresh_config(CONFIGS['baseline'])
        image = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            baseline_levels = build('baseline', seed=11).backbone(image)
        modes = {
            'pdr3': (False, True),
            'pdr34': (True, True),
            'pdr34_nogate': (True, False),
        }
        for name, (relay4, gate) in modes.items():
            config = audit.fresh_config(CONFIGS[name])
            diff = audit.differences(baseline, config)
            self.assertTrue(all(key in ('__include__', 'output_dir')
                                or key == 'PDR' or key.startswith('PDR.')
                                for key in diff), diff)
            self.assertFalse(config['MERT']['enabled'])
            self.assertFalse(config['SECD']['enabled'])
            self.assertEqual(config['epoches'], baseline['epoches'])
            self.assertEqual(config['optimizer'], baseline['optimizer'])
            self.assertEqual(config['lr_scheduler'], baseline['lr_scheduler'])
            backbone = build(name, seed=11).backbone
            self.assertEqual(backbone.pdr.use_relay4, relay4)
            self.assertEqual(backbone.pdr.use_semantic_gate, gate)
            self.assertEqual([len(stage.blocks) for stage in backbone.res_layers],
                             [2, 2, 2, 2])
            with torch.no_grad():
                levels = backbone(image)
            for level, reference in zip(levels, baseline_levels):
                self.assertEqual(level.shape, reference.shape)
                self.assertEqual(level.dtype, reference.dtype)
                self.assertEqual(level.device, reference.device)
            self.assertEqual([tuple(level.shape) for level in levels],
                             [(1, 128, 80, 80), (1, 256, 40, 40),
                              (1, 512, 20, 20)])

        build('pdr34', seed=12)
        clean = build('baseline', seed=12).backbone
        self.assertIsNone(clean.pdr)
        self.assertFalse(clean.pdr_enabled)

    def test_persistent_chain_and_stage5_consumes_enhanced_c4(self):
        backbone = build('pdr34', seed=13).backbone
        seen = {}
        def capture_input(name):
            def hook(module, inputs):
                seen[name] = inputs[0].detach()
            return hook

        def capture_output(name):
            def hook(module, inputs, output):
                seen[name] = output.detach()
            return hook

        hooks = [
            backbone.pdr.relay4.register_forward_pre_hook(
                capture_input('relay4_input')),
            backbone.pdr.relay3.register_forward_hook(
                capture_output('relay3_output')),
            backbone.pdr.injection3.register_forward_hook(
                capture_output('enhanced_c3')),
            backbone.res_layers[2].register_forward_pre_hook(
                capture_input('stage4_input')),
            backbone.pdr.injection4.register_forward_hook(
                capture_output('enhanced_c4')),
            backbone.res_layers[3].register_forward_pre_hook(
                capture_input('stage5_input')),
        ]
        with torch.no_grad():
            backbone(torch.randn(1, 3, 128, 128))
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(seen['relay4_input'], seen['relay3_output'])
        torch.testing.assert_close(seen['stage4_input'], seen['enhanced_c3'])
        torch.testing.assert_close(seen['stage5_input'], seen['enhanced_c4'])

    def test_pretrained_original_keys_are_preserved(self):
        source_model = PResNet(18, variant='d', return_idx=[1, 2, 3],
                               freeze_norm=False, pretrained=False)
        source = source_model.state_dict()
        options = {
            'enabled': True, 'use_relay3': True, 'use_relay4': True,
            'use_semantic_gate': True,
            'detail_channels': {'c2': 32, 'c3': 64, 'c4': 96},
        }
        with patch('torch.hub.load_state_dict_from_url', return_value=source):
            candidate = PResNet(
                18, variant='d', return_idx=[1, 2, 3], freeze_norm=False,
                pretrained=True, PDR=options)
        report = candidate.pretrained_load_report
        self.assertFalse(report['unexpected_keys'])
        self.assertTrue(report['missing_keys'])
        self.assertTrue(all(key.startswith('pdr.') for key in report['missing_keys']))
        for key, tensor in source.items():
            torch.testing.assert_close(candidate.state_dict()[key], tensor, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_amp_forward_backward_is_finite_for_all_modes(self):
        for name in ('pdr3', 'pdr34', 'pdr34_nogate'):
            model = build(name, seed=17).backbone.cuda().train()
            image = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                levels = model(image)
                loss = sum(level.float().square().mean() for level in levels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), name)
            self.assertTrue(torch.isfinite(image.grad).all(), name)
            for parameter in model.pdr.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            del model, image, levels, loss
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
