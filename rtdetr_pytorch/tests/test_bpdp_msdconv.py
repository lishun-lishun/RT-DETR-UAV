"""Synthetic and integration tests for BDPD and MSDConv."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_bpdp_msdconv_audit', ROOT / 'tools' / 'analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
core = audit.import_model_source(selective=True)

from src.nn.backbone.backbone_modules.bdpd import BDPDDownsample
from src.nn.backbone.backbone_modules.msdconv import MSDConv
from src.nn.backbone.presnet import PResNet


CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'bpdp': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml',
    'msdconv': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml',
    'combined': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp_msdconv.yml',
}


def build(name, seed=0):
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(CONFIGS[name]), PResNet={'pretrained': False})
    model = config.model.eval()
    model.multi_scale = None
    return model


def prior_presnet_class():
    """Load the committed pre-BDPD/MSDConv PResNet as a read-only reference."""
    source = subprocess.check_output(
        ['git', 'show', 'HEAD:rtdetr_pytorch/src/nn/backbone/presnet.py'],
        cwd=ROOT, text=True, encoding='utf-8')
    source = source.replace('from src.core import register', 'register = lambda cls: cls')
    namespace = {'__name__': 'src.nn.backbone._pre_bpdp_msd_reference',
                 '__package__': 'src.nn.backbone'}
    exec(compile(source, '<read-only prior PResNet>', 'exec'), namespace)
    return namespace['PResNet']


def assert_nested_equal(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_nested_equal(test, first, second)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    else:
        test.assertEqual(left, right)


def interleave_phases(phases):
    batch, count, channels, height, width = phases.shape
    assert count == 4
    output = phases.new_empty(batch, channels, 2 * height, 2 * width)
    output[:, :, 0::2, 0::2] = phases[:, 0]
    output[:, :, 0::2, 1::2] = phases[:, 1]
    output[:, :, 1::2, 0::2] = phases[:, 2]
    output[:, :, 1::2, 1::2] = phases[:, 3]
    return output


class BDPDTests(unittest.TestCase):
    def test_constant_impulse_permutation_blur_and_texture(self):
        module = BDPDDownsample(1, 4).eval()
        constant = torch.ones(1, 1, 15, 17)
        fields = module.phase_components(constant)
        self.assertEqual(tuple(fields['phases'].shape), (1, 4, 1, 8, 9))
        self.assertEqual(fields['padding'], (1, 1))
        self.assertLess(fields['detail'].abs().max().item(), 1e-7)
        self.assertLess(fields['weighted_detail'].abs().max().item(), 1e-7)

        impulse = torch.zeros(1, 1, 16, 16)
        impulse[:, :, 4, 6] = 1
        fields = module.phase_components(impulse)
        self.assertEqual(fields['phases'][0, 0, 0, 2, 3].item(), 1)
        self.assertEqual(fields['phases'][:, 1:].abs().max().item(), 0)
        self.assertGreater(fields['detail'].abs().sum().item(), 0)

        torch.manual_seed(3)
        phases = torch.randn(1, 4, 1, 8, 9)
        original = module.phase_components(interleave_phases(phases))
        permuted = module.phase_components(interleave_phases(phases[:, [2, 0, 3, 1]]))
        torch.testing.assert_close(original['base'], permuted['base'])
        torch.testing.assert_close(
            original['detail'].square().sum(dim=1),
            permuted['detail'].square().sum(dim=1))

        size = 33
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        sharp = torch.zeros(1, 1, size, size)
        sharp[:, :, size // 2, size // 2] = 1
        blurred = torch.exp(-((xx - size // 2).square()
                              + (yy - size // 2).square()).float() / 18)[None, None]
        sharp_fields = module.phase_components(sharp)
        blur_fields = module.phase_components(blurred)
        sharp_ratio = (sharp_fields['base'].norm()
                       / (sharp_fields['weighted_detail'].norm() + 1e-6))
        blur_ratio = (blur_fields['base'].norm()
                      / (blur_fields['weighted_detail'].norm() + 1e-6))
        self.assertGreater(blur_ratio.item(), sharp_ratio.item())

        # A two-pixel checker texture keeps alternating signs inside each
        # phase plane; its local signed mean should cancel rather than score 1.
        checker = (((xx // 2 + yy // 2) % 2).float() * 2 - 1)[None, None]
        consistency = module.phase_components(checker)['consistency']
        self.assertLess(consistency.mean().item(), 0.5)
        self.assertLess((consistency > 0.95).float().mean().item(), 0.1)

    def test_backward_all_core_branches(self):
        torch.manual_seed(5)
        module = BDPDDownsample(4, 8).train()
        sample = torch.randn(2, 4, 17, 19, requires_grad=True)
        probe = torch.randn(2, 8, 9, 10)
        loss = (module(sample) * probe).mean()
        loss.backward()
        gradients = {
            'base': module.base_projection.conv.weight.grad,
            'detail': module.detail_projection.conv.weight.grad,
            'alpha': module.raw_alpha.grad,
            'input': sample.grad,
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0, name)


class MSDConvTests(unittest.TestCase):
    def test_constant_sharp_blur_low_frequency_and_texture(self):
        module = MSDConv(8).eval()
        constant = torch.full((1, 8, 33, 35), 2.0)
        fields = module.scale_space(constant)
        self.assertLess(fields['H'].abs().max().item(), 1e-6)
        self.assertLess(fields['M'].abs().max().item(), 1e-6)
        torch.testing.assert_close(fields['L'], constant, rtol=0, atol=2e-6)
        torch.testing.assert_close(module(constant), constant, rtol=0, atol=2e-6)

        size = 33
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        sharp = torch.zeros(1, 8, size, size)
        sharp[:, :, size // 2, size // 2] = 1
        blurred = torch.exp(-((xx - size // 2).square()
                              + (yy - size // 2).square()).float() / 18)
        blurred = blurred[None, None].expand_as(sharp)
        sharp_fields = module.scale_space(sharp)
        blur_fields = module.scale_space(blurred)
        center = (0, slice(None), 0, size // 2, size // 2)
        self.assertGreater(sharp_fields['wH'][center].mean().item(),
                           sharp_fields['wM'][center].mean().item())
        self.assertGreater(blur_fields['wM'][center].mean().item(),
                           sharp_fields['wM'][center].mean().item())

        ramp = ((xx.float() + yy.float()) / (2 * size))[None, None].expand_as(sharp)
        ramp_fields = module.scale_space(ramp)
        self.assertGreater(ramp_fields['L'].norm().item(),
                           ramp_fields['H'].norm().item() + ramp_fields['M'].norm().item())

        checker = (((xx + yy) % 2).float() * 2 - 1)[None, None].expand_as(sharp)
        checker_fields = module.scale_space(checker)
        self.assertGreater(checker_fields['H'].norm().item(), 0)
        self.assertLess(checker_fields['eta'].mean().item(), 0.95)
        self.assertFalse(module.kernel3.requires_grad)
        self.assertFalse(module.kernel5.requires_grad)
        self.assertEqual(MSDConv(12, groups=8).groups, 6)

    def test_backward_all_core_branches(self):
        torch.manual_seed(7)
        module = MSDConv(8).train()
        sample = torch.randn(2, 8, 17, 19, requires_grad=True)
        probe = torch.randn_like(sample)
        loss = (module(sample) * probe).mean()
        loss.backward()
        gradients = {
            'router': module.context_router.weight.grad,
            'projection': module.projection.conv.weight.grad,
            'beta': module.raw_beta.grad,
            'input': sample.grad,
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.norm().item(), 0, name)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_fairness_structure_and_shapes(self):
        baseline = audit.fresh_config(CONFIGS['baseline'])
        expected = {'baseline': (False, False), 'bpdp': (True, False),
                    'msdconv': (False, True), 'combined': (True, True)}
        for name, switches in expected.items():
            config = audit.fresh_config(CONFIGS[name])
            diff = audit.differences(baseline, config)
            allowed = ('BackboneModification.', 'BDPD.', 'MSDConv.')
            self.assertTrue(all(key in ('__include__', 'output_dir')
                                or key.startswith(allowed) for key in diff), diff)
            self.assertEqual((config['BackboneModification']['bpdp'],
                              config['BackboneModification']['msdconv']), switches)
            backbone = build(name, seed=11).backbone
            self.assertEqual([len(stage.blocks) for stage in backbone.res_layers],
                             [2, 2, 2, 2])
            self.assertEqual(backbone.out_channels, [128, 256, 512])
            self.assertEqual(backbone.out_strides, [8, 16, 32])
            self.assertEqual(isinstance(
                backbone.res_layers[1].blocks[0].branch2a, BDPDDownsample), switches[0])
            self.assertEqual(isinstance(backbone.msdconv_p3, MSDConv), switches[1])
            with torch.no_grad():
                levels = backbone(torch.randn(1, 3, 640, 640))
            self.assertEqual([tuple(level.shape) for level in levels],
                             [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)])

        # Process-global YAML registry state must not leak candidate switches
        # into a subsequently constructed Baseline.
        build('combined', seed=12)
        clean = build('baseline', seed=12).backbone
        self.assertFalse(clean.bpdp_enabled)
        self.assertFalse(clean.msdconv_enabled)
        self.assertNotIsInstance(clean.res_layers[1].blocks[0].branch2a,
                                 BDPDDownsample)
        self.assertIsNone(clean.msdconv_p3)

    def test_baseline_equivalence_backbone_encoder_decoder(self):
        current = build('baseline', seed=13)
        reference = copy.deepcopy(current)
        prior = prior_presnet_class()(
            18, variant='d', num_stages=4, return_idx=[1, 2, 3],
            freeze_at=-1, freeze_norm=False, pretrained=False).eval()
        prior.load_state_dict(current.backbone.state_dict(), strict=True)
        reference.backbone = prior
        image = torch.randn(1, 3, 640, 640)
        captures = {'current_encoder': [], 'prior_encoder': [],
                    'current_decoder': [], 'prior_decoder': []}
        hooks = [
            current.encoder.register_forward_hook(
                lambda module, inputs, output: captures['current_encoder'].append(output)),
            reference.encoder.register_forward_hook(
                lambda module, inputs, output: captures['prior_encoder'].append(output)),
            current.decoder.register_forward_hook(
                lambda module, inputs, output: captures['current_decoder'].append(output)),
            reference.decoder.register_forward_hook(
                lambda module, inputs, output: captures['prior_decoder'].append(output)),
        ]
        with torch.no_grad():
            current_features = current.backbone(image)
            prior_features = reference.backbone(image)
            current_output = current(image)
            prior_output = reference(image)
        for hook in hooks:
            hook.remove()
        assert_nested_equal(self, current_features, prior_features)
        assert_nested_equal(self, captures['current_encoder'], captures['prior_encoder'])
        assert_nested_equal(self, captures['current_decoder'], captures['prior_decoder'])
        assert_nested_equal(self, current_output, prior_output)

    def test_pretrained_compatibility_is_explicit(self):
        source = PResNet(18, variant='d', return_idx=[1, 2, 3],
                         freeze_norm=False).state_dict()
        cases = {'baseline': (False, False), 'bpdp': (True, False),
                 'msdconv': (False, True), 'combined': (True, True)}
        with patch('torch.hub.load_state_dict_from_url', return_value=source):
            for name, (bpdp, msdconv) in cases.items():
                candidate = PResNet(
                    18, variant='d', return_idx=[1, 2, 3], freeze_norm=False,
                    pretrained=True,
                    BackboneModification={'bpdp': bpdp, 'msdconv': msdconv})
                report = candidate.pretrained_load_report
                if name == 'baseline':
                    self.assertEqual(report['missing_keys'], [])
                    self.assertEqual(report['unexpected_keys'], [])
                if bpdp:
                    self.assertTrue(candidate.replaced_pretrained_keys)
                    self.assertTrue(all(key.startswith(
                        'res_layers.1.blocks.0.branch2a.')
                        for key in candidate.replaced_pretrained_keys))
                for key in set(source).intersection(candidate.state_dict()):
                    self.assertTrue(torch.equal(source[key], candidate.state_dict()[key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_amp_no_nan_inf(self):
        for name in ('bpdp', 'msdconv', 'combined'):
            model = build(name, seed=17).backbone.cuda().train()
            image = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                levels = model(image)
                loss = sum(level.float().square().mean() for level in levels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), name)
            self.assertTrue(torch.isfinite(image.grad).all(), name)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            del model, image, levels, loss
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
