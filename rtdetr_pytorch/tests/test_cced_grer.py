"""CCED/GRER math, integration, compatibility and AMP regressions.

Run with:
    python -m unittest tests.test_cced_grer -v
"""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_cced_grer_audit',
                                              ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
core = audit.import_model_source(selective=True)

from src.nn.backbone.presnet import PResNet
from src.nn.backbone.backbone_plugins.cced import CCEDTransition
from src.nn.backbone.backbone_plugins.grer import GRERRelay


CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'cced': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34.yml',
    'grer': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_grer34.yml',
    'combined': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_cced34_grer34.yml',
}


def build(name, seed=0):
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(CONFIGS[name]), PResNet={'pretrained': False})
    config.model.multi_scale = None
    return config.model, config.yaml_cfg


def old_presnet_class():
    source = subprocess.check_output(
        ['git', 'show', 'HEAD:rtdetr_pytorch/src/nn/backbone/presnet.py'],
        cwd=ROOT, text=True, encoding='utf-8')
    source = source.replace('from src.core import register', 'register = lambda cls: cls')
    namespace = {'__name__': 'src.nn.backbone._pre_cced_grer_reference',
                 '__package__': 'src.nn.backbone'}
    exec(compile(source, '<readonly pre-CCED/GRER PResNet>', 'exec'), namespace)
    return namespace['PResNet']


def assemble_phases(phases):
    batch, channels, height, width = phases[0].shape
    result = phases[0].new_empty(batch, channels, height * 2, width * 2)
    result[:, :, 0::2, 0::2] = phases[0]
    result[:, :, 0::2, 1::2] = phases[1]
    result[:, :, 1::2, 0::2] = phases[2]
    result[:, :, 1::2, 1::2] = phases[3]
    return result


def assert_nested_close(test, left, right, atol=0, rtol=0):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_close(test, left[key], right[key], atol, rtol)
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_close(test, a, b, atol, rtol)


class CCEDMathTests(unittest.TestCase):
    def test_shapes_group_fallback_normalization_and_odd_input(self):
        module = CCEDTransition(12, 20, groups=8, fusion={'alpha_init': .01})
        self.assertEqual(module.groups, 4)
        x = torch.randn(2, 12, 15, 17)
        dense, stats = module.consensus_evidence(x, return_stats=True)
        self.assertEqual(dense.shape, (2, 12, 8, 9))
        self.assertEqual(stats['phases'].shape, (2, 12, 4, 8, 9))
        self.assertEqual(stats['grouped_residual'].shape, (2, 4, 3, 4, 8, 9))
        self.assertEqual(stats['q'].shape, (2, 4, 4, 8, 9))
        self.assertEqual(stats['consensus'].shape, (2, 4, 8, 9))
        torch.testing.assert_close(stats['q'].sum(2),
                                   torch.ones_like(stats['q'].sum(2)))
        torch.testing.assert_close(stats['consensus'].sum(1),
                                   torch.ones_like(stats['consensus'].sum(1)))
        self.assertEqual(module(x).shape, (2, 20, 8, 9))

    def test_identical_phases_have_zero_evidence_without_softmax(self):
        module = CCEDTransition(8, 16, groups=4).eval()
        phase = torch.randn(2, 8, 5, 7)
        x = assemble_phases([phase, phase, phase, phase])
        with patch('torch.nn.functional.softmax', side_effect=AssertionError('forbidden')):
            dense = module.consensus_evidence(x)
        torch.testing.assert_close(dense, torch.zeros_like(dense), atol=0, rtol=0)

    def test_cross_group_consensus_suppresses_single_group_support(self):
        module = CCEDTransition(8, 16, groups=4)
        zeros = [torch.zeros(1, 8, 3, 3) for _ in range(4)]
        common = [value.clone() for value in zeros]
        common[0].fill_(4)
        _, common_stats = module.consensus_evidence(assemble_phases(common), True)

        disagreement = [value.clone() for value in zeros]
        for group, phase in enumerate(range(4)):
            disagreement[phase][:, group * 2:(group + 1) * 2].fill_(4)
        _, disagreement_stats = module.consensus_evidence(
            assemble_phases(disagreement), True)
        common_c0 = common_stats['consensus'][:, 0].mean()
        disagreement_c0 = disagreement_stats['consensus'][:, 0].mean()
        self.assertGreater(common_c0.item(), .45)
        self.assertGreater(common_c0.item(), disagreement_c0.item() + .15)
        self.assertEqual(common_stats['consensus'].mean((0, 2, 3)).argmax().item(), 0)

    def test_phase_permutation_has_no_absolute_phase_bias(self):
        module = CCEDTransition(8, 16, groups=4)
        phases = [torch.randn(2, 8, 4, 5) for _ in range(4)]
        _, original = module.consensus_evidence(assemble_phases(phases), True)
        permutation = [2, 0, 3, 1]
        _, permuted = module.consensus_evidence(
            assemble_phases([phases[index] for index in permutation]), True)
        torch.testing.assert_close(permuted['consensus'],
                                   original['consensus'][:, permutation])
        torch.testing.assert_close(permuted['q'], original['q'][:, :, permutation])


class GRERMathTests(unittest.TestCase):
    def test_flat_map_mad_safe_and_finite(self):
        module = GRERRelay(8, 16, groups=4)
        _, stats = module.rarity_features(torch.ones(2, 8, 11, 13), True)
        torch.testing.assert_close(stats['local_score'],
                                   torch.full_like(stats['local_score'], module.eps ** .5))
        torch.testing.assert_close(stats['mad'], torch.zeros_like(stats['mad']))
        self.assertTrue(torch.isfinite(stats['z']).all())
        self.assertTrue(torch.isfinite(stats['gate']).all())
        self.assertTrue(((stats['gate'] >= 0) & (stats['gate'] <= 1)).all())

    def test_sparse_location_is_rarer_than_repeated_texture(self):
        module = GRERRelay(8, 16, groups=4)
        sparse = torch.zeros(1, 8, 17, 17)
        sparse[:, :, 8, 8] = 10
        _, sparse_stats = module.rarity_features(sparse, True)
        repeated = torch.zeros_like(sparse)
        repeated[:, :, 0::2, 0::2] = 10
        _, repeated_stats = module.rarity_features(repeated, True)
        sparse_gate = sparse_stats['gate'][0, 0, 8, 8]
        repeated_gate = repeated_stats['gate'][0, 0, 8, 8]
        self.assertGreater(sparse_stats['z'][0, 0, 8, 8].item(), 2)
        self.assertGreater(sparse_gate.item(), .99)
        self.assertGreater(sparse_gate.item(), repeated_gate.item() + .4)

    def test_per_image_median_mad_and_relay_uses_x_not_residual(self):
        module = GRERRelay(8, 16, groups=4)
        first = torch.randn(1, 8, 9, 11)
        second = 4 * torch.randn(1, 8, 9, 11) + 7
        batch = torch.cat((first, second))
        rare, stats = module.rarity_features(batch, True)
        for index, sample in enumerate((first, second)):
            single_rare, single = module.rarity_features(sample, True)
            torch.testing.assert_close(stats['median'][index:index + 1], single['median'])
            torch.testing.assert_close(stats['mad'][index:index + 1], single['mad'])
            torch.testing.assert_close(rare[index:index + 1], single_rare)
        expected = stats['gate'] * batch
        if expected.shape[-2] % 2 or expected.shape[-1] % 2:
            expected = F.pad(expected, (0, expected.shape[-1] % 2,
                                        0, expected.shape[-2] % 2), mode='replicate')
        expected = F.avg_pool2d(expected, 2, 2)
        torch.testing.assert_close(rare, expected)

    def test_median_mad_graph_is_differentiable_and_finite(self):
        module = GRERRelay(8, 16, groups=4)
        x = torch.randn(2, 8, 9, 11, requires_grad=True)
        rare, stats = module.rarity_features(x, True)
        loss = (rare.square().mean() + stats['gate'].mean()
                + stats['median'].mean() + stats['mad'].mean()
                + stats['local_score'].mean())
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_configs_only_change_output_cced_grer_and_disable_other_methods(self):
        baseline = audit.fresh_config(CONFIGS['baseline'])
        expected = {
            'baseline': (False, False), 'cced': (True, False),
            'grer': (False, True), 'combined': (True, True),
        }
        for name, switches in expected.items():
            config = audit.fresh_config(CONFIGS[name])
            difference = audit.differences(
                {k: v for k, v in baseline.items() if k != '__include__'},
                {k: v for k, v in config.items() if k != '__include__'})
            forbidden = [key for key in difference if key != 'output_dir'
                         and key not in ('CCED', 'GRER')
                         and not key.startswith(('CCED.', 'GRER.'))]
            self.assertFalse(forbidden, (name, forbidden))
            self.assertEqual((config['CCED']['enabled'], config['GRER']['enabled']), switches)
            self.assertFalse(config['MERT']['enabled'])
            self.assertFalse(config['SECD']['enabled'])

    def test_real_640_stage_shapes_and_return_protocol(self):
        model, _ = build('combined')
        backbone = model.backbone.eval()
        records, hooks = [], []
        for index, stage in enumerate(backbone.res_layers):
            hooks.append(stage.register_forward_hook(
                lambda module, inputs, output, i=index:
                records.append((f'S{i + 2}', tuple(output.shape)))))
        with torch.no_grad():
            outputs = backbone(torch.randn(1, 3, 640, 640))
        for hook in hooks:
            hook.remove()
        self.assertEqual(records, [('S2', (1, 64, 160, 160)),
                                   ('S3', (1, 128, 80, 80)),
                                   ('S4', (1, 256, 40, 40)),
                                   ('S5', (1, 512, 20, 20))])
        self.assertEqual([tuple(value.shape) for value in outputs],
                         [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)])
        self.assertEqual(backbone.out_channels, [128, 256, 512])
        self.assertEqual(backbone.out_strides, [8, 16, 32])

    def test_disabled_mode_matches_readonly_prechange_backbone_encoder_decoder(self):
        model, _ = build('baseline', seed=11)
        reference = copy.deepcopy(model)
        original = old_presnet_class()(
            18, variant='d', num_stages=4, return_idx=[1, 2, 3],
            freeze_at=-1, freeze_norm=False, pretrained=False).eval()
        original.load_state_dict(model.backbone.state_dict(), strict=True)
        reference.backbone = original
        model.eval()
        reference.eval()
        captures = {'new_encoder': [], 'old_encoder': [],
                    'new_decoder': [], 'old_decoder': []}
        hooks = [
            model.encoder.register_forward_hook(
                lambda m, i, o: captures['new_encoder'].append(o)),
            reference.encoder.register_forward_hook(
                lambda m, i, o: captures['old_encoder'].append(o)),
            model.decoder.register_forward_hook(
                lambda m, i, o: captures['new_decoder'].append(o)),
            reference.decoder.register_forward_hook(
                lambda m, i, o: captures['old_decoder'].append(o)),
        ]
        image = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            new_features = model.backbone(image)
            old_features = reference.backbone(image)
            new_output = model(image)
            old_output = reference(image)
        for hook in hooks:
            hook.remove()
        assert_nested_close(self, new_features, old_features)
        assert_nested_close(self, captures['new_encoder'], captures['old_encoder'])
        assert_nested_close(self, captures['new_decoder'], captures['old_decoder'])
        assert_nested_close(self, new_output, old_output)

    def test_zero_alpha_all_modes_match_baseline_exactly(self):
        baseline, _ = build('baseline', seed=17)
        baseline.eval()
        image = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            base_features, base_output = baseline.backbone(image), baseline(image)
        for name in ('cced', 'grer', 'combined'):
            candidate, _ = build(name, seed=17)
            candidate.eval()
            with torch.no_grad():
                features, output = candidate.backbone(image), candidate(image)
            assert_nested_close(self, features, base_features)
            assert_nested_close(self, output, base_output)

    def test_parallel_formula_and_stage5_consumes_enhanced_f4(self):
        model, _ = build('combined', seed=23)
        backbone = model.backbone.eval()
        with torch.no_grad():
            backbone.cced_34.raw_alpha.fill_(.3)
            backbone.grer_34.raw_alpha.fill_(-.2)
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            x = F.max_pool2d(backbone.conv1(image), 3, 2, 1)
            f2 = backbone.res_layers[0](x)
            f3 = backbone.res_layers[1](f2)
            f4_base = backbone.res_layers[2](f3)
            f4 = (f4_base + backbone.cced_34(f3, base=f4_base)
                  + backbone.grer_34(f3, base=f4_base))
            f5 = backbone.res_layers[3](f4)
            actual = backbone(image)
        torch.testing.assert_close(actual[0], f3)
        torch.testing.assert_close(actual[1], f4)
        torch.testing.assert_close(actual[2], f5)
        self.assertGreater((f4 - f4_base).abs().max().item(), 0)

    def test_state_dict_rng_and_pretrained_compatibility(self):
        torch.manual_seed(31)
        baseline = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False)
        after_baseline = torch.random.get_rng_state()
        torch.manual_seed(31)
        combined = PResNet(
            18, return_idx=[1, 2, 3], freeze_norm=False,
            CCED={'enabled': True}, GRER={'enabled': True})
        self.assertTrue(torch.equal(torch.random.get_rng_state(), after_baseline))
        original = baseline.state_dict()
        combined_state = combined.state_dict()
        for key, value in original.items():
            self.assertIn(key, combined_state)
            self.assertTrue(torch.equal(value, combined_state[key]), key)
        additions = set(combined_state) - set(original)
        self.assertTrue(additions)
        self.assertTrue(all(key.startswith(('cced_34.', 'grer_34.')) for key in additions))

        with patch('torch.hub.load_state_dict_from_url', return_value=original):
            loaded = PResNet(
                18, return_idx=[1, 2, 3], freeze_norm=False, pretrained=True,
                CCED={'enabled': True}, GRER={'enabled': True})
        for key, value in original.items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_three_step_alpha_and_projection_gradient_activation(self):
        for module in (CCEDTransition(8, 16, groups=4),
                       GRERRelay(8, 16, groups=4)):
            optimizer = torch.optim.SGD(module.parameters(), lr=.1)
            probe = torch.randn(2, 16, 6, 6)
            records = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                x = torch.randn(2, 8, 12, 12, requires_grad=True)
                base = torch.cat((F.avg_pool2d(x, 2, 2),
                                  F.avg_pool2d(x, 2, 2)), dim=1)
                loss = ((base + module(x, base=base)) * probe).mean()
                loss.backward()
                projection_grad = module.projection.conv.weight.grad.norm()
                records.append((module.raw_alpha.grad.abs().item(),
                                projection_grad.item(), x.grad.norm().item()))
                optimizer.step()
            print(type(module).__name__, 'step_grad_norms=', records)
            self.assertGreater(records[0][0], 0)
            self.assertEqual(records[0][1], 0)
            self.assertGreater(records[1][1], 0)
            self.assertTrue(all(value > 0 for record in records for value in (record[0], record[2])))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_amp_640_forward_backward_finite(self):
        for name in ('cced', 'grer', 'combined'):
            model, _ = build(name, seed=41)
            backbone = model.backbone.cuda().train()
            for module in (backbone.cced_34, backbone.grer_34):
                if module is not None:
                    with torch.no_grad():
                        module.raw_alpha.fill_(.01)
            image = torch.randn(1, 3, 640, 640, device='cuda', requires_grad=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = backbone(image)
                loss = sum(value.float().square().mean() for value in outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(image.grad).all())
            for parameter in backbone.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            del model, backbone, image, outputs, loss
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
