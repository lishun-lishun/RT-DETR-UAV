"""PResNet18 Baseline/HSDR/PHSB structure, compatibility and AMP tests.

    python -m unittest discover -s tests -p test_backbone_variants.py -v
"""

import copy
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_variant_test_audit', ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
core = audit.import_model_source(selective=True)

from src.nn.backbone.presnet import BasicBlock, PResNet
from src.nn.backbone.phsb import PHSBBranch


CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'hsdr_b': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hsdr_b.yml',
    'hsdr_a': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hsdr_a.yml',
    'phsb': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_phsb.yml',
}


def build(name, seed=0):
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(CONFIGS[name]), PResNet={'pretrained': False})
    model = config.model.eval()
    model.multi_scale = None
    return model


def old_presnet_class():
    source = subprocess.check_output(
        ['git', 'show', 'HEAD:rtdetr_pytorch/src/nn/backbone/presnet.py'],
        cwd=ROOT, text=True, encoding='utf-8')
    source = source.replace('from src.core import register', 'register = lambda cls: cls')
    namespace = {'__name__': 'src.nn.backbone._pre_backbone_variant_reference',
                 '__package__': 'src.nn.backbone'}
    exec(compile(source, '<readonly pre-HSDR/PHSB PResNet>', 'exec'), namespace)
    return namespace['PResNet']


def assert_nested_equal(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_nested_equal(test, first, second)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test, left[key], right[key])


class BackboneVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_fairness_and_no_existing_method(self):
        from importlib.util import module_from_spec, spec_from_file_location
        script = ROOT / 'tools/benchmark_backbone_variants.py'
        module_spec = spec_from_file_location('_variant_benchmark', script)
        benchmark = module_from_spec(module_spec)
        module_spec.loader.exec_module(benchmark)
        report = benchmark.audit_configs()
        self.assertTrue(report['passed'], report['failures'])
        for name, entry in report['configs'].items():
            self.assertFalse([key for key in entry['differences_from_baseline']
                              if key != 'output_dir'
                              and not key.startswith(('BackboneVariant.', 'PHSB.'))], name)

    def test_real_stage_counts_shapes_channels_and_strides(self):
        expected_counts = {'baseline': [2, 2, 2, 2], 'hsdr_b': [2, 3, 2, 1],
                           'hsdr_a': [2, 4, 3, 1], 'phsb': [2, 2, 2, 2]}
        image = torch.randn(1, 3, 640, 640)
        for name in CONFIGS:
            model = build(name)
            backbone = model.backbone
            self.assertEqual([len(stage.blocks) for stage in backbone.res_layers],
                             expected_counts[name])
            records, hooks = [], []
            for idx, stage in enumerate(backbone.res_layers):
                hooks.append(stage.register_forward_hook(
                    lambda module, inputs, output, i=idx:
                    records.append((i, tuple(output.shape)))))
            with torch.no_grad():
                levels = backbone(image)
            for hook in hooks:
                hook.remove()
            self.assertEqual(records,
                             [(0, (1, 64, 160, 160)),
                              (1, (1, 128, 80, 80)),
                              (2, (1, 256, 40, 40)),
                              (3, (1, 512, 20, 20))])
            self.assertEqual([tuple(level.shape) for level in levels],
                             [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)])
            self.assertEqual(backbone.out_channels, [128, 256, 512])
            self.assertEqual(backbone.out_strides, [8, 16, 32])

    def test_baseline_matches_readonly_prechange_full_detector(self):
        model = build('baseline', seed=11)
        reference = copy.deepcopy(model)
        original = old_presnet_class()(
            18, variant='d', num_stages=4, return_idx=[1, 2, 3],
            freeze_at=-1, freeze_norm=False, pretrained=False).eval()
        original.load_state_dict(model.backbone.state_dict(), strict=True)
        reference.backbone = original
        image = torch.randn(1, 3, 640, 640)
        captured = {'new_encoder': [], 'old_encoder': [],
                    'new_decoder': [], 'old_decoder': []}
        hooks = [
            model.encoder.register_forward_hook(
                lambda module, inputs, output: captured['new_encoder'].append(output)),
            reference.encoder.register_forward_hook(
                lambda module, inputs, output: captured['old_encoder'].append(output)),
            model.decoder.register_forward_hook(
                lambda module, inputs, output: captured['new_decoder'].append(output)),
            reference.decoder.register_forward_hook(
                lambda module, inputs, output: captured['old_decoder'].append(output)),
        ]
        with torch.no_grad():
            current_features, old_features = model.backbone(image), reference.backbone(image)
            current, old = model(image), reference(image)
        for hook in hooks:
            hook.remove()
        assert_nested_equal(self, current_features, old_features)
        assert_nested_equal(self, captured['new_encoder'], captured['old_encoder'])
        assert_nested_equal(self, captured['new_decoder'], captured['old_decoder'])
        assert_nested_equal(self, current, old)

    def test_phsb_zero_alpha_matches_baseline_full_detector(self):
        baseline, phsb = build('baseline', seed=17), build('phsb', seed=17)
        self.assertEqual(phsb.backbone.phsb.hidden_channels, 96)
        self.assertEqual(len(phsb.backbone.phsb.hr_blocks), 3)
        self.assertEqual(phsb.backbone.phsb.alpha3_effective.item(), 0)
        self.assertEqual(phsb.backbone.phsb.alpha4_effective.item(), 0)
        image = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            base_levels, phsb_levels = baseline.backbone(image), phsb.backbone(image)
            base_output, phsb_output = baseline(image), phsb(image)
        assert_nested_equal(self, base_levels, phsb_levels)
        assert_nested_equal(self, base_output, phsb_output)

    def test_phsb_main_stage4_reads_base_f3_and_stage5_reads_enhanced_f4(self):
        backbone = build('phsb', seed=23).backbone.eval()
        with torch.no_grad():
            backbone.phsb.raw_alpha3.fill_(.3)
            backbone.phsb.raw_alpha4.fill_(.4)
        image = torch.randn(1, 3, 128, 128)
        stage4_inputs, stage5_inputs = [], []
        hooks = [
            backbone.res_layers[2].register_forward_pre_hook(
                lambda module, inputs: stage4_inputs.append(inputs[0])),
            backbone.res_layers[3].register_forward_pre_hook(
                lambda module, inputs: stage5_inputs.append(inputs[0])),
        ]
        with torch.no_grad():
            stem = F.max_pool2d(backbone.conv1(image), 3, 2, 1)
            f2 = backbone.res_layers[0](stem)
            f3_base = backbone.res_layers[1](f2)
            f4_base = backbone.res_layers[2](f3_base)
            p3_residual, p4_residual = backbone.phsb(f3_base, f4_base)
            f3, f4 = f3_base + p3_residual, f4_base + p4_residual
            f5 = backbone.res_layers[3](f4)
            levels = backbone(image)
        for hook in hooks:
            hook.remove()
        assert_nested_equal(self, levels, [f3, f4, f5])
        self.assertGreater((f3 - f3_base).abs().max().item(), 0)
        self.assertGreater((f4 - f4_base).abs().max().item(), 0)
        assert_nested_equal(self, stage4_inputs[-1], f3_base)
        assert_nested_equal(self, stage5_inputs[-1], f4)

    def test_original_keys_and_rng_preserved(self):
        torch.manual_seed(31)
        baseline = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False)
        rng_after_baseline = torch.random.get_rng_state()
        original = baseline.state_dict()
        for variant, extra in (({'type': 'hsdr', 'stage_blocks': [2, 3, 2, 1]},
                                ('res_layers.1.blocks.2.',)),
                               ({'type': 'hsdr', 'stage_blocks': [2, 4, 3, 1]},
                                ('res_layers.1.blocks.2.', 'res_layers.1.blocks.3.',
                                 'res_layers.2.blocks.2.')),
                               ({'type': 'phsb'}, ('phsb.',))):
            torch.manual_seed(31)
            candidate = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                                BackboneVariant=variant)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_after_baseline))
            current = candidate.state_dict()
            for key in set(original) & set(current):
                self.assertTrue(torch.equal(original[key], current[key]), key)
            added = set(current) - set(original)
            self.assertTrue(added)
            self.assertTrue(all(key.startswith(extra) for key in added))
            removed = set(original) - set(current)
            if variant['type'] == 'hsdr':
                self.assertTrue(all(key.startswith('res_layers.3.blocks.1.')
                                    for key in removed))
            else:
                self.assertFalse(removed)

    def test_debug_statistics_are_opt_in_and_variant_selection_is_guarded(self):
        image = torch.randn(1, 3, 128, 128)
        with patch('builtins.print') as log:
            baseline = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                               BackboneVariant={'type': 'baseline', 'debug': True})
            baseline.eval()(image)
        messages = ' '.join(str(call.args[0]) for call in log.call_args_list)
        for name in ('P3', 'P4', 'P5', 'spatial_var', 'channel_var'):
            self.assertIn(name, messages)
        with patch('builtins.print') as log:
            phsb = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                           BackboneVariant={'type': 'phsb'},
                           PHSB={'debug': True, 'debug_interval': 1})
            phsb.eval()(image)
        messages = ' '.join(str(call.args[0]) for call in log.call_args_list)
        for name in ('alpha3_effective', 'alpha4_effective', 'E3/P3',
                     'E4/P4', 'HR_mean', 'HR_std'):
            self.assertIn(name, messages)
        with self.assertRaisesRegex(ValueError, 'cannot mix existing backbone plugins'):
            PResNet(18, BackboneVariant={'type': 'phsb'}, SECD={'enabled': True})
        with self.assertRaisesRegex(ValueError, 'stage_blocks'):
            PResNet(18, BackboneVariant={'type': 'hsdr',
                                        'stage_blocks': [2, 2, 2, 2]})

    def test_pretrained_mapping_and_invalid_missing_key_rejected(self):
        baseline = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False)
        source = baseline.state_dict()
        variants = (
            ({'type': 'baseline'}, 0, 0, ()),
            ({'type': 'hsdr', 'stage_blocks': [2, 3, 2, 1]}, 10, 12,
             ('res_layers.1.blocks.2.',)),
            ({'type': 'hsdr', 'stage_blocks': [2, 4, 3, 1]}, 30, 12,
             ('res_layers.1.blocks.2.', 'res_layers.1.blocks.3.',
              'res_layers.2.blocks.2.')),
            ({'type': 'phsb'}, 47, 0, ('phsb.',)),
        )
        with patch('torch.hub.load_state_dict_from_url', return_value=source):
            for variant, missing_count, unexpected_count, allowed in variants:
                loaded = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                                 pretrained=True, BackboneVariant=variant)
                report = loaded.pretrained_load_report
                self.assertEqual(len(report['missing_keys']), missing_count)
                self.assertEqual(len(report['unexpected_keys']), unexpected_count)
                self.assertTrue(all(key.startswith(allowed)
                                    for key in report['missing_keys']))
                self.assertTrue(all(key.startswith('res_layers.3.blocks.1.')
                                    for key in report['unexpected_keys']))
                for key in set(source) & set(loaded.state_dict()):
                    self.assertTrue(torch.equal(loaded.state_dict()[key], source[key]), key)
        damaged = dict(source)
        damaged.pop('conv1.conv1_1.conv.weight')
        with patch('torch.hub.load_state_dict_from_url', return_value=damaged):
            with self.assertRaisesRegex(RuntimeError, 'pretrained backbone keys mismatch'):
                PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                        pretrained=True,
                        BackboneVariant={'type': 'hsdr', 'stage_blocks': [2, 3, 2, 1]})

    def test_backward_finite_and_new_hsdr_blocks_trainable(self):
        image = torch.randn(2, 3, 128, 128)
        for name in ('hsdr_b', 'hsdr_a', 'phsb'):
            backbone = build(name, seed=37).backbone.train()
            if name == 'phsb':
                with torch.no_grad():
                    backbone.phsb.raw_alpha3.fill_(.01)
                    backbone.phsb.raw_alpha4.fill_(.01)
            output = backbone(image)
            loss = sum(level.float().square().mean() for level in output)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            new_prefixes = ('res_layers.1.blocks.2.', 'res_layers.2.blocks.2.') \
                if name.startswith('hsdr') else ('phsb.',)
            gradients = [p.grad for key, p in backbone.named_parameters()
                         if key.startswith(new_prefixes)]
            self.assertTrue(gradients)
            self.assertTrue(all(grad is not None and torch.isfinite(grad).all()
                                for grad in gradients), name)
            self.assertTrue(any(grad.abs().sum() > 0 for grad in gradients))

    def test_phsb_three_step_gradient_activation(self):
        torch.manual_seed(47)
        branch = PHSBBranch(8, 16, BasicBlock, branch_ratio=.75,
                            num_blocks=3, alpha_init=0.)
        optimizer = torch.optim.SGD(branch.parameters(), lr=.1)
        probe3 = torch.randn(2, 8, 12, 12)
        probe4 = torch.randn(2, 16, 6, 6)
        records = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(2, 8, 12, 12, requires_grad=True)
            base4 = torch.cat((F.avg_pool2d(x, 2, 2),
                               F.avg_pool2d(x, 2, 2)), dim=1)
            residual3, residual4 = branch(x, base4)
            loss = ((x + residual3) * probe3).mean() + \
                   ((base4 + residual4) * probe4).mean()
            loss.backward()
            def norm(parameter):
                return 0. if parameter.grad is None else parameter.grad.norm().item()
            record = {
                'alpha3': norm(branch.raw_alpha3), 'alpha4': norm(branch.raw_alpha4),
                'reduction': norm(branch.reduction.conv.weight),
                'hr_block': norm(branch.hr_blocks[0].branch2a.conv.weight),
                'p3_projection': norm(branch.p3_projection.conv.weight),
                'p4_projection': norm(branch.p4_projection.conv.weight),
                'F3': x.grad.norm().item(),
            }
            records.append(record)
            optimizer.step()
        print('PHSB step_grad_norms=', records)
        self.assertGreater(records[0]['alpha3'], 0)
        self.assertGreater(records[0]['alpha4'], 0)
        self.assertEqual(records[0]['reduction'], 0)
        for name in ('reduction', 'hr_block', 'p3_projection', 'p4_projection'):
            self.assertGreater(records[1][name], 0, name)
        for record in records:
            self.assertGreater(record['F3'], 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_640_amp_forward_backward_finite(self):
        for name in CONFIGS:
            backbone = build(name, seed=41).backbone.cuda().train()
            if name == 'phsb':
                with torch.no_grad():
                    backbone.phsb.raw_alpha3.fill_(.01)
                    backbone.phsb.raw_alpha4.fill_(.01)
            image = torch.randn(1, 3, 640, 640, device='cuda', requires_grad=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = backbone(image)
                loss = sum(value.float().square().mean() for value in outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), name)
            self.assertTrue(torch.isfinite(image.grad).all(), name)
            for parameter in backbone.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            del backbone, image, outputs, loss
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
