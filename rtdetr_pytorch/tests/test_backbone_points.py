"""Real backbone/detector smoke tests, without data, training or downloads.

python -m unittest discover -s tests -p test_backbone_points.py -v
UAV-DCNv4 uses torchvision's packaged deformable sampler and needs no custom
extension. CUDA AMP tests use the same autocast/GradScaler mechanism as det_engine; prediction-derived
smoke losses do NOT test the full COCO loss/matcher or detection accuracy.
"""

import ast
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch
import warnings

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_test_points_benchmark', ROOT / 'tools/benchmark_backbone_points.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
core = benchmark.audit.import_model_source(selective=True)
from src.nn.backbone.presnet import PResNet
from src.nn.backbone.plugin_points import build_plugins
from src.nn.backbone.plugin_points.dcnv4 import UAVDCNv4, load_backend


RUNNABLE = [m for m in benchmark.METHODS if m != 'Baseline']


def original_class():
    source = subprocess.check_output(['git', 'show', 'HEAD:rtdetr_pytorch/src/nn/backbone/presnet.py'],
                                     cwd=ROOT, text=True, encoding='utf-8')
    source = source.replace('from src.core import register', 'register = lambda cls: cls')
    namespace = {'__name__': 'src.nn.backbone._original_points_test', '__package__': 'src.nn.backbone'}
    exec(compile(source, '<readonly Git HEAD PResNet>', 'exec'), namespace)
    return namespace['PResNet']


class BackbonePointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_configs_only_change_points_and_output(self):
        audit = benchmark.resolved_audit()
        self.assertTrue(audit['passed'], audit['failures'])
        self.assertEqual(len(audit['methods']), 7)

    def test_off_matches_readonly_baseline_weights_keys_params_rng_outputs(self):
        torch.manual_seed(7)
        reference = original_class()(18, return_idx=[1, 2, 3], freeze_norm=False).eval()
        reference_rng = torch.random.get_rng_state()
        torch.manual_seed(7)
        model = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                       BackbonePlugins={p: {'enabled': False} for p in ('P0', 'P1', 'P2', 'P3', 'P4')}).eval()
        self.assertIsNone(model.plugins)
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in reference.parameters()))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), reference_rng))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
        image = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            left, right = reference(image), model(image)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_zero_init_640_shapes_original_weights_and_full_detector(self):
        image = torch.randn(1, 3, 640, 640)
        reference, _ = benchmark.build_model('Baseline', selective=True)
        reference.eval()
        with torch.no_grad():
            ref_features, ref_prediction = reference.backbone(image), reference(image)
        for method in RUNNABLE:
            with self.subTest(method=method):
                model, _ = benchmark.build_model(method, selective=True)
                model.eval()
                with torch.no_grad():
                    features, prediction = model.backbone(image), model(image)
                    ref_features, ref_prediction = reference.backbone(image), reference(image)
                self.assertEqual([list(f.shape) for f in features],
                                 [[1, 128, 80, 80], [1, 256, 40, 40], [1, 512, 20, 20]])
                for a, b in zip(features, ref_features):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for name in ('pred_boxes', 'pred_logits'):
                    torch.testing.assert_close(prediction[name], ref_prediction[name], rtol=0, atol=0)
                self.assertEqual(type(model.backbone.res_layers[0]).__name__, 'Blocks')
                for name, value in reference.state_dict().items():
                    self.assertIn(name, model.state_dict())
                    self.assertTrue(torch.equal(value.cpu(), model.state_dict()[name].cpu()), name)
                additions = set(model.state_dict()) - set(reference.state_dict())
                self.assertTrue(additions and all(k.startswith('backbone.plugins.') for k in additions))

    def test_alpha_can_start_learning_at_zero(self):
        for method in RUNNABLE:
            with self.subTest(method=method):
                model, _ = benchmark.build_model(method, selective=True)
                backbone = model.backbone.train()
                backbone.to('cpu')
                features = backbone(torch.randn(2, 3, 96, 96))
                loss = sum(t.square().mean() for t in features)
                loss.backward()
                for plugin in backbone.plugins.values():
                    self.assertIsNotNone(plugin.raw_alpha.grad)
                    self.assertTrue(torch.isfinite(plugin.raw_alpha.grad))
                    self.assertNotEqual(plugin.raw_alpha.grad.item(), 0)

    def test_nonzero_gate_plugin_and_original_gradients_finite(self):
        for method in RUNNABLE:
            with self.subTest(method=method):
                model, _ = benchmark.build_model(method, selective=True)
                backbone = model.backbone.train()
                backbone.to('cpu')
                with torch.no_grad():
                    for plugin in backbone.plugins.values():
                        plugin.raw_alpha.fill_(.4)
                features = backbone(torch.randn(2, 3, 96, 96))
                sum(t.square().mean() for t in features).backward()
                for name, parameter in backbone.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for point, plugin in backbone.plugins.items():
                    self.assertNotEqual(plugin.raw_alpha.grad.item(), 0, point)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for name, p in plugin.named_parameters() if name != 'raw_alpha'))
                self.assertGreater(backbone.conv1.conv1_1.conv.weight.grad.abs().sum().item(), 0)

    def test_pretrained_load_all_original_keys_only_new_plugin_keys_missing(self):
        state = PResNet(18, freeze_norm=False).state_dict()
        for method in benchmark.METHODS:
            with self.subTest(method=method):
                points = benchmark.audit.fresh_config(benchmark.config_path(method)).get('BackbonePlugins')
                with patch('torch.hub.load_state_dict_from_url', return_value=state) as download:
                    model = PResNet(18, freeze_norm=False, pretrained=True, BackbonePlugins=points)
                download.assert_called_once()
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                missing = model.load_state_dict(state, strict=False).missing_keys
                self.assertTrue(all(k.startswith('plugins.') for k in missing))
        bad_state = dict(state)
        bad_state.pop('conv1.conv1_1.conv.weight')
        with patch('torch.hub.load_state_dict_from_url', return_value=bad_state):
            with self.assertRaisesRegex(RuntimeError, 'backbone keys mismatch'):
                PResNet(18, freeze_norm=False, pretrained=True,
                        BackbonePlugins={'P3': {'enabled': True, 'type': 'secd'}})

    def test_yaml_candidate_candidate_off_no_shared_registry_leak(self):
        benchmark.build_model('P1-DEConv', selective=True,
                              override={'P1': {'enabled': True, 'alpha_max': .75}})
        second, _ = benchmark.build_model('P0-SRFD', selective=True)
        self.assertEqual(list(second.backbone.plugins), ['p0'])
        self.assertEqual(second.backbone.plugins.p0.alpha_max, .2)
        third, _ = benchmark.build_model('P1-DEConv', selective=True)
        self.assertEqual(third.backbone.plugins.p1.alpha_max, .2)
        baseline, _ = benchmark.build_model('Baseline', selective=True)
        self.assertIsNone(baseline.backbone.plugins)

    def test_mutual_exclusion_validation_and_combinations_warn(self):
        for invalid in ({'P4': {'enabled': True, 'type': ['fadc', 'wtconv']}},
                        {'P4': {'enabled': True, 'type': 'both'}},
                        {'P5': {'enabled': True}}, {'P1': {'enabled': 'false'}}):
            with self.assertRaises((ValueError, TypeError)):
                build_plugins(invalid, 18, 'd', 4)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            model = PResNet(18, return_idx=[1, 2, 3], freeze_norm=False,
                           BackbonePlugins={'P0': {'enabled': True}, 'P3': {'enabled': True}}).eval()
        self.assertTrue(any('combination' in str(w.message) for w in emitted))
        with torch.no_grad():
            features = model(torch.randn(1, 3, 97, 99))
        self.assertEqual([list(t.shape[-2:]) for t in features], [[13, 13], [7, 7], [4, 4]])
        with self.assertRaisesRegex(ValueError, 'do not mix'):
            PResNet(18, SECD={'enabled': True}, BackbonePlugins={'P3': {'enabled': True}})

    def test_uav_dcnv4_self_contained_math_and_backward(self):
        model, _ = benchmark.build_model('P2-UAVDCNv4', selective=True)
        module = model.backbone.plugins.p2.branch
        self.assertIsInstance(module, UAVDCNv4)
        self.assertTrue(callable(load_backend()))
        self.assertEqual(module.groups, 4)
        self.assertTrue(torch.equal(module.offset_mask.weight,
                                    torch.zeros_like(module.offset_mask.weight)))
        x = torch.randn(2, 128, 15, 17, requires_grad=True)
        offset, mask = module._predict_offset_mask(x)
        self.assertTrue(torch.equal(offset, torch.zeros_like(offset)))
        normalized = mask.reshape(2, 4, 9, 15, 17).sum(2)
        torch.testing.assert_close(normalized, torch.ones_like(normalized))
        result = module(x)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(module.offset_mask.weight.grad.abs().sum().item(), 0)

        # At initialization, zero offsets + uniform masks reproduce a local
        # mean, which cancels in the adaptive term. The remaining branch is
        # exactly the configured high-frequency evidence (up to BN epsilon).
        small = UAVDCNv4(4, groups=1, detail_gain=.5).eval()
        image = torch.randn(2, 4, 11, 13)
        expected = .5 * (image - F.avg_pool2d(
            image, 3, stride=1, padding=1, count_include_pad=True))
        expected = expected / math.sqrt(1 + small.bn.eps)
        torch.testing.assert_close(small(image), expected, atol=1e-6, rtol=1e-6)

        for options in ({'groups': 3}, {'max_offset': 0},
                        {'temperature': 0}, {'detail_gain': 1}):
            with self.assertRaises(ValueError):
                UAVDCNv4(128, **options)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_train_backbone_640_and_fp16_output_dtype(self):
        for method in RUNNABLE:
            with self.subTest(method=method):
                model, _ = benchmark.build_model(method, selective=True)
                backbone = model.backbone.cuda().train()
                for plugin in backbone.plugins.values():
                    with torch.no_grad():
                        plugin.raw_alpha.fill_(.4)
                optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-5)
                scaler = torch.amp.GradScaler('cuda', init_scale=128)
                with torch.autocast(device_type='cuda', cache_enabled=True):
                    features = backbone(torch.randn(1, 3, 640, 640, device='cuda'))
                self.assertTrue(all(t.dtype == torch.float16 and torch.isfinite(t).all() for t in features))
                with torch.autocast('cuda', enabled=False):
                    loss = sum(t.float().square().mean() for t in features)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                self.assertTrue(torch.isfinite(loss))
                for name, parameter in backbone.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                scaler.step(optimizer)
                scaler.update()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_detector_training_denoising_amp_backward(self):
        for method in RUNNABLE:
            with self.subTest(method=method):
                model, _ = benchmark.build_model(method, selective=True)
                model.cuda().train()
                # Smoke only: avoid a random scale draw to keep the requested
                # 640 test shape. The training YAML/protocol remains unchanged.
                model.multi_scale = None
                with torch.no_grad():
                    for plugin in model.backbone.plugins.values():
                        plugin.raw_alpha.fill_(.4)
                targets = [{'labels': torch.zeros(1, dtype=torch.int64, device='cuda'),
                            'boxes': torch.tensor([[.5, .5, .02, .02]], device='cuda')}]
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                scaler = torch.amp.GradScaler('cuda', init_scale=128)
                with torch.autocast(device_type='cuda', cache_enabled=True):
                    outputs = model(torch.randn(1, 3, 640, 640, device='cuda'), targets)
                self.assertIn('dn_meta', outputs)
                with torch.autocast('cuda', enabled=False):
                    branches = [outputs] + outputs['aux_outputs'] + outputs['dn_aux_outputs']
                    loss = sum(branch['pred_logits'].float().square().mean()
                               + branch['pred_boxes'].float().square().mean() for branch in branches)
                self.assertTrue(torch.isfinite(loss))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                for name, parameter in model.backbone.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(all(torch.isfinite(branch[key]).all()
                                    for branch in branches for key in ('pred_logits', 'pred_boxes')))
                scaler.step(optimizer)
                scaler.update()


class SourceMathTests(unittest.TestCase):
    def test_wtconv_db1_fixed_transform_reconstruction_and_odd_backward(self):
        from src.nn.backbone.plugin_points.wtconv import WTConv2d
        module = WTConv2d(8, wt_levels=2)
        x = torch.randn(2, 8, 10, 12)
        coefficients = F.conv2d(x, module.wt_filter, stride=2, groups=8)
        reconstructed = F.conv_transpose2d(coefficients, module.iwt_filter, stride=2, groups=8)
        torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=1e-6)
        odd = torch.randn(2, 8, 9, 11, requires_grad=True)
        out = module(odd)
        self.assertEqual(out.shape, odd.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(odd.grad).all())

    def test_fadc_frequency_selection_matches_actual_reference_class(self):
        from src.nn.backbone.plugin_points.fadc import FrequencySelection
        reference_path = ROOT.parents[1] / 'RTDETR-main/ultralytics/nn/extra_modules/fadc.py'
        if not reference_path.exists():
            self.skipTest('optional local reference project not available on server')
        tree = ast.parse(reference_path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'FrequencySelection']
        namespace = {'torch': torch, 'nn': nn, 'F': F}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference_path), 'exec'), namespace)
        original = namespace['FrequencySelection'](8, k_list=[3, 5, 7, 9], spatial_group=1)
        module = FrequencySelection(8)
        with torch.no_grad():
            for conv in original.freq_weight_conv_list:
                conv.weight.normal_(0, .1)
        module.load_state_dict(original.state_dict())
        x = torch.randn(2, 8, 10, 12, requires_grad=True)
        left = original(x)
        right = module(x)
        torch.testing.assert_close(right, left, atol=1e-6, rtol=1e-6)
        a = torch.autograd.grad(left.square().mean(), x, retain_graph=True)[0]
        b = torch.autograd.grad(right.square().mean(), x)[0]
        torch.testing.assert_close(b, a, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
