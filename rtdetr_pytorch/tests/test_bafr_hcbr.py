"""Synthetic and integration checks for PResNet18 BAFR and HCBR.

No dataset, checkpoint download, or training run is required.  Run from
``rtdetr_pytorch`` with ``python -m unittest discover -s tests -p
test_bafr_hcbr.py -v``.
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
    '_bafr_hcbr_audit', ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
core = audit.import_model_source(selective=True)

from src.nn.backbone.backbone_modules.bafr import BAFRBlock, BAFRSpatialField
from src.nn.backbone.backbone_modules.hcbr import HCBR
from src.nn.backbone.presnet import PResNet


CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'bafr': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bafr.yml',
    'hcbr': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_hcbr.yml',
    'combined': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bafr_hcbr.yml',
}


def build(name, seed=0):
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(CONFIGS[name]), PResNet={'pretrained': False})
    model = config.model.eval()
    model.multi_scale = None
    return model


def prior_presnet_class():
    """Read-only reference from committed source before this feature."""
    source = subprocess.check_output(
        ['git', 'show', 'HEAD:rtdetr_pytorch/src/nn/backbone/presnet.py'],
        cwd=ROOT, text=True, encoding='utf-8')
    source = source.replace('from src.core import register', 'register = lambda cls: cls')
    namespace = {'__name__': 'src.nn.backbone._pre_bafr_hcbr_reference',
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


class BAFRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_constant_sharp_blur_and_background(self):
        field = BAFRSpatialField(4)
        constant = torch.full((1, 4, 41, 41), 2.0)
        ef, es, b = field.route_evidence(constant)
        self.assertEqual(ef.abs().max().item(), 0)
        self.assertEqual(es.abs().max().item(), 0)
        self.assertEqual(b.abs().max().item(), 0)

        y, x = torch.meshgrid(torch.arange(41), torch.arange(41), indexing='ij')
        sharp = torch.zeros_like(constant)
        sharp[:, :, 20, 20] = 1
        blur = torch.exp(-((x - 20).square() + (y - 20).square()).float() / 18)
        blur = blur[None, None].expand_as(sharp)
        _, _, sharp_b = field.route_evidence(sharp)
        _, _, blur_b = field.route_evidence(blur)
        sharp_mean = sharp_b[:, :, 15:26, 15:26].mean()
        blur_mean = blur_b[:, :, 15:26, 15:26].mean()
        self.assertGreater(blur_mean.item(), sharp_mean.item())
        print(f'BAFR sharp_b={sharp_mean.item():.6f} blur_b={blur_mean.item():.6f}')

        # Uniform support is route-neutral even right up against all borders.
        texture = torch.randn(2, 4, 9, 11)
        route = field.route_evidence(texture)[2]
        self.assertEqual(tuple(route.shape), (2, 1, 9, 11))
        self.assertTrue(torch.isfinite(route).all())
        self.assertGreaterEqual(route.min().item(), 0)
        self.assertLessEqual(route.max().item(), 1)

    def test_gradients(self):
        torch.manual_seed(11)
        field = BAFRSpatialField(4)
        optimizer = torch.optim.SGD(field.parameters(), lr=0.05)
        probe = torch.randn(2, 4, 12, 13)
        records = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(2, 4, 12, 13, requires_grad=True)
            loss = (field(x) * probe).mean()
            loss.backward()
            record = {
                'fine': field.fine_dw.weight.grad.norm().item(),
                'support1': field.support_dw1.weight.grad.norm().item(),
                'support2': field.support_dw2.weight.grad.norm().item(),
                'input': x.grad.norm().item(),
            }
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) and value > 0
                                for value in record.values()), record)
            records.append(record)
            optimizer.step()
        print('BAFR step_grad_norms=', records)


class HCBRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_constant_anomaly_texture_and_border(self):
        module = HCBR(2)
        constant = torch.ones(1, 2, 25, 27)
        fields = module.compute_components(constant)
        for name in ('background7', 'background11', 'background'):
            torch.testing.assert_close(fields[name], constant, rtol=0, atol=0)
        self.assertEqual(fields['residual'].abs().max().item(), 0)
        self.assertLess(fields['gate'].abs().max().item(), 1e-5)

        anomaly = torch.zeros(1, 2, 25, 27)
        anomaly[:, 1] = 1
        anomaly[:, 0, 12, 13] = 3
        gate = module.compute_components(anomaly)['gate']
        center = gate[0, 0, 12, 13].item()
        background = gate[0, 0, 9:12, 9:12].mean().item()
        self.assertGreater(center, background + 0.5)
        self.assertLess(background, 0.05)

        # Repeated scalar texture with fixed channel direction is predictable
        # in cosine space even though neighboring pixel magnitudes differ.
        y, x = torch.meshgrid(torch.arange(25), torch.arange(27), indexing='ij')
        pattern = (1 + 0.2 * ((x + y) % 2).float())[None, None]
        repeated = torch.cat((pattern, 2 * pattern), dim=1)
        repeated_gate = module.compute_components(repeated)['gate']
        self.assertLess(repeated_gate.abs().max().item(), 1e-5)
        print(f'HCBR anomaly_gate={center:.6f} background_gate={background:.6f} '
              f'repeated_gate={repeated_gate.mean().item():.6f}')

        # Replicate padding keeps a constant feature map exact even when its
        # spatial extent is smaller than the 11x11 outer neighborhood.
        tiny = torch.full((1, 2, 3, 4), 1.5)
        small = module.compute_components(tiny)
        for name in ('background7', 'background11'):
            torch.testing.assert_close(small[name], tiny, rtol=0, atol=0)

    def test_hcbr_zero_init_equivalence(self):
        module = HCBR(4).eval()
        x = torch.randn(2, 4, 13, 17)
        self.assertEqual(module.lambda_effective.item(), 0)
        self.assertTrue(torch.equal(module(x), x))

    def test_gradients(self):
        torch.manual_seed(13)
        module = HCBR(4)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
        x = torch.randn(2, 4, 15, 17)
        records = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            sample = x.clone().requires_grad_(True)
            fields = module.compute_components(sample)
            probe = (fields['gate'] * fields['residual']).detach()
            loss = (module(sample) * probe).mean()
            loss.backward()
            record = {
                'lambda': module.raw_lambda.grad.norm().item(),
                'router': module.scale_router.weight.grad.norm().item(),
                'input': sample.grad.norm().item(),
            }
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in record.values()))
            self.assertGreater(record['lambda'], 0)
            self.assertGreater(record['input'], 0)
            records.append(record)
            optimizer.step()
        self.assertEqual(records[0]['router'], 0)
        self.assertGreater(records[1]['router'], 0)
        print('HCBR step_grad_norms=', records)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_structure_and_config_fairness(self):
        baseline = audit.fresh_config(CONFIGS['baseline'])
        expected = {
            'baseline': (False, False), 'bafr': (True, False),
            'hcbr': (False, True), 'combined': (True, True),
        }
        for name, (bafr_on, hcbr_on) in expected.items():
            config = audit.fresh_config(CONFIGS[name])
            diff = audit.differences(baseline, config)
            self.assertTrue(all(key in ('__include__', 'output_dir') or key.startswith(
                ('BackboneEnhancement.', 'BAFR.', 'HCBR.')) for key in diff), diff)
            self.assertEqual(config['BackboneEnhancement'],
                             {'bafr': bafr_on, 'hcbr': hcbr_on})
            for switch in ('MERT', 'SECD', 'CCED', 'GRER'):
                self.assertFalse(config[switch]['enabled'])
            backbone = build(name, seed=17).backbone
            self.assertEqual([len(stage.blocks) for stage in backbone.res_layers],
                             [2, 2, 2, 2])
            self.assertEqual(backbone.out_channels, [128, 256, 512])
            self.assertEqual(backbone.out_strides, [8, 16, 32])
            self.assertEqual(isinstance(backbone.res_layers[1].blocks[1], BAFRBlock),
                             bafr_on)
            self.assertEqual(backbone.hcbr_p3 is not None, hcbr_on)
            self.assertEqual(backbone.hcbr_p4 is not None, hcbr_on)
            with torch.no_grad():
                levels = backbone(torch.randn(1, 3, 640, 640))
            self.assertEqual([tuple(level.shape) for level in levels],
                             [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)])

    def test_combined_execution_order(self):
        backbone = build('combined', seed=18).backbone
        events = []
        modules = (
            ('bafr', backbone.res_layers[1].blocks[1]),
            ('hcbr_p3', backbone.hcbr_p3),
            ('s4_first', backbone.res_layers[2].blocks[0]),
            ('hcbr_p4', backbone.hcbr_p4),
            ('s4_remaining', backbone.res_layers[2].blocks[1]),
            ('s5_first', backbone.res_layers[3].blocks[0]),
        )
        hooks = [module.register_forward_hook(
            lambda module, inputs, output, label=name: events.append(label))
            for name, module in modules]
        with torch.no_grad():
            backbone(torch.randn(1, 3, 128, 128))
        for hook in hooks:
            hook.remove()
        self.assertEqual(events, [name for name, _ in modules])

    def test_baseline_equivalence(self):
        model = build('baseline', seed=19)
        reference = copy.deepcopy(model)
        prior = prior_presnet_class()(
            18, variant='d', num_stages=4, return_idx=[1, 2, 3],
            freeze_at=-1, freeze_norm=False, pretrained=False).eval()
        prior.load_state_dict(model.backbone.state_dict(), strict=True)
        reference.backbone = prior
        image = torch.randn(1, 3, 640, 640)
        captures = {'current_encoder': [], 'old_encoder': [],
                    'current_decoder': [], 'old_decoder': []}
        hooks = [
            model.encoder.register_forward_hook(
                lambda module, inputs, output: captures['current_encoder'].append(output)),
            reference.encoder.register_forward_hook(
                lambda module, inputs, output: captures['old_encoder'].append(output)),
            model.decoder.register_forward_hook(
                lambda module, inputs, output: captures['current_decoder'].append(output)),
            reference.decoder.register_forward_hook(
                lambda module, inputs, output: captures['old_decoder'].append(output)),
        ]
        with torch.no_grad():
            current_features = model.backbone(image)
            old_features = reference.backbone(image)
            current = model(image)
            old = reference(image)
        for hook in hooks:
            hook.remove()
        assert_nested_equal(self, current_features, old_features)
        assert_nested_equal(self, captures['current_encoder'], captures['old_encoder'])
        assert_nested_equal(self, captures['current_decoder'], captures['old_decoder'])
        assert_nested_equal(self, current, old)

    def test_hcbr_zero_init_equivalence(self):
        baseline, hcbr = build('baseline', seed=23), build('hcbr', seed=23)
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, hcbr.state_dict()[name]), name)
        image = torch.randn(1, 3, 640, 640)
        captures = {'base_encoder': [], 'hcbr_encoder': [],
                    'base_decoder': [], 'hcbr_decoder': []}
        hooks = [
            baseline.encoder.register_forward_hook(
                lambda module, inputs, output: captures['base_encoder'].append(output)),
            hcbr.encoder.register_forward_hook(
                lambda module, inputs, output: captures['hcbr_encoder'].append(output)),
            baseline.decoder.register_forward_hook(
                lambda module, inputs, output: captures['base_decoder'].append(output)),
            hcbr.decoder.register_forward_hook(
                lambda module, inputs, output: captures['hcbr_decoder'].append(output)),
        ]
        with torch.no_grad():
            base_levels, hcbr_levels = baseline.backbone(image), hcbr.backbone(image)
            base_output, hcbr_output = baseline(image), hcbr(image)
        for hook in hooks:
            hook.remove()
        assert_nested_equal(self, base_levels, hcbr_levels)
        assert_nested_equal(self, captures['base_encoder'], captures['hcbr_encoder'])
        assert_nested_equal(self, captures['base_decoder'], captures['hcbr_decoder'])
        assert_nested_equal(self, base_output, hcbr_output)

    def test_pretrained_compatibility(self):
        source = PResNet(18, variant='d', return_idx=[1, 2, 3],
                         freeze_norm=False).state_dict()
        cases = {
            'baseline': {'bafr': False, 'hcbr': False},
            'bafr': {'bafr': True, 'hcbr': False},
            'hcbr': {'bafr': False, 'hcbr': True},
            'combined': {'bafr': True, 'hcbr': True},
        }
        with patch('torch.hub.load_state_dict_from_url', return_value=source):
            for name, selection in cases.items():
                candidate = PResNet(
                    18, variant='d', return_idx=[1, 2, 3],
                    freeze_norm=False, pretrained=True,
                    BackboneEnhancement=selection)
                report = candidate.pretrained_load_report
                missing = report['missing_keys']
                unexpected = report['unexpected_keys']
                self.assertTrue(all(key.startswith(('hcbr_p3.', 'hcbr_p4.',
                    'res_layers.1.blocks.1.branch2a.conv.')) for key in missing), missing)
                expected_old_conv = 'res_layers.1.blocks.1.branch2a.conv.weight'
                self.assertEqual(unexpected, [expected_old_conv] if selection['bafr'] else [])
                if not selection['bafr'] and not selection['hcbr']:
                    self.assertEqual(missing, [])
                else:
                    self.assertTrue(missing)
                for key in set(source).intersection(candidate.state_dict()):
                    self.assertTrue(torch.equal(source[key], candidate.state_dict()[key]), key)

        damaged = dict(source)
        damaged.pop('conv1.conv1_1.conv.weight')
        with patch('torch.hub.load_state_dict_from_url', return_value=damaged):
            for selection in cases.values():
                with self.assertRaises(RuntimeError):
                    PResNet(18, variant='d', return_idx=[1, 2, 3],
                            freeze_norm=False, pretrained=True,
                            BackboneEnhancement=selection)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_amp(self):
        for name in ('bafr', 'hcbr', 'combined'):
            backbone = build(name, seed=29).backbone.cuda().train()
            image = torch.randn(1, 3, 128, 128, device='cuda', requires_grad=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = backbone(image)
                loss = sum(level.float().square().mean() for level in outputs)
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
