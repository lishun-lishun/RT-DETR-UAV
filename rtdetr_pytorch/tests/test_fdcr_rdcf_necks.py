"""Acceptance tests for independent post-CCFF FDCR/RDCF experiments.

The default suite keeps feature sizes compact.  Set
``RUN_FDCR_RDCF_FULL_MODEL_TESTS=1`` for complete detector equivalence and
``RUN_FDCR_RDCF_FULL_RES=1`` for all four detectors at 480/640/800.
Formal dataset training is never started by this module.
"""

import copy
import gc
import os
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
PRES_SPDR = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_spdr.yml'
HR_SPDR = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_spdr.yml'
PRES_FDCR = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_fdcr.yml'
HR_FDCR = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_fdcr.yml'
PRES_RDCF = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_rdcf.yml'
HR_RDCF = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_rdcf.yml'

CANDIDATES = (PRES_FDCR, HR_FDCR, PRES_RDCF, HR_RDCF)
REFERENCE_FOR = {
    PRES_FDCR: PRES_BASE,
    PRES_RDCF: PRES_BASE,
    HR_FDCR: HR_BASE,
    HR_RDCF: HR_BASE,
}
EXPECTED_OPTIONS = {
    'FDCR': {
        'enabled': True,
        'gamma_max': 0.30,
        'gamma_init': 0.05,
    },
    'RDCF': {
        'enabled': True,
        'eta_max': 0.30,
        'eta_init': 0.05,
        'deploy': False,
    },
}


def is_hrnet(path):
    return 'hrnetv2' in Path(path).name


def method_for(path):
    stem = Path(path).stem
    if stem.endswith('_fdcr'):
        return 'FDCR'
    if stem.endswith('_rdcf'):
        return 'RDCF'
    raise ValueError(f'not a candidate config: {path}')


def prefixes_for(path):
    prefix = method_for(path).lower()
    return (f'encoder.{prefix}3.', f'encoder.{prefix}4.')


def build(path, **extra_overrides):
    overrides = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
        if is_hrnet(path) else {'PResNet': {'pretrained': False}})
    overrides.update(extra_overrides)
    config = YAMLConfig(str(path), **overrides)
    model = config.model
    model.multi_scale = None
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    return config, model


def encoder_kwargs(channels):
    return dict(
        in_channels=channels,
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=8,
        dim_feedforward=64,
        expansion=0.5,
        depth_mult=0.34,
        num_encoder_layers=0,
        eval_spatial_size=None,
    )


def synthetic_features(channels, batch=1, requires_grad=False):
    return [
        torch.randn(batch, channels[0], 16, 16,
                    requires_grad=requires_grad),
        torch.randn(batch, channels[1], 8, 8,
                    requires_grad=requires_grad),
        torch.randn(batch, channels[2], 4, 4,
                    requires_grad=requires_grad),
    ]


def assert_tensor_lists_close(case, expected, actual,
                              atol=1e-6, rtol=1e-5):
    case.assertEqual(len(expected), len(actual))
    for index, (left, right) in enumerate(zip(expected, actual)):
        case.assertTrue(
            torch.allclose(left, right, atol=atol, rtol=rtol),
            f'level {index}: max_abs='
            f'{(left.float() - right.float()).abs().max().item()}')


def recursive_tensor_loss(value):
    if torch.is_tensor(value) and value.is_floating_point():
        return value.float().square().mean()
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        children = ()
    terms = [recursive_tensor_loss(child) for child in children]
    terms = [term for term in terms if term is not None]
    return sum(terms) if terms else None


class FDCRRDCFConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_final_top_level_dut_config_set_is_exactly_eight(self):
        actual = {path.name for path in CONFIG_DIR.glob(
            'rtdetr*_dut_anti_uav*.yml')}
        expected = {path.name for path in (
            PRES_BASE, HR_BASE, PRES_SPDR, HR_SPDR,
            PRES_FDCR, HR_FDCR, PRES_RDCF, HR_RDCF)}
        self.assertEqual(actual, expected)

    def test_resolved_configs_change_only_method_and_output(self):
        for candidate in CANDIDATES:
            method = method_for(candidate)
            baseline = REFERENCE_FOR[candidate]
            reference_config = fresh_config(baseline)
            candidate_config = fresh_config(candidate)
            diff = differences(reference_config, candidate_config)
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', method)
                or key.startswith(method + '.'))]
            self.assertEqual(unexpected, [], f'{candidate.name}: {diff}')
            self.assertEqual(
                candidate_config[method], EXPECTED_OPTIONS[method])
            other = 'RDCF' if method == 'FDCR' else 'FDCR'
            self.assertFalse(candidate_config[other]['enabled'])
            self.assertFalse(candidate_config['SPDR']['enabled'])

    def test_four_candidates_enable_exactly_one_new_neck_and_no_spdr(self):
        for candidate in CANDIDATES:
            _config, model = build(candidate)
            method = method_for(candidate)
            encoder = model.encoder
            self.assertEqual(encoder.fdcr_enabled, method == 'FDCR')
            self.assertEqual(encoder.rdcf_enabled, method == 'RDCF')
            self.assertFalse(encoder.spdr_enabled)
            for prefix in ('fdcr', 'rdcf'):
                expected = prefix.upper() == method
                self.assertEqual(hasattr(encoder, prefix + '3'), expected)
                self.assertEqual(hasattr(encoder, prefix + '4'), expected)
            block3 = getattr(encoder, method.lower() + '3')
            block4 = getattr(encoder, method.lower() + '4')
            self.assertIsNot(block3, block4)
            self.assertTrue(set(map(id, block3.parameters())).isdisjoint(
                set(map(id, block4.parameters()))))
            del model, _config
            gc.collect()

    def test_both_retained_spdr_detectors_build_and_forward(self):
        for path in (PRES_SPDR, HR_SPDR):
            _config, model = build(path)
            model.eval()
            self.assertTrue(model.encoder.spdr_enabled)
            self.assertFalse(model.encoder.fdcr_enabled)
            self.assertFalse(model.encoder.rdcf_enabled)
            with torch.inference_mode():
                output = model(torch.randn(1, 3, 128, 128))
            self.assertEqual(tuple(output['pred_logits'].shape), (1, 300, 1))
            self.assertEqual(tuple(output['pred_boxes'].shape), (1, 300, 4))
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in output.values()
                                if torch.is_tensor(value)))
            del model, _config, output
            gc.collect()

    def test_fdcr_and_rdcf_reject_joint_enablement(self):
        with self.assertRaisesRegex(
                ValueError,
                'FDCR and RDCF must be evaluated independently'):
            HybridEncoder(
                **encoder_kwargs([16, 32, 64]),
                FDCR={'enabled': True},
                RDCF={'enabled': True})


class FDCRRDCFEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_disabled_encoder_is_original_for_both_backbone_contracts(self):
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = encoder_kwargs(channels)
            torch.manual_seed(7)
            reference = HybridEncoder(**kwargs).eval()
            for method in ('FDCR', 'RDCF'):
                torch.manual_seed(7)
                disabled = HybridEncoder(
                    **kwargs, **{method: {'enabled': False}}).eval()
                self.assertEqual(reference.state_dict().keys(),
                                 disabled.state_dict().keys())
                for key, value in reference.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, disabled.state_dict()[key]), key)
                features = synthetic_features(channels)
                with torch.inference_mode():
                    expected = reference(features)
                    actual = disabled(features)
                assert_tensor_lists_close(self, expected, actual)

    def test_common_seeded_weights_are_exact_and_n5_is_identity(self):
        options = {
            'FDCR': {'enabled': True, 'gamma_max': 0.30,
                     'gamma_init': 0.05},
            'RDCF': {'enabled': True, 'eta_max': 0.30,
                     'eta_init': 0.05, 'deploy': False},
        }
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = encoder_kwargs(channels)
            for method in ('FDCR', 'RDCF'):
                torch.manual_seed(13)
                reference = HybridEncoder(**kwargs).eval()
                torch.manual_seed(13)
                candidate = HybridEncoder(
                    **kwargs, **{method: options[method]}).eval()
                reference_state = reference.state_dict()
                candidate_state = candidate.state_dict()
                common = set(reference_state).intersection(candidate_state)
                self.assertEqual(common, set(reference_state))
                changed = [key for key in common if not torch.equal(
                    reference_state[key], candidate_state[key])]
                self.assertEqual(changed, [], method)
                features = synthetic_features(channels)
                with torch.inference_mode():
                    baseline_outputs = reference(features)
                    candidate_outputs = candidate(features)
                self.assertTrue(torch.equal(
                    baseline_outputs[2], candidate_outputs[2]), method)

    def test_dummy_backward_reaches_every_new_parameter(self):
        options = {
            'FDCR': {'enabled': True},
            'RDCF': {'enabled': True},
        }
        for channels in ([128, 256, 512], [36, 72, 144]):
            for method in ('FDCR', 'RDCF'):
                encoder = HybridEncoder(
                    **encoder_kwargs(channels),
                    **{method: options[method]}).train()
                outputs = encoder(synthetic_features(
                    channels, batch=2, requires_grad=True))
                loss = sum(value.float().square().mean()
                           for value in outputs)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                prefix = method.lower()
                parameters = [(name, parameter)
                              for name, parameter in encoder.named_parameters()
                              if name.startswith((prefix + '3.', prefix + '4.'))]
                self.assertTrue(parameters)
                bad = [name for name, parameter in parameters
                       if parameter.grad is None
                       or not torch.isfinite(parameter.grad).all()
                       or parameter.grad.abs().sum().item() == 0.0]
                self.assertEqual(bad, [], method)

    def test_rdcf_train_deploy_equivalence_and_strict_state_load(self):
        config = {
            'enabled': True,
            'eta_max': 0.30,
            'eta_init': 0.05,
            'deploy': False,
        }
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = encoder_kwargs(channels)
            training = HybridEncoder(
                **kwargs, RDCF=config).eval()
            features = synthetic_features(channels)
            with torch.inference_mode():
                before = training(features)
            training.rdcf3.switch_to_deploy()
            training.rdcf4.switch_to_deploy()
            with torch.inference_mode():
                converted = training(features)
            assert_tensor_lists_close(
                self, before, converted, atol=1e-5, rtol=1e-4)
            self.assertTrue(torch.equal(before[2], converted[2]))

            deploy_config = copy.deepcopy(config)
            deploy_config['deploy'] = True
            deployed = HybridEncoder(
                **kwargs, RDCF=deploy_config).eval()
            incompatible = deployed.load_state_dict(
                training.state_dict(), strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            with torch.inference_mode():
                loaded = deployed(features)
            assert_tensor_lists_close(
                self, converted, loaded, atol=1e-5, rtol=1e-4)
            state_keys = set(training.state_dict())
            self.assertTrue(any('.reparam_conv.' in key
                                for key in state_keys))
            self.assertFalse(any('.dw_3x3.' in key
                                 or '.dw_1x9.' in key
                                 or '.dw_9x1.' in key
                                 for key in state_keys))

    def test_cpu_bfloat16_autocast_is_finite(self):
        for method in ('FDCR', 'RDCF'):
            encoder = HybridEncoder(
                **encoder_kwargs([36, 72, 144]),
                **{method: {'enabled': True}}).eval()
            features = synthetic_features(
                [36, 72, 144], requires_grad=True)
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                outputs = encoder(features)
                loss = sum(value.float().square().mean()
                           for value in outputs)
            self.assertTrue(torch.isfinite(loss), method)
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in outputs), method)
            loss.backward()
            prefix = method.lower()
            bad = [name for name, parameter in encoder.named_parameters()
                   if name.startswith((prefix + '3.', prefix + '4.'))
                   and (parameter.grad is None
                        or not torch.isfinite(parameter.grad).all())]
            self.assertEqual(bad, [], method)


class FDCRRDCFFullModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_optimizer_policy_and_full_common_weights(self):
        for candidate_path in CANDIDATES:
            method = method_for(candidate_path)
            torch.manual_seed(31)
            _reference_config, reference = build(
                REFERENCE_FOR[candidate_path])
            torch.manual_seed(31)
            config, candidate = build(candidate_path)

            reference_state = reference.state_dict()
            candidate_state = candidate.state_dict()
            common = set(reference_state).intersection(candidate_state)
            self.assertEqual(common, set(reference_state))
            changed = [key for key in common if not torch.equal(
                reference_state[key], candidate_state[key])]
            self.assertEqual(changed, [], candidate_path.name)

            assignment = {}
            for group_index, group in enumerate(
                    config.optimizer.param_groups):
                for parameter in group['params']:
                    assignment[id(parameter)] = (
                        group_index, group['lr'], group['weight_decay'])
            rows = [(name, parameter)
                    for name, parameter in candidate.named_parameters()
                    if name.startswith(prefixes_for(candidate_path))]
            self.assertTrue(rows)
            for name, parameter in rows:
                self.assertIn(id(parameter), assignment, name)
                _group, lr, weight_decay = assignment[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                expected_decay = 0.0 if name.endswith('.bias') else 1e-4
                self.assertAlmostEqual(
                    weight_decay, expected_decay, places=12, msg=name)
                if name.endswith(('raw_gamma', 'raw_eta')):
                    self.assertEqual(weight_decay, 1e-4, name)

            del _reference_config, reference, config, candidate
            gc.collect()

    @unittest.skipUnless(
        os.environ.get('RUN_FDCR_RDCF_FULL_MODEL_TESTS') == '1',
        'set RUN_FDCR_RDCF_FULL_MODEL_TESTS=1 for full detector audit')
    def test_disabled_full_models_match_features_and_predictions(self):
        image = torch.randn(1, 3, 128, 128)
        for candidate_path in CANDIDATES:
            method = method_for(candidate_path)
            torch.manual_seed(17)
            _baseline_config, baseline = build(
                REFERENCE_FOR[candidate_path])
            torch.manual_seed(17)
            _disabled_config, disabled = build(
                candidate_path, **{method: {'enabled': False}})
            self.assertEqual(baseline.state_dict().keys(),
                             disabled.state_dict().keys())
            disabled.load_state_dict(baseline.state_dict(), strict=True)
            baseline.eval()
            disabled.eval()
            with torch.inference_mode():
                baseline_backbone = baseline.backbone(image)
                disabled_backbone = disabled.backbone(image)
                baseline_neck = baseline.encoder(baseline_backbone)
                disabled_neck = disabled.encoder(disabled_backbone)
                baseline_output = baseline(image)
                disabled_output = disabled(image)
            assert_tensor_lists_close(
                self, baseline_backbone, disabled_backbone)
            assert_tensor_lists_close(self, baseline_neck, disabled_neck)
            for key in ('pred_logits', 'pred_boxes'):
                self.assertTrue(torch.allclose(
                    baseline_output[key], disabled_output[key],
                    atol=1e-6, rtol=1e-5),
                    f'{candidate_path.name}: {key}')
            del baseline, disabled, _baseline_config, _disabled_config
            gc.collect()

    @unittest.skipUnless(
        os.environ.get('RUN_FDCR_RDCF_FULL_MODEL_TESTS') == '1',
        'set RUN_FDCR_RDCF_FULL_MODEL_TESTS=1 for full detector audit')
    def test_full_detector_dummy_backward_reaches_every_new_parameter(self):
        for path in CANDIDATES:
            _config, model = build(path)
            model.train()
            image = torch.randn(1, 3, 128, 128)
            targets = [{
                'labels': torch.tensor([0], dtype=torch.long),
                'boxes': torch.tensor(
                    [[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
            }]
            output = model(image, targets)
            loss = recursive_tensor_loss(output)
            self.assertIsNotNone(loss)
            self.assertTrue(torch.isfinite(loss), path.name)
            loss.backward()
            rows = [(name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith(prefixes_for(path))]
            self.assertTrue(rows)
            bad = [name for name, parameter in rows
                   if parameter.grad is None
                   or not torch.isfinite(parameter.grad).all()
                   or parameter.grad.abs().sum().item() == 0.0]
            self.assertEqual(bad, [], path.name)
            del model, _config, image, output, loss
            gc.collect()

    @unittest.skipUnless(
        os.environ.get('RUN_FDCR_RDCF_FULL_RES') == '1',
        'set RUN_FDCR_RDCF_FULL_RES=1 for 480/640/800 detector audit')
    def test_all_four_candidates_forward_at_480_640_800(self):
        for path in CANDIDATES:
            _config, model = build(path)
            model.eval()
            captured = []
            handle = model.encoder.register_forward_hook(
                lambda _module, _inputs, output: captured.append(
                    [tuple(value.shape) for value in output]))
            try:
                for size in (480, 640, 800):
                    with torch.inference_mode():
                        output = model(torch.randn(1, 3, size, size))
                    self.assertEqual(captured[-1], [
                        (1, 256, size // 8, size // 8),
                        (1, 256, size // 16, size // 16),
                        (1, 256, size // 32, size // 32),
                    ], f'{path.name}@{size}')
                    self.assertEqual(tuple(output['pred_logits'].shape),
                                     (1, 300, 1))
                    self.assertEqual(tuple(output['pred_boxes'].shape),
                                     (1, 300, 4))
                    self.assertTrue(all(
                        torch.isfinite(value).all()
                        for value in output.values()
                        if torch.is_tensor(value)))
            finally:
                handle.remove()
            del model, _config
            gc.collect()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_cuda_amp_forward_backward_is_finite(self):
        for path in CANDIDATES:
            channels = ([36, 72, 144] if is_hrnet(path)
                        else [128, 256, 512])
            _config, model = build(path)
            encoder = model.encoder.cuda().train()
            features = [value.cuda() for value in synthetic_features(
                channels, batch=2)]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = encoder(features)
                loss = sum(value.float().square().mean()
                           for value in outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), path.name)
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in outputs), path.name)
            prefixes = tuple(prefix.split('encoder.', 1)[1]
                             for prefix in prefixes_for(path))
            bad = [name for name, parameter in encoder.named_parameters()
                   if name.startswith(prefixes)
                   and (parameter.grad is None
                        or not torch.isfinite(parameter.grad).all())]
            self.assertEqual(bad, [], path.name)
            del model, encoder, _config
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
