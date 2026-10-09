"""Acceptance tests for the independent PCX/ESDR/PSCA experiments.

Default tests use compact encoder features.  Set
``RUN_PCX_ESDR_PSCA_FULL_MODEL_TESTS=1`` for complete detector equivalence and
dummy backward, and ``RUN_PCX_ESDR_PSCA_FULL_RES=1`` for all six detectors at
480/640/800.  This module never starts formal training.
"""

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
from tools import validate_pcx_esdr_psca_necks as audit  # noqa: E402


ROOT = PROJECT_DIR
CONFIG_DIR = ROOT / 'configs/rtdetr'
PRES_BASE = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav.yml'
HR_BASE = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav.yml'
PRES_PCX = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_pcx.yml'
HR_PCX = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_pcx.yml'
PRES_ESDR = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_esdr.yml'
HR_ESDR = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_esdr.yml'
PRES_PSCA = CONFIG_DIR / 'rtdetr_r18vd_dut_anti_uav_psca.yml'
HR_PSCA = CONFIG_DIR / 'rtdetr_hrnetv2_w18_dut_anti_uav_psca.yml'

CANDIDATES = (
    PRES_PCX, HR_PCX, PRES_ESDR, HR_ESDR, PRES_PSCA, HR_PSCA)
REFERENCE_FOR = {
    PRES_PCX: PRES_BASE,
    PRES_ESDR: PRES_BASE,
    PRES_PSCA: PRES_BASE,
    HR_PCX: HR_BASE,
    HR_ESDR: HR_BASE,
    HR_PSCA: HR_BASE,
}


def is_hrnet(path):
    return 'hrnetv2' in Path(path).name


def method_for(path):
    suffix = Path(path).stem.rsplit('_', 1)[-1].upper()
    if suffix not in audit.METHODS:
        raise ValueError(path)
    return suffix


def prefixes_for(path):
    method = method_for(path)
    if method == 'PCX':
        return ('encoder.pcx.',)
    if method == 'ESDR':
        return ('encoder.esdr34.', 'encoder.esdr45.')
    return ('encoder.psca3.', 'encoder.psca4.')


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


def options_for(method, enabled=True):
    options = dict(audit.EXPECTED_OPTIONS[method])
    options['enabled'] = enabled
    return options


def assert_tensor_lists_close(case, expected, actual,
                              atol=1e-6, rtol=1e-5):
    case.assertEqual(len(expected), len(actual))
    for index, (left, right) in enumerate(zip(expected, actual)):
        case.assertTrue(torch.allclose(
            left, right, atol=atol, rtol=rtol),
            f'level {index}: max_abs='
            f'{(left.float() - right.float()).abs().max().item()}')


class PCXESDRPSCAConfigTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_eight_required_configs_exist(self):
        required = (PRES_BASE, HR_BASE, *CANDIDATES)
        self.assertTrue(all(path.is_file() for path in required))

    def test_resolved_configs_change_only_method_output_and_include(self):
        for candidate in CANDIDATES:
            method = method_for(candidate)
            reference = fresh_config(REFERENCE_FOR[candidate])
            resolved = fresh_config(candidate)
            diff = differences(reference, resolved)
            unexpected = [key for key in diff if not (
                key in ('__include__', 'output_dir', method)
                or key.startswith(method + '.'))]
            self.assertEqual(unexpected, [], f'{candidate.name}: {diff}')
            self.assertEqual(resolved[method], audit.EXPECTED_OPTIONS[method])
            for other in audit.METHODS:
                if other != method:
                    self.assertFalse(
                        resolved.get(other, {}).get('enabled', False))

    def test_all_pair_and_triple_combinations_are_rejected(self):
        combinations = (
            ('PCX', 'ESDR'),
            ('PCX', 'PSCA'),
            ('ESDR', 'PSCA'),
            ('PCX', 'ESDR', 'PSCA'),
        )
        for enabled in combinations:
            with self.subTest(enabled=enabled):
                with self.assertRaisesRegex(
                        ValueError,
                        'PCX, ESDR and PSCA must be evaluated independently'):
                    HybridEncoder(
                        **encoder_kwargs([16, 32, 64]),
                        **{name: {'enabled': True} for name in enabled})

    def test_new_methods_cannot_mix_with_historical_necks(self):
        historical = (
            'ACR', 'SLR', 'PAF', 'BOR', 'DGFR', 'SPDR', 'FDCR', 'RDCF')
        for method in audit.METHODS:
            for old_method in historical:
                with self.subTest(method=method, old_method=old_method):
                    with self.assertRaisesRegex(
                            ValueError,
                            'independent experiments and cannot mix'):
                        HybridEncoder(
                            **encoder_kwargs([16, 32, 64]),
                            **{method: {'enabled': True},
                               old_method: {'enabled': True}})

    def test_six_candidates_enable_exactly_one_method(self):
        expected_attributes = {
            'PCX': {'pcx'},
            'ESDR': {'esdr34', 'esdr45'},
            'PSCA': {'psca3', 'psca4'},
        }
        all_attributes = set().union(*expected_attributes.values())
        for candidate in CANDIDATES:
            _config, model = build(candidate)
            method = method_for(candidate)
            states = {
                name: getattr(model.encoder, name.lower() + '_enabled')
                for name in audit.METHODS
            }
            self.assertEqual(sum(states.values()), 1, candidate.name)
            self.assertTrue(states[method], candidate.name)
            actual = {name for name in all_attributes
                      if hasattr(model.encoder, name)}
            self.assertEqual(actual, expected_attributes[method])
            if method == 'PCX':
                blocks = (model.encoder.pcx.refine3,
                          model.encoder.pcx.refine4,
                          model.encoder.pcx.refine5)
                parameter_ids = [set(map(id, block.parameters()))
                                 for block in blocks]
                self.assertTrue(all(
                    parameter_ids[left].isdisjoint(parameter_ids[right])
                    for left in range(3)
                    for right in range(left + 1, 3)))
            else:
                names = sorted(actual)
                first = getattr(model.encoder, names[0])
                second = getattr(model.encoder, names[1])
                self.assertTrue(set(map(id, first.parameters())).isdisjoint(
                    set(map(id, second.parameters()))))
            del model, _config
            gc.collect()


class PCXESDRPSCAEncoderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_disabled_encoder_is_exact_original_for_both_contracts(self):
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = encoder_kwargs(channels)
            torch.manual_seed(7)
            reference = HybridEncoder(**kwargs).eval()
            for method in audit.METHODS:
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

    def test_common_seeded_weights_and_exact_insertion_semantics(self):
        for channels in ([128, 256, 512], [36, 72, 144]):
            kwargs = encoder_kwargs(channels)
            for method in audit.METHODS:
                with self.subTest(channels=channels, method=method):
                    torch.manual_seed(13)
                    baseline = HybridEncoder(**kwargs).eval()
                    torch.manual_seed(13)
                    candidate = HybridEncoder(
                        **kwargs, **{method: options_for(method)}).eval()
                    baseline_state = baseline.state_dict()
                    candidate_state = candidate.state_dict()
                    common = set(baseline_state).intersection(candidate_state)
                    self.assertEqual(common, set(baseline_state))
                    self.assertEqual([
                        key for key in common if not torch.equal(
                            baseline_state[key], candidate_state[key])], [])
                    features = synthetic_features(channels)
                    with torch.inference_mode():
                        original = baseline(features)

                    if method == 'PCX':
                        actual, trace = audit._trace_pcx(
                            candidate, original, features)
                        self.assertEqual(trace['call_count'], 1)
                        self.assertEqual(trace['input_levels'], 3)
                    elif method == 'ESDR':
                        actual, trace = audit._trace_esdr(candidate, features)
                        self.assertEqual(trace['call_count'], 2)
                        self.assertTrue(all(
                            row['original_downsample_object_preserved']
                            for row in trace['transitions']))
                    else:
                        actual, trace = audit._trace_psca(
                            candidate, original, features)
                        self.assertEqual(trace['levels'], [3, 4])
                        self.assertTrue(torch.equal(original[2], actual[2]))
                    self.assertEqual(
                        [tuple(value.shape) for value in actual],
                        [tuple(value.shape) for value in original])

    def test_dummy_backward_reaches_every_new_parameter(self):
        local_prefixes = {
            'PCX': ('pcx.',),
            'ESDR': ('esdr34.', 'esdr45.'),
            'PSCA': ('psca3.', 'psca4.'),
        }
        for channels in ([128, 256, 512], [36, 72, 144]):
            for method in audit.METHODS:
                with self.subTest(channels=channels, method=method):
                    encoder = HybridEncoder(
                        **encoder_kwargs(channels),
                        **{method: options_for(method)}).train()
                    outputs = encoder(synthetic_features(
                        channels, batch=2, requires_grad=True))
                    loss = sum(value.float().square().mean()
                               for value in outputs)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    rows = [(name, parameter)
                            for name, parameter in encoder.named_parameters()
                            if name.startswith(local_prefixes[method])]
                    self.assertTrue(rows)
                    invalid = [name for name, parameter in rows
                               if parameter.grad is None
                               or not torch.isfinite(parameter.grad).all()
                               or parameter.grad.abs().sum().item() == 0.0]
                    self.assertEqual(invalid, [], method)

    def test_cpu_bfloat16_all_methods_are_finite(self):
        for method in audit.METHODS:
            encoder = HybridEncoder(
                **encoder_kwargs([36, 72, 144]),
                **{method: options_for(method)}).train()
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


class PCXESDRPSCAFullModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_real_optimizer_policy_for_all_new_parameters(self):
        for candidate in CANDIDATES:
            config, model = build(candidate)
            assignment = {}
            for group_index, group in enumerate(config.optimizer.param_groups):
                for parameter in group['params']:
                    assignment[id(parameter)] = (
                        group_index, group['lr'], group['weight_decay'])
            rows = [(name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith(prefixes_for(candidate))]
            self.assertTrue(rows, candidate.name)
            for name, parameter in rows:
                self.assertIn(id(parameter), assignment, name)
                _group, lr, weight_decay = assignment[id(parameter)]
                self.assertAlmostEqual(lr, 3e-4, places=12, msg=name)
                expected_decay = 0.0 if name.endswith('.bias') else 1e-4
                self.assertAlmostEqual(
                    weight_decay, expected_decay, places=12, msg=name)
                if name.endswith(('raw_beta', 'raw_alpha')):
                    self.assertEqual(weight_decay, 1e-4, name)
            del config, model
            gc.collect()

    def test_psca_attention_macs_are_reported_separately_at_640(self):
        _config, model = build(PRES_PSCA)
        macs = audit.psca_attention_macs(model, 640)
        self.assertEqual(macs['QK'], 102_400_000)
        self.assertEqual(macs['AV'], 204_800_000)
        self.assertEqual(macs['total'], 307_200_000)
        self.assertEqual(set(macs['levels']), {'N3', 'N4'})
        del _config, model
        gc.collect()

    @unittest.skipUnless(
        os.environ.get('RUN_PCX_ESDR_PSCA_FULL_MODEL_TESTS') == '1',
        'set RUN_PCX_ESDR_PSCA_FULL_MODEL_TESTS=1 for detector audit')
    def test_disabled_complete_detectors_match_features_and_predictions(self):
        result = audit.disabled_equivalence(input_size=128)
        self.assertEqual(set(result), set(audit.CANDIDATES))

    @unittest.skipUnless(
        os.environ.get('RUN_PCX_ESDR_PSCA_FULL_MODEL_TESTS') == '1',
        'set RUN_PCX_ESDR_PSCA_FULL_MODEL_TESTS=1 for detector audit')
    def test_complete_detector_dummy_backward(self):
        for name in audit.CANDIDATES:
            _config, model = audit.build(name)
            result = audit.full_detector_backward(model, name, input_size=128)
            self.assertIn('ALL PRESENT, FINITE, NONZERO',
                          result['new_parameter_gradients'])
            del _config, model
            gc.collect()

    @unittest.skipUnless(
        os.environ.get('RUN_PCX_ESDR_PSCA_FULL_RES') == '1',
        'set RUN_PCX_ESDR_PSCA_FULL_RES=1 for 480/640/800 audit')
    def test_six_complete_detectors_forward_at_480_640_800(self):
        for name in audit.CANDIDATES:
            _config, model = audit.build(name)
            result = audit.dynamic_detector_forward(
                model, name, (480, 640, 800))
            self.assertEqual(set(result), {'480', '640', '800'})
            del _config, model
            gc.collect()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_cuda_amp_encoder_forward_backward_is_finite(self):
        for name in audit.CANDIDATES:
            channels = ([36, 72, 144] if audit.is_hrnet(name)
                        else [128, 256, 512])
            _config, model = audit.build(name, training=True)
            encoder = model.encoder.cuda()
            features = [value.cuda() for value in
                        synthetic_features(channels, batch=2)]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = encoder(features)
                loss = sum(value.float().square().mean()
                           for value in outputs)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), name)
            self.assertTrue(all(torch.isfinite(value).all()
                                for value in outputs), name)
            prefixes = audit.local_prefixes_for(name)
            invalid = [parameter_name
                       for parameter_name, parameter
                       in encoder.named_parameters()
                       if parameter_name.startswith(prefixes)
                       and (parameter.grad is None
                            or not torch.isfinite(parameter.grad).all())]
            self.assertEqual(invalid, [], name)
            del _config, model, encoder
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
