"""SECD unit tests, isolated from optional dataset/transformer dependencies.

Run: python -m unittest discover -s tests -p test_secd.py -v
The original reference is read-only code from the current Git HEAD, never an
old experimental implementation. No real pretrained download or training.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = '_secd_test_runtime'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PROJECT / 'src' / 'nn' / 'backbone')]
sys.modules.setdefault(PACKAGE, package)


def _load_module(name, path, source=None):
    spec = importlib.util.spec_from_file_location(f'{PACKAGE}.{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if source is None:
        spec.loader.exec_module(module)
    else:
        # Only avoid the registration side effect; all model code is intact.
        source = source.replace('from src.core import register',
                                'register = lambda cls: cls')
        exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


COMMON = _load_module('common', PROJECT / 'src/nn/backbone/common.py')
SECD = _load_module('secd', PROJECT / 'src/nn/backbone/secd.py').SECDTransition
presnet_path = PROJECT / 'src/nn/backbone/presnet.py'
PResNet = _load_module('presnet', presnet_path,
                       presnet_path.read_text(encoding='utf-8')).PResNet
original_source = subprocess.check_output(
    ['git', 'show', 'HEAD:rtdetr_pytorch/src/nn/backbone/presnet.py'],
    cwd=PROJECT, text=True, encoding='utf-8')
OriginalPResNet = _load_module('original_presnet', presnet_path,
                               original_source).PResNet


def phase_image(phases):
    """[B,C,4,h,w] -> [B,C,2h,2w], in the specified phase order."""
    batch, channels, _, height, width = phases.shape
    image = torch.empty(batch, channels, 2 * height, 2 * width,
                        dtype=phases.dtype)
    image[:, :, 0::2, 0::2] = phases[:, :, 0]
    image[:, :, 0::2, 1::2] = phases[:, :, 1]
    image[:, :, 1::2, 0::2] = phases[:, :, 2]
    image[:, :, 1::2, 1::2] = phases[:, :, 3]
    return image


class SECDMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)
        self.module = SECD(16, 32, groups=8).eval()

    def test_uniform_phases_zero_evidence_and_finite_gradient(self):
        cells = torch.randn(2, 16, 1, 3, 4)
        x = phase_image(cells.repeat(1, 1, 4, 1, 1)).requires_grad_()
        sparse, kappa = self.module.sparse_evidence(x)
        torch.testing.assert_close(sparse, torch.zeros_like(sparse), rtol=0, atol=0)
        torch.testing.assert_close(kappa, torch.zeros_like(kappa), rtol=0, atol=0)
        torch.testing.assert_close(self.module.projection(sparse),
                                   torch.zeros(2, 32, 3, 4), rtol=0, atol=0)
        sparse.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_single_sparse_phase_concentrates(self):
        phases = torch.zeros(1, 16, 4, 3, 4)
        phases[:, :, 0] = 10
        _, kappa = self.module.sparse_evidence(phase_image(phases))
        self.assertGreater(kappa.min().item(), 0.99)

    def test_distributed_texture_less_concentrated(self):
        phases = torch.tensor([10., 10.01, 10.02, 10.03]).reshape(1, 1, 4, 1, 1)
        _, distributed = self.module.sparse_evidence(
            phase_image(phases.repeat(1, 16, 1, 3, 4)))
        phases = torch.tensor([10., 0., 0., 0.]).reshape(1, 1, 4, 1, 1)
        _, single = self.module.sparse_evidence(
            phase_image(phases.repeat(1, 16, 1, 3, 4)))
        self.assertLess(distributed.max().item(), 0.01)
        self.assertTrue((distributed < single).all())

    def test_phase_permutation_invariant(self):
        phases = torch.randn(2, 16, 4, 3, 4)
        sparse, kappa = self.module.sparse_evidence(phase_image(phases))
        permuted_sparse, permuted_kappa = self.module.sparse_evidence(
            phase_image(phases[:, :, [3, 1, 0, 2]]))
        torch.testing.assert_close(kappa, permuted_kappa)
        torch.testing.assert_close(sparse, permuted_sparse)

    def test_formula_matches_independent_group_reference(self):
        phases = torch.randn(2, 16, 4, 3, 4, dtype=torch.float64)
        residual = phases - phases.mean(dim=2, keepdim=True)
        expected_sparse, expected_kappa = [], []
        for group in range(8):
            values = residual[:, group * 2:(group + 1) * 2]
            energy = (values.square().mean(dim=1) + 1e-6).sqrt()
            probability = (energy / 0.1).softmax(dim=1)
            concentration = ((4 * probability.square().sum(dim=1) - 1) / 3).clamp(0, 1)
            evidence = concentration.unsqueeze(1) * (probability.unsqueeze(1) * values).sum(dim=2)
            expected_sparse.append(evidence)
            expected_kappa.append(concentration)
        sparse, kappa = self.module.sparse_evidence(phase_image(phases))
        torch.testing.assert_close(sparse, torch.cat(expected_sparse, dim=1))
        torch.testing.assert_close(kappa, torch.stack(expected_kappa, dim=1))

    def test_odd_shape_replication_and_ceil_output(self):
        x = torch.randn(2, 16, 7, 9)
        sparse, kappa = self.module.sparse_evidence(x)
        self.assertEqual(sparse.shape, (2, 16, 4, 5))
        self.assertEqual(kappa.shape, (2, 8, 4, 5))
        reference, _ = self.module.sparse_evidence(nn.functional.pad(x, (0, 1, 0, 1), mode='replicate'))
        torch.testing.assert_close(sparse, reference)

    def test_low_precision_energy_softmax_and_zero_gradients_stable(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = (torch.randn(2, 16, 7, 9) * 1000).to(dtype).requires_grad_()
                sparse, kappa = self.module.sparse_evidence(x)
                self.assertEqual(sparse.dtype, dtype)
                self.assertEqual(kappa.dtype, torch.float32)
                self.assertTrue(torch.isfinite(sparse).all())
                self.assertTrue(torch.isfinite(kappa).all())
                sparse.float().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                uniform = torch.ones(2, 16, 6, 8, dtype=dtype, requires_grad=True)
                sparse, _ = self.module.sparse_evidence(uniform)
                sparse.float().sum().backward()
                self.assertTrue(torch.isfinite(uniform.grad).all())

    def test_nonzero_alpha_gradients_reach_projection_and_input(self):
        module = SECD(16, 32, alpha_init=0.1).eval()
        x = torch.randn(2, 16, 6, 8, requires_grad=True)
        module(x).square().sum().backward()
        for gradient in (module.raw_alpha.grad, module.projection.conv.weight.grad, x.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_alpha_zero_gets_gradient_but_projection_initially_zero(self):
        x = torch.randn(2, 16, 6, 8, requires_grad=True)
        self.module(x).sum().backward()
        self.assertGreater(self.module.raw_alpha.grad.abs().item(), 0)
        self.assertEqual(self.module.projection.conv.weight.grad.abs().sum().item(), 0)

    def test_cpu_autocast_gradient_finite(self):
        module = SECD(16, 32, alpha_init=0.1).eval()
        x = torch.randn(2, 16, 6, 8, requires_grad=True)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = module(x)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(module.raw_alpha.grad))

    def test_bounded_alpha_and_defaults(self):
        self.assertEqual(self.module.alpha_eff.item(), 0)
        with torch.no_grad():
            self.module.raw_alpha.fill_(100)
        self.assertAlmostEqual(self.module.alpha_eff.item(), 0.2)
        with torch.no_grad():
            self.module.raw_alpha.fill_(-100)
        self.assertAlmostEqual(self.module.alpha_eff.item(), -0.2)

    def test_invalid_parameters_explicitly_rejected(self):
        for options in ({'groups': 3}, {'groups': 0}, {'temperature': 0},
                        {'eps': 0}, {'alpha_max': -1}, {'alpha_mode': 'linear'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SECD(16, 32, **options)


class PResNetCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _models(self, secd, **options):
        torch.manual_seed(19)
        original = OriginalPResNet(18, return_idx=[1, 2, 3], **options).eval()
        torch.manual_seed(19)
        modified = PResNet(18, return_idx=[1, 2, 3], SECD=secd, **options).eval()
        return original, modified

    def test_disabled_structure_keys_parameters_rng_and_exact_forward(self):
        original, modified = self._models({'enabled': False})
        self.assertEqual(list(original.state_dict()), list(modified.state_dict()))
        self.assertEqual(sum(p.numel() for p in original.parameters()),
                         sum(p.numel() for p in modified.parameters()))
        self.assertEqual([(n, type(m).__name__) for n, m in original.named_modules()],
                         [(n, type(m).__name__) for n, m in modified.named_modules()])
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, modified.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 3, 97, 83)
        with torch.no_grad():
            for expected, actual in zip(original(x), modified(x)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.manual_seed(21)
        OriginalPResNet(18)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(21)
        PResNet(18, SECD={'enabled': False})
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_all_alpha_zero_variants_exact_original_outputs(self):
        for transitions in (['3to4'], ['4to5'], ['3to4', '4to5']):
            for freeze_norm in (True, False):
                with self.subTest(transitions=transitions, freeze_norm=freeze_norm):
                    original, modified = self._models({'enabled': True, 'transitions': transitions},
                                                      freeze_norm=freeze_norm)
                    x = torch.randn(2, 3, 97, 83)
                    with torch.no_grad():
                        for expected, actual in zip(original(x), modified(x)):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_enabled_preserves_subsequent_encoder_decoder_rng_stream(self):
        torch.manual_seed(22)
        OriginalPResNet(18)
        expected_next = torch.rand(32)
        torch.manual_seed(22)
        PResNet(18, SECD={'enabled': True, 'transitions': ['3to4', '4to5']})
        torch.testing.assert_close(torch.rand(32), expected_next, rtol=0, atol=0)

    def test_channels_strides_and_bypass_projection(self):
        model = PResNet(18, return_idx=[1, 2, 3],
                        SECD={'enabled': True, 'transitions': ['3to4', '4to5']}).eval()
        self.assertEqual(model.out_channels, [128, 256, 512])
        self.assertEqual(model.out_strides, [8, 16, 32])
        self.assertEqual(model.secd_34.projection.conv.weight.shape, (256, 128, 1, 1))
        self.assertEqual(model.secd_45.projection.conv.weight.shape, (512, 256, 1, 1))
        with torch.no_grad():
            self.assertEqual([tuple(o.shape) for o in model(torch.randn(1, 3, 128, 128))],
                             [(1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)])

    def test_norm_matches_backbone_freeze_policy(self):
        for freeze_norm in (True, False):
            model = PResNet(18, freeze_norm=freeze_norm, SECD={'enabled': True})
            expected = COMMON.FrozenBatchNorm2d if freeze_norm else nn.BatchNorm2d
            self.assertIsInstance(model.secd_34.projection.norm, expected)

    def test_original_keys_preserved_and_only_secd_missing(self):
        original, modified = self._models({'enabled': True, 'transitions': ['3to4', '4to5']})
        state = original.state_dict()
        incompatible = modified.load_state_dict(state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(k.startswith(('secd_34.', 'secd_45.')) for k in incompatible.missing_keys))
        self.assertTrue(set(state).issubset(modified.state_dict()))
        self.assertTrue(any(k.startswith('res_layers.2.blocks.') for k in state))

    def test_original_pretrained_constructor_loads_bypass_only_missing(self):
        original = OriginalPResNet(18)
        with patch('torch.hub.load_state_dict_from_url', return_value=original.state_dict()) as download:
            model = PResNet(18, pretrained=True, SECD={'enabled': True})
        download.assert_called_once()
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)

    def test_pretrained_rejects_original_missing_and_unexpected(self):
        state = OriginalPResNet(18).state_dict()
        missing = dict(state)
        del missing['res_layers.2.blocks.0.branch2a.conv.weight']
        unexpected = dict(state, bad_original_key=torch.zeros(1))
        for invalid in (missing, unexpected):
            with self.subTest(keys=list(invalid)[-1:]), \
                    patch('torch.hub.load_state_dict_from_url', return_value=invalid), \
                    self.assertRaisesRegex(RuntimeError, 'backbone keys mismatch'):
                PResNet(18, pretrained=True, SECD={'enabled': True})

    def test_nonzero_bypasses_receive_previous_stage_features(self):
        model = PResNet(18, SECD={'enabled': True, 'transitions': ['3to4', '4to5'],
                                'alpha_init': 0.1}).eval()
        captured = {}
        hooks = []
        for index in (1, 2):
            def capture(module, inputs, output, index=index):
                output.retain_grad()
                captured[index] = output
            hooks.append(model.res_layers[index].register_forward_hook(capture))
        model(torch.randn(2, 3, 64, 64))[-1].square().mean().backward()
        for bypass in (model.secd_34, model.secd_45):
            for gradient in (bypass.raw_alpha.grad, bypass.projection.conv.weight.grad):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
        for feature in captured.values():
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        for hook in hooks:
            hook.remove()

    def test_freeze_at_also_freezes_bypass_target_stage(self):
        model = PResNet(18, freeze_at=3, SECD={'enabled': True,
                                            'transitions': ['3to4', '4to5']})
        self.assertTrue(all(not p.requires_grad for p in model.secd_34.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.secd_45.parameters()))

    def test_invalid_transitions_and_unavailable_stages_rejected(self):
        for transitions in ([], ['2to3'], ['3to4', '3to4'], '3to4'):
            with self.subTest(transitions=transitions), self.assertRaises(ValueError):
                PResNet(18, SECD={'enabled': True, 'transitions': transitions})
        with self.assertRaises(ValueError):
            PResNet(18, num_stages=3, return_idx=[1, 2],
                    SECD={'enabled': True, 'transitions': ['4to5']})


if __name__ == '__main__':
    unittest.main()
