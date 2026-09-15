"""CTER math, independent switches, MERT regression and inference tests.

Run: python -m unittest tests.test_cter tests.test_mert -v
"""

import copy
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing test support (including optional dependency stubs), not
# the implementation of MERT or its loss.
from tests.test_mert import PROJECT_DIR, DesiredQueryMatcher, make_target as _make_target
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.misc.amp import autocast_context
from src.misc.inference_audit import InferenceAudit
from src.solver.cter_loss import (
    CTERLoss, CTERTrainingPlugin, context_ring_occupancy,
    evidence_relay_loss, soft_cell_occupancy, target_evidence_margin,
)
from src.solver.det_engine import evaluate, train_one_epoch
from src.solver.mert import MERTTrainingPlugin, MicroShiftPairGenerator
from src.solver.solver import BaseSolver
from src.zoo.rtdetr.rtdetr import RTDETR


CTER_CONFIG = {
    'enabled': True, 'transitions': ['3to4', '4to5'],
    'context_scale': 1.5, 'eps': 1e-6,
    'hard_negative': {'temperature': 0.1}, 'relay_ratio': 0.9,
    'occupancy': {'mode': 'soft_overlap'},
    'stage_validity': {'enabled': True}, 'loss_weight': 0.05, 'debug': False,
}


def make_target(*args, **kwargs):
    target = _make_target(*args, **kwargs)
    # Real DataLoader targets contain tensor values; the shared matcher helper
    # also permits an integer for direct math tests, but the engine uses .to().
    target.pop('desired_query')
    return target


def config_path(suffix):
    return str(PROJECT_DIR / 'configs' / 'rtdetr' /
               ('rtdetr_r18vd_200e_dut_anti_uav_' + suffix + '.yml'))


class TestCTERMath(unittest.TestCase):
    def test_soft_occupancy_full_half_and_subcell(self):
        boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 2., 4.], [1., 1., 2., 2.]])
        occupancy = soft_cell_occupancy(boxes, (16, 16), (4, 4), stride=4)
        self.assertAlmostEqual(float(occupancy[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(occupancy[1, 0, 0]), 0.5)
        self.assertAlmostEqual(float(occupancy[2].sum()), 1 / 16)

    def test_context_ring_excludes_target_and_other_gt_exact_union(self):
        boxes = torch.tensor([[4., 4., 8., 8.], [6., 6., 10., 10.]])
        background = context_ring_occupancy(boxes[0], boxes, (16, 16), (16, 16), 1, 2.)
        self.assertTrue(torch.equal(background[4:8, 4:8], torch.zeros(4, 4)))
        self.assertTrue(torch.equal(background[6:10, 6:10], torch.zeros(4, 4)))
        # Context=64, GT union=16+16-4, free area=36.
        self.assertAlmostEqual(float(background.sum()), 36.)

    def test_fractional_context_does_not_drop_complete_target_cell(self):
        boxes = torch.tensor([[3., 3., 5., 5.]])
        positive = soft_cell_occupancy(boxes, (8, 8), (1, 1), 8)
        background = context_ring_occupancy(boxes[0], boxes, (8, 8), (1, 1), 8, 2.)
        self.assertAlmostEqual(float(positive.sum()), 4 / 64)
        self.assertAlmostEqual(float(background.sum()), (16 - 4) / 64)

    def test_identical_target_background_margin_is_zero(self):
        normalized = torch.zeros(2, 3, 3)
        normalized[0] = 1.
        positive = torch.zeros(3, 3)
        positive[1, 1] = 1.
        background = 1 - positive
        margin, occupancy, valid = target_evidence_margin(normalized, positive, background)
        self.assertLess(abs(float(margin)), 2e-6)
        self.assertEqual(float(occupancy), 1.)
        self.assertTrue(bool(valid))

    def test_opposite_target_background_increases_margin(self):
        normalized = torch.zeros(2, 3, 3)
        normalized[0] = -1.
        normalized[0, 1, 1] = 1.
        positive = torch.zeros(3, 3)
        positive[1, 1] = 1.
        margin, _, valid = target_evidence_margin(normalized, positive, 1 - positive)
        self.assertGreater(float(margin), 1.9)
        self.assertTrue(bool(valid))

    def test_logsumexp_emphasizes_the_hard_negative(self):
        normalized = torch.tensor([[[1., 1., -1.]], [[0., 0., 0.]]])
        positive = torch.tensor([[1., 0., 0.]])
        background = torch.tensor([[0., 1., 1.]])
        margin, _, _ = target_evidence_margin(normalized, positive, background)
        # A hard positive-like background dominates; a mean would yield margin~1.
        self.assertLess(float(margin), 0.1)

    def test_relay_satisfied(self):
        loss = evidence_relay_loss(torch.tensor([1.]), torch.tensor([0.95]))
        self.assertEqual(float(loss), 0.)

    def test_relay_violated(self):
        loss = evidence_relay_loss(torch.tensor([1.]), torch.tensor([0.5]))
        self.assertAlmostEqual(float(loss), 0.16, places=6)

    def test_stop_gradient_reference(self):
        source = torch.tensor([1.], requires_grad=True)
        destination = torch.tensor([0.5], requires_grad=True)
        evidence_relay_loss(source, destination).sum().backward()
        self.assertIsNone(source.grad)
        self.assertLess(float(destination.grad), 0.)

    def test_empty_background_stays_finite_and_invalid(self):
        normalized = F.normalize(torch.randn(4, 3, 3), dim=0)
        margin, _, valid = target_evidence_margin(normalized, torch.ones(3, 3), torch.zeros(3, 3))
        self.assertTrue(bool(torch.isfinite(margin)))
        self.assertFalse(bool(valid))

    def test_partial_border_cell_uses_actual_clipped_cell_area(self):
        occupancy = soft_cell_occupancy(torch.tensor([[8., 8., 10., 10.]]),
                                        (10, 10), (3, 3), stride=4)
        self.assertAlmostEqual(float(occupancy[0, 2, 2]), 1.)

    def test_debug_stats_keys_exist_even_on_empty_gt_rank(self):
        config = dict(CTER_CONFIG, debug=True, transitions=['3to4'])
        features = [torch.randn(1, 4, size, size, requires_grad=True) for size in (16, 8, 4)]
        empty = CTERTrainingPlugin(config, [1, 2, 4], ['s3', 's4', 's5'])
        nonempty = CTERTrainingPlugin(config, [1, 2, 4], ['s3', 's4', 's5'])
        empty.calculate_loss(features, [{'boxes': torch.empty(0, 4)}], (16, 16))
        nonempty.calculate_loss(features, [make_target()], (16, 16))
        self.assertEqual(set(empty.debug_sums), set(nonempty.debug_sums))
        summary = nonempty.debug_summary()
        self.assertIn('margin_s5_mean', summary)
        self.assertIn('relay_ratio_34_observed', summary)

    def test_ddp_denominator_empty_rank_also_participates(self):
        features = [torch.randn(1, 4, size, size, requires_grad=True) for size in (8, 4, 2)]
        criterion = CTERLoss(CTER_CONFIG, [1, 2, 4])
        with mock.patch('src.solver.cter_loss.tdist.is_available', return_value=True), \
                mock.patch('src.solver.cter_loss.tdist.is_initialized', return_value=True), \
                mock.patch('src.solver.cter_loss.tdist.get_world_size', return_value=3), \
                mock.patch('src.solver.cter_loss.tdist.all_reduce') as reduce:
            loss, _ = criterion(features, [{'boxes': torch.empty(0, 4)}], (8, 8))
        self.assertEqual(reduce.call_count, 1)
        self.assertEqual(float(loss), 0.)

    def test_empty_gt_zero_loss_still_has_graph(self):
        features = [torch.randn(1, 4, size, size, requires_grad=True) for size in (8, 4, 2)]
        loss, stats = CTERLoss(CTER_CONFIG, [1, 2, 4])(
            features, [{'boxes': torch.empty(0, 4)}], (8, 8)
        )
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats, {})
        loss.backward()

    def test_debug_false_uses_no_stats_cpu_sync(self):
        features = [torch.randn(1, 4, size, size, requires_grad=True) for size in (16, 8, 4)]
        plugin = CTERTrainingPlugin(CTER_CONFIG, [1, 2, 4], ['s3', 's4', 's5'])
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unexpected cpu copy')), \
                mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('unexpected item')), \
                mock.patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('unexpected tolist')):
            loss = plugin.calculate_loss(features, [make_target()], (16, 16))
            loss.backward()
            self.assertEqual(plugin.debug_summary(), {})


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.out_strides = [1, 2, 4]
        self.return_idx = [1, 2, 3]

    def forward(self, images):
        feature = self.conv(images).relu() + 0.1
        return [feature, F.avg_pool2d(feature, 2), F.avg_pool2d(feature, 4)]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, features):
        return [self.conv(feature) for feature in features]


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features, targets=None):
        feature = features[0]
        batch_size = feature.shape[0]
        signal = feature.mean(dim=(1, 2, 3)) * self.scale * 0.01
        base = feature.new_tensor([0.5, 0.5, 0.1, 0.1])[None, None].expand(batch_size, 3, 4)
        layers = [base + signal[:, None, None] * factor for factor in (1., 1.1, 1.2)]
        logits = signal[:, None, None].expand(batch_size, 3, 1)
        return {
            'pred_boxes': layers[-1], 'pred_logits': logits,
            'aux_outputs': [
                {'pred_logits': logits, 'pred_boxes': layers[0]},
                {'pred_logits': logits, 'pred_boxes': layers[1]},
                {'pred_logits': logits, 'pred_boxes': base},
            ],
        }


class TinyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = DesiredQueryMatcher()

    def forward(self, outputs, targets):
        return {'loss_det': outputs['pred_boxes'].square().mean() + outputs['pred_logits'].square().mean()}


def tiny_detector():
    return RTDETR(TinyBackbone(), TinyEncoder(), TinyDecoder(), multi_scale=[16])


class TestCTERIntegration(unittest.TestCase):
    def test_configs_preserve_training_policy_and_mert_exactly(self):
        late = load_config(config_path('mert_v2_exp4_late_xywh'), {})
        for suffix in ('cter_34', 'cter_45', 'cter_345', 'cter_mert_late_xywh',
                       'cter_34_mert_late_xywh', 'cter_45_mert_late_xywh'):
            config = load_config(config_path(suffix), {})
            self.assertTrue(config['CTER']['enabled'])
            for key in ('RTDETR', 'PResNet', 'HybridEncoder', 'RTDETRTransformer',
                        'optimizer', 'lr_scheduler', 'epoches', 'train_dataloader', 'val_dataloader'):
                self.assertEqual(config[key], late[key], key)
            if 'mert' in suffix:
                self.assertEqual(config['MERT'], late['MERT'])
            else:
                self.assertFalse(config['MERT']['enabled'])
        # Independent loads must not retain previous CTER switches.
        load_config(config_path('cter_345'))
        fresh = load_config(config_path('mert_v2_exp4_late_xywh'))
        self.assertNotIn('CTER', fresh)
        baseline = load_config(config_path('mert_v2_exp0_baseline'), {})
        for key in ('RTDETR', 'PResNet', 'HybridEncoder', 'RTDETRTransformer',
                    'optimizer', 'lr_scheduler', 'epoches', 'train_dataloader', 'val_dataloader'):
            self.assertEqual(baseline[key], late[key], key)

    def test_four_switch_modes_forward_and_shift_counts(self):
        late = load_config(config_path('mert_v2_exp4_late_xywh'), {})['MERT']
        images, targets = torch.randn(1, 3, 16, 16), [make_target()]
        for enable_cter, enable_mert in ((False, False), (True, False), (False, True), (True, True)):
            model, criterion = tiny_detector(), TinyCriterion()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            batches, shifts = [], []
            original_shift = MicroShiftPairGenerator.__call__

            def count_shift(generator, *args, **kwargs):
                shifts.append(1)
                return original_shift(generator, *args, **kwargs)

            handle = model.register_forward_pre_hook(lambda _module, inputs: batches.append(inputs[0].shape[0]))
            mert_config = copy.deepcopy(late)
            mert_config['enabled'] = enable_mert
            cter_config = dict(CTER_CONFIG, enabled=enable_cter)
            with mock.patch.object(MicroShiftPairGenerator, '__call__', new=count_shift):
                stats = train_one_epoch(model, criterion, [(images, targets)], optimizer,
                                        torch.device('cpu'), 0, mert_config=mert_config,
                                        cter_config=cter_config)
            handle.remove()
            self.assertEqual(batches, [2 if enable_mert else 1])
            self.assertEqual(len(shifts), int(enable_mert))
            self.assertEqual('loss_cter' in stats, enable_cter)
            self.assertEqual('loss_mert' in stats, enable_mert)

    def test_cter_disabled_matches_original_detection_mert_step(self):
        late = load_config(config_path('mert_v2_exp4_late_xywh'), {})['MERT']
        initial = tiny_detector().state_dict()
        images, targets = torch.randn(1, 3, 16, 16), [make_target()]
        for enabled in (False, True):
            config = dict(late, enabled=enabled)
            reference, integrated = tiny_detector(), tiny_detector()
            reference.load_state_dict(initial)
            integrated.load_state_dict(initial)
            criterion = TinyCriterion()
            optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            reference.train()
            torch.manual_seed(0)
            np.random.seed(0)
            if enabled:
                plugin = MERTTrainingPlugin(config, criterion.matcher)
                pair = plugin.prepare(reference, images, targets)
                with plugin.disable_internal_multiscale(reference):
                    outputs = reference(torch.cat([pair['original_images'], pair['shifted_images']]),
                                        pair['original_targets'] + pair['shifted_targets'])
                losses = criterion(outputs, pair['original_targets'] + pair['shifted_targets'])
                original, shifted = plugin.split_concatenated_outputs(outputs, 1)
                losses['loss_mert'] = plugin.calculate_loss(original, shifted, pair)
            else:
                losses = criterion(reference(images, targets), targets)
            expected = sum(losses.values())
            expected.backward()
            optimizer.step()
            torch.manual_seed(0)
            np.random.seed(0)
            captured_pairs = []
            original_prepare = MERTTrainingPlugin.prepare

            def capture_prepare(plugin_instance, *args):
                actual_pair = original_prepare(plugin_instance, *args)
                captured_pairs.append(actual_pair)
                return actual_pair

            with mock.patch.object(MERTTrainingPlugin, 'prepare', new=capture_prepare):
                stats = train_one_epoch(integrated, TinyCriterion(), [(images, targets)],
                                        torch.optim.SGD(integrated.parameters(), lr=0.01),
                                        torch.device('cpu'), 0, mert_config=config,
                                        cter_config={'enabled': False})
            if enabled:
                actual_pair = captured_pairs[0]
                for key in ('original_images', 'shifted_images', 'shifts'):
                    self.assertTrue(torch.equal(pair[key], actual_pair[key]), key)
                for key in ('original_targets', 'shifted_targets'):
                    for expected_target, actual_target in zip(pair[key], actual_pair[key]):
                        for field in ('boxes', 'origin_gt_id', 'mert_fully_visible'):
                            self.assertTrue(torch.equal(expected_target[field], actual_target[field]), field)
            else:
                self.assertEqual(captured_pairs, [])
            self.assertAlmostEqual(stats['loss'], float(expected), places=7)
            self.assertEqual(set(losses), set(stats) - {'loss', 'lr'})
            for name, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, integrated.state_dict()[name]), name)

    def test_cter_loss_has_no_direct_encoder_decoder_gradient(self):
        model = tiny_detector().train()
        outputs, features, size = model(torch.randn(1, 3, 16, 16), [make_target()], True)
        plugin = CTERTrainingPlugin(CTER_CONFIG, model.backbone.out_strides, ['s3', 's4', 's5'])
        plugin.calculate_loss(features, [make_target()], size).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))
        self.assertIsNotNone(model.backbone.conv.weight.grad)

    def test_real_r18_four_configs_parameter_keys_and_eval_outputs_match(self):
        images = torch.randn(1, 3, 128, 128)
        reference_state, reference_output, parameter_count, reference_keys = None, None, None, None
        for suffix in ('mert_v2_exp0_baseline', 'mert_v2_exp4_late_xywh',
                       'cter_345', 'cter_mert_late_xywh'):
            config = YAMLConfig(config_path(suffix))
            config.yaml_cfg['PResNet']['pretrained'] = False
            config.yaml_cfg['HybridEncoder']['eval_spatial_size'] = None
            config.yaml_cfg['RTDETRTransformer']['eval_spatial_size'] = None
            model = config.model.eval()
            count = sum(parameter.numel() for parameter in model.parameters())
            self.assertEqual(model.backbone.out_strides, [8, 16, 32])
            self.assertFalse(hasattr(model, 'cter'))
            self.assertFalse(hasattr(model, 'mert'))
            if reference_state is None:
                reference_state = copy.deepcopy(model.state_dict())
                reference_keys = list(reference_state)
                parameter_count = count
            else:
                self.assertEqual(count, parameter_count)
                self.assertEqual(list(model.state_dict()), reference_keys)
                model.load_state_dict(reference_state, strict=True)
            with torch.no_grad(), InferenceAudit(model) as audit:
                output = model(images)
            self.assertEqual(audit.counts, {'model': 1, 'backbone': 1, 'encoder': 1, 'decoder': 1})
            if reference_output is None:
                reference_output = output
                # Compare with the original baseline's direct backbone ->
                # encoder -> decoder path, bypassing the optional return API.
                with torch.no_grad():
                    direct = model.decoder(model.encoder(model.backbone(images)))
                for key in ('pred_logits', 'pred_boxes'):
                    self.assertTrue(torch.equal(output[key], direct[key]), key)
            else:
                for key in ('pred_logits', 'pred_boxes'):
                    self.assertTrue(torch.equal(output[key], reference_output[key]), suffix + ':' + key)
        self.assertEqual(sum(parameter.numel() for parameter in CTERLoss(CTER_CONFIG, [8, 16, 32]).parameters()), 0)

    def test_eval_one_forward_and_no_training_loss_initialization(self):
        class FakeCOCOEvaluator:
            def __init__(self, *_args):
                self.coco_eval = {'bbox': type('Eval', (), {'stats': np.zeros(12)})()}
            def update(self, *_args): pass
            def synchronize_between_processes(self): pass
            def accumulate(self): pass
            def summarize(self): pass

        class FakePostprocessor:
            iou_types = ('bbox',)
            def __call__(self, outputs, sizes):
                return [{} for _ in range(outputs['pred_boxes'].shape[0])]

        model = tiny_detector()
        with mock.patch('src.solver.det_engine.CocoEvaluator', FakeCOCOEvaluator), \
                mock.patch('src.solver.det_engine.MERTTrainingPlugin', side_effect=AssertionError('MERT in eval')), \
                mock.patch('src.solver.det_engine.CTERTrainingPlugin', side_effect=AssertionError('CTER in eval')):
            stats, _ = evaluate(model, TinyCriterion(), FakePostprocessor(),
                                [(torch.randn(1, 3, 16, 16), [make_target()])],
                                None, torch.device('cpu'), None, debug_eval_amp=True)
        self.assertIn('coco_eval_bbox', stats)
        self.assertFalse(model.training)
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model.backbone._forward_hooks)

    def test_eval_solver_does_not_access_scaler(self):
        class EvaluationConfig:
            device = torch.device('cpu')
            last_epoch = -1
            find_unused_parameters = False
            sync_bn = False
            tuning = None
            resume = None
            ema = None
            model = tiny_detector()
            criterion = TinyCriterion()
            postprocessor = None
            val_dataloader = type('Loader', (), {'shuffle': False})()
            @property
            def scaler(self):
                raise AssertionError('Evaluation must not instantiate/access GradScaler')

        with tempfile.TemporaryDirectory() as directory:
            config = EvaluationConfig()
            config.output_dir = directory
            solver = BaseSolver(config)
            solver.eval()
            self.assertIsNone(solver.scaler)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA dtype verification requires GPU')
    def test_cuda_actual_autocast_dtype(self):
        model = tiny_detector().cuda().eval()
        with torch.no_grad(), InferenceAudit(model) as audit:
            with autocast_context(torch.device('cuda'), True):
                model(torch.randn(1, 3, 16, 16, device='cuda'))
        audit.report(expected_amp=True)
        self.assertEqual(audit.values['first Conv output dtype'], 'torch.float16')


if __name__ == '__main__':
    unittest.main()
