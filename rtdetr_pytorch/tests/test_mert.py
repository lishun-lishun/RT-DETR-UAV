"""CPU tests for the training-only MERT plugin and original matcher/criterion.

Run: python -m unittest tests.test_mert -v
Optional COCO/RegNet dependencies are stubbed only by tests._support; the
actual detector, decoder, Hungarian matcher, and losses are not stubbed.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests._support import prepare_imports

prepare_imports()

from src.solver.mert import (MERT, average_loss_dicts, decoder_box_trajectory,
                             disable_internal_multiscale, inverse_box_shift,
                             shift_images, split_concatenated_outputs)
from src.zoo.rtdetr.matcher import HungarianMatcher
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer


class TinyDetector(nn.Module):
    def __init__(self, multi_scale=None):
        super().__init__()
        self.multi_scale = multi_scale
        self.weight = nn.Parameter(torch.ones(()))


def make_target(boxes, ids=None, image_id=11):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    result = {'boxes': boxes, 'labels': torch.zeros(len(boxes), dtype=torch.long),
              'image_id': torch.tensor([image_id]), 'orig_size': torch.tensor([20, 10]),
              'size': torch.tensor([20, 10]), 'area': torch.ones(len(boxes)),
              'iscrowd': torch.zeros(len(boxes), dtype=torch.long)}
    if ids is not None:
        result['origin_gt_id'] = torch.as_tensor(ids, dtype=torch.long)
    return result


def make_outputs(trajectory, encoder_value=0.99, dn_value=0.01):
    """Wrap [L,B,Q,4] exactly as the original training decoder does."""
    layers, batch_size, queries, _ = trajectory.shape
    logits = torch.zeros(batch_size, queries, 1)
    output = {'pred_boxes': trajectory[-1], 'pred_logits': logits.clone()}
    output['aux_outputs'] = [{'pred_boxes': trajectory[i], 'pred_logits': logits.clone()}
                             for i in range(layers - 1)]
    output['aux_outputs'].append({'pred_boxes': torch.full_like(trajectory[0], encoder_value),
                                  'pred_logits': logits.clone()})
    output['dn_aux_outputs'] = [{'pred_boxes': torch.full_like(trajectory[0], dn_value),
                                 'pred_logits': logits.clone()}]
    return output


def single_pair_matcher():
    return Mock(side_effect=[[(torch.tensor([0]), torch.tensor([0]))],
                             [(torch.tensor([0]), torch.tensor([0]))]])


class MERTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        self.plugin = MERT({'enabled': True, 'small_object_weighting': {'enabled': False}})
        self.model = TinyDetector().train()
        self.samples = torch.arange(200, dtype=torch.float32).reshape(1, 1, 10, 20)

    def batch(self, target=None, shift=(1, -1), samples=None):
        if target is None:
            target = make_target([[0.5, 0.5, 0.2, 0.2]], ids=[42])
        return self.plugin.prepare(self.samples if samples is None else samples,
                                   [target], self.model, shifts=[shift])

    def test_shift_pixels_zero_fill_no_wrap_all_directions(self):
        image = torch.arange(1, 13).reshape(1, 1, 3, 4).float()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not dx and not dy:
                    continue
                moved = shift_images(image, torch.tensor([[dx, dy]]))
                for y in range(3):
                    for x in range(4):
                        sx, sy = x - dx, y - dy
                        expected = image[0, 0, sy, sx] if 0 <= sx < 4 and 0 <= sy < 3 else 0
                        self.assertEqual(moved[0, 0, y, x], expected)

    def test_gt_shift_inverse_and_original_not_mutated(self):
        target = make_target([[0.5, 0.5, 0.2, 0.2]], ids=[103])
        before = {key: value.clone() for key, value in target.items()}
        batch = self.batch(target)
        torch.testing.assert_close(batch.shifted_targets[0]['boxes'],
                                   torch.tensor([[0.55, 0.4, 0.2, 0.2]]))
        restored = inverse_box_shift(batch.shifted_targets[0]['boxes'],
                                     batch.shifts[0], batch.spatial_size)
        torch.testing.assert_close(restored, target['boxes'])
        self.assertEqual(batch.targets[0]['origin_gt_id'].tolist(), [103])
        self.assertEqual(batch.shifted_targets[0]['origin_gt_id'].tolist(), [103])
        for key in before:
            torch.testing.assert_close(target[key], before[key])
        self.assertNotIn('fully_visible', target)
        # Predictions are never clamped during inverse alignment.
        box = torch.tensor([[0.01, 0.02, 0.2, 0.2]], requires_grad=True)
        inverse = inverse_box_shift(box, torch.tensor([1, 1]), (10, 20))
        self.assertLess(inverse[0, 0].item(), 0)
        inverse.sum().backward()
        torch.testing.assert_close(box.grad, torch.ones_like(box))

    def test_ids_stable_after_clip_and_border_mask(self):
        target = make_target([[0.5, 0.5, 0.2, 0.2],
                              [0.94, 0.5, 0.12, 0.2],
                              [0.99, 0.5, 0.02, 0.2]], ids=[9, 3, 77])
        batch = self.batch(target, shift=(1, 0))
        original, moved = batch.targets[0], batch.shifted_targets[0]
        self.assertEqual(original['origin_gt_id'].tolist(), [9, 3, 77])
        self.assertEqual(moved['origin_gt_id'].tolist(), [9, 3])
        self.assertEqual(moved['fully_visible'].tolist(), [True, False])
        self.assertEqual(len(original['boxes']), 3)  # Original detection GTs never removed.
        self.assertEqual(len(moved['boxes']), 2)  # Clipped GT still detection-supervised.
        torch.testing.assert_close(moved['boxes'][1], torch.tensor([0.965, 0.5, 0.07, 0.2]))
        for key in ('labels', 'area', 'iscrowd'):
            self.assertEqual(len(moved[key]), 2)
        for key in ('image_id', 'orig_size', 'size'):
            torch.testing.assert_close(moved[key], target[key])

    def test_auto_ids_and_duplicate_ids_rejected(self):
        target = make_target([[0.2, 0.5, 0.1, 0.1], [0.8, 0.5, 0.1, 0.1]])
        batch = self.batch(target)
        self.assertEqual(batch.targets[0]['origin_gt_id'].tolist(), [0, 1])
        self.assertEqual(batch.shifted_targets[0]['origin_gt_id'].tolist(), [0, 1])
        target['origin_gt_id'] = torch.tensor([3, 3])
        with self.assertRaises(ValueError):
            self.batch(target)

    def test_sanitize_keeps_stale_metadata_but_mert_realigns_it(self):
        from torchvision import datapoints
        from torchvision.transforms.v2 import SanitizeBoundingBox

        # The original transform filters boxes/labels, but not area/iscrowd.
        # Retain the third annotation, not the first: truncation is incorrect.
        target = {
            'boxes': datapoints.BoundingBox(
                [[0, 0, 0, 2], [2, 2, 3, 2], [8, 4, 12, 6],
                 [1, 1, 1, 1], [0, 0, .5, .5]],
                format='XYXY', spatial_size=(10, 20), dtype=torch.float32),
            'labels': torch.zeros(5, dtype=torch.long),
            'area': torch.tensor([101., 102., 103., 104., 105.]),
            'iscrowd': torch.zeros(5, dtype=torch.long),
            'image_id': torch.tensor([11]), 'size': torch.tensor([20, 10]),
            'orig_size': torch.tensor([20, 10]),
        }
        _, target = SanitizeBoundingBox()(torch.zeros(3, 10, 20), target)
        self.assertEqual(len(target['boxes']), 1)
        self.assertEqual(len(target['labels']), 1)
        self.assertEqual(len(target['area']), 5)
        self.assertEqual(len(target['iscrowd']), 5)
        from torchvision.ops import box_convert
        target['boxes'] = box_convert(target['boxes'], 'xyxy', 'cxcywh') / \
            torch.tensor([20., 10., 20., 10.])
        before = {key: value.clone() for key, value in target.items()}
        batch = self.batch(target, shift=(1, 0))
        for view in (batch.targets[0], batch.shifted_targets[0]):
            torch.testing.assert_close(view['area'], torch.tensor([8.]))
            self.assertEqual(view['iscrowd'].tolist(), [0])
            self.assertEqual(view['origin_gt_id'].tolist(), [0])
        for key, value in before.items():
            torch.testing.assert_close(target[key], value)

    def test_stale_metadata_with_empty_targets_is_safe(self):
        target = make_target([])
        target['area'] = torch.arange(5, dtype=torch.float32)
        target['iscrowd'] = torch.zeros(5, dtype=torch.long)
        batch = self.batch(target)
        for view in (batch.targets[0], batch.shifted_targets[0]):
            for key in ('boxes', 'labels', 'area', 'iscrowd', 'origin_gt_id', 'fully_visible'):
                self.assertEqual(view[key].shape[0], 0, key)

    def test_stale_metadata_rebuilt_before_shifted_box_filter(self):
        target = make_target([[0.5, 0.5, 0.2, 0.2],
                              [0.94, 0.5, 0.12, 0.2],
                              [0.99, 0.5, 0.02, 0.2]], ids=[9, 3, 77])
        target['area'] = torch.arange(5, dtype=torch.float32)
        target['iscrowd'] = torch.zeros(5, dtype=torch.long)
        batch = self.batch(target, shift=(1, 0))
        torch.testing.assert_close(batch.targets[0]['area'], torch.tensor([8., 4.8, .8]))
        self.assertEqual(batch.shifted_targets[0]['origin_gt_id'].tolist(), [9, 3])
        torch.testing.assert_close(batch.shifted_targets[0]['area'], torch.tensor([8., 2.8]))
        self.assertEqual(batch.shifted_targets[0]['iscrowd'].tolist(), [0, 0])

    def test_essential_mismatched_fields_are_not_truncated(self):
        for key, value in (
            ('labels', torch.zeros(5, dtype=torch.long)),
            ('masks', torch.zeros(5, 10, 20, dtype=torch.bool)),
            ('fully_visible', torch.ones(5, dtype=torch.bool)),
            ('origin_gt_id', torch.arange(5)),
        ):
            with self.subTest(field=key):
                target = make_target([[.5, .5, .2, .2]])
                target[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self.batch(target)

    def test_stale_nonzero_crowd_flags_cannot_be_guessed(self):
        target = make_target([[.5, .5, .2, .2]])
        target['iscrowd'] = torch.tensor([0, 1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'iscrowd'):
            self.batch(target)

    def test_masks_shift_with_gt_and_sampled_resolution(self):
        target = make_target([[0.5, 0.5, 0.2, 0.2]])
        target['masks'] = torch.zeros(1, 5, 10, dtype=torch.bool)
        target['masks'][0, 2, 4] = True
        batch = self.batch(target, shift=(1, 0))
        resized = F.interpolate(target['masks'][:, None].float(), (10, 20), mode='nearest').bool()
        expected = shift_images(resized, torch.tensor([[1, 0]]))[:, 0]
        torch.testing.assert_close(batch.shifted_targets[0]['masks'], expected)
        torch.testing.assert_close(target['masks'].sum(), torch.tensor(1))

    def test_multiscale_sampled_before_one_pixel_shift_and_restored(self):
        self.model.multi_scale = [16, 32, 48]
        with patch('src.solver.mert.np.random.choice', return_value=32) as choose:
            batch = self.batch(shift=(1, 0))
        choose.assert_called_once_with(self.model.multi_scale)
        self.assertEqual(batch.samples.shape[-2:], (32, 32))
        torch.testing.assert_close(batch.shifted_samples[..., 1:], batch.samples[..., :-1])
        torch.testing.assert_close(batch.shifted_targets[0]['boxes'][0, 0], torch.tensor(0.5 + 1 / 32))
        wrapped = Mock(module=self.model)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with disable_internal_multiscale(wrapped):
                self.assertIsNone(self.model.multi_scale)
                raise RuntimeError('test')
        self.assertEqual(self.model.multi_scale, [16, 32, 48])

    def test_sampled_shifts_exclude_zero_and_invalid_explicit_shifts(self):
        samples = self.samples.expand(128, -1, -1, -1)
        targets = [make_target([[0.5, 0.5, 0.2, 0.2]]) for _ in range(128)]
        batch = self.plugin.prepare(samples, targets, self.model)
        self.assertTrue(bool((batch.shifts != 0).any(-1).all()))
        self.assertLessEqual(batch.shifts.abs().max(), 1)
        self.assertEqual(batch.shifts.unique(dim=0).shape[0], 8)
        for invalid in ([[0, 0]], [[2, 0]], [[0.5, 1]], [[1, 1], [0, 1]]):
            with self.assertRaises(ValueError):
                self.plugin.prepare(self.samples, [make_target([])], self.model, shifts=invalid)

    def test_disabled_and_eval_no_shift_resize_rng_or_model_state(self):
        keys = tuple(self.model.state_dict())
        parameter_count = sum(p.numel() for p in self.model.parameters())
        for plugin, model in ((MERT({'enabled': False}), self.model),
                              (self.plugin, self.model.eval())):
            with patch('src.solver.mert.shift_images', side_effect=AssertionError), \
                    patch('src.solver.mert.np.random.choice', side_effect=AssertionError), \
                    patch('src.solver.mert.torch.randint', side_effect=AssertionError):
                self.assertIsNone(plugin.prepare(self.samples, [make_target([])], model))
        self.assertNotIsInstance(self.plugin, nn.Module)
        self.assertEqual(keys, tuple(self.model.state_dict()))
        self.assertEqual(parameter_count, sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(self.plugin.calculate_loss({}, {}, None, Mock()), {})

    def test_trajectory_excludes_encoder_and_dn(self):
        trajectory = torch.rand(3, 2, 5, 4)
        output = make_outputs(trajectory)
        actual = decoder_box_trajectory(output)
        self.assertEqual(actual.shape, (3, 2, 5, 4))
        torch.testing.assert_close(actual, trajectory)
        single = {'pred_boxes': trajectory[-1]}
        self.assertEqual(decoder_box_trajectory(single).shape, (1, 2, 5, 4))

    def test_real_hungarian_different_query_and_gt_order_still_pair(self):
        target = make_target([[0.25, 0.4, 0.10, 0.12],
                              [0.75, 0.6, 0.18, 0.14]], ids=[101, 205])
        batch = self.batch(target, shift=(1, 0))
        base = target['boxes']
        increments = torch.tensor([[0.01, 0.02, 0.015, 0.012],
                                   [-0.02, 0.01, -0.01, 0.016]])
        original = torch.stack((base - increments * 2, base - increments, base))[:, None]
        shifted = original[:, :, [1, 0]].clone()
        shifted[..., 0] += 1 / 20
        # Reverse shifted GT order as well. The stable IDs carry the correspondence.
        moved = batch.shifted_targets[0]
        for key in ('boxes', 'labels', 'origin_gt_id', 'fully_visible', 'area', 'iscrowd'):
            moved[key] = moved[key].flip(0)
        matcher = HungarianMatcher({'cost_class': 0, 'cost_bbox': 1, 'cost_giou': 1},
                                    use_focal_loss=True)
        original_output, shifted_output = make_outputs(original), make_outputs(shifted)
        original_indices = matcher(original_output, batch.targets)[0]
        shifted_indices = matcher(shifted_output, batch.shifted_targets)[0]
        self.assertEqual(original_indices[0].tolist(), [0, 1])
        self.assertEqual(shifted_indices[0].tolist(), [0, 1])
        # Query 0 refers to GT 101 in original, but GT 205 in shifted.
        original_id = batch.targets[0]['origin_gt_id'][original_indices[1][0]]
        shifted_id = batch.shifted_targets[0]['origin_gt_id'][shifted_indices[1][0]]
        self.assertNotEqual(original_id.item(), shifted_id.item())
        loss = self.plugin.calculate_loss(original_output, shifted_output, batch, matcher)['loss_mert']
        self.assertLess(loss.item(), 1e-10)

    def test_unmatched_gt_and_invisible_gt_are_not_regularized(self):
        target = make_target([[0.5, 0.5, 0.2, 0.2], [0.94, 0.5, 0.12, 0.2]], ids=[9, 3])
        batch = self.batch(target, shift=(1, 0))
        trajectory = target['boxes'][None, None].expand(3, 1, 2, 4).clone().requires_grad_()
        shifted = trajectory.detach().clone()
        shifted[:, :, 0, 0] += 1 / 20
        shifted[0, :, 1] = 0.05  # Huge trajectory error on masked border object.
        shifted.requires_grad_()
        matcher = Mock(side_effect=[[(torch.tensor([0, 1]), torch.tensor([0, 1]))],
                                    [(torch.tensor([0, 1]), torch.tensor([0, 1]))]])
        loss = self.plugin.calculate_loss(make_outputs(trajectory), make_outputs(shifted),
                                           batch, matcher)['loss_mert']
        self.assertLess(loss.item(), 1e-10)
        # Neither view matches the same origin GT: differentiable zero.
        matcher = Mock(side_effect=[[(torch.tensor([0]), torch.tensor([0]))],
                                    [(torch.tensor([1]), torch.tensor([1]))]])
        loss = self.plugin.calculate_loss(make_outputs(trajectory), make_outputs(shifted),
                                           batch, matcher)['loss_mert']
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(trajectory.grad)
        self.assertIsNotNone(shifted.grad)

    def test_no_gt_or_no_transitions_returns_differentiable_zero(self):
        for target, layers in ((make_target([]), 3),
                               (make_target([[0.5, 0.5, 0.2, 0.2]]), 1)):
            batch = self.batch(target)
            original = torch.rand(layers, 1, 2, 4, requires_grad=True)
            shifted = torch.rand(layers, 1, 2, 4, requires_grad=True)
            matcher = Mock(side_effect=AssertionError('No matching required'))
            loss = self.plugin.calculate_loss(make_outputs(original), make_outputs(shifted),
                                               batch, matcher)['loss_mert']
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertIsNotNone(original.grad)
            self.assertIsNotNone(shifted.grad)
            matcher.assert_not_called()

    def test_all_shifted_gts_removed_preserves_original_supervision_and_zero(self):
        target = make_target([[0.99, 0.5, 0.02, 0.2]], ids=[7])
        batch = self.batch(target, shift=(1, 0))
        self.assertEqual(len(batch.targets[0]['boxes']), 1)
        self.assertEqual(len(batch.shifted_targets[0]['boxes']), 0)
        self.assertEqual(batch.targets[0]['origin_gt_id'].tolist(), [7])
        original = torch.rand(3, 1, 2, 4, requires_grad=True)
        shifted = torch.rand(3, 1, 2, 4, requires_grad=True)
        matcher = Mock(side_effect=AssertionError('No shifted GTs to match'))
        loss = self.plugin.calculate_loss(make_outputs(original), make_outputs(shifted),
                                           batch, matcher)['loss_mert']
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(original.grad)
        self.assertIsNotNone(shifted.grad)
        matcher.assert_not_called()

    def test_two_decoder_outputs_have_one_observable_transition(self):
        batch = self.batch(shift=(1, 0))
        original = torch.full((2, 1, 1, 4), 0.4)
        shifted = original.clone()
        shifted[1, ..., 0] += 0.02
        shifted[..., 0] += 1 / 20
        loss = self.plugin.calculate_loss(make_outputs(original), make_outputs(shifted),
                                           batch, single_pair_matcher())['loss_mert']
        torch.testing.assert_close(loss, torch.tensor((0.02 - 0.005) * 0.1))

    def test_late_two_transitions_coordinate_sum_and_xywh_weights(self):
        batch = self.batch(shift=(1, 0))
        original = torch.full((4, 1, 1, 4), 0.4, requires_grad=True)
        shifted = original.detach().clone()
        errors = torch.tensor([0.02, 0.03, 0.04, 0.05])
        shifted[0] += 0.3  # First transition deliberately enormous: excluded.
        shifted[2] += errors
        shifted[3] += errors * 3
        shifted[..., 0] += 1 / 20
        shifted.requires_grad_()
        loss = self.plugin.calculate_loss(make_outputs(original), make_outputs(shifted),
                                           batch, single_pair_matcher())['loss_mert']
        per_coordinate = F.smooth_l1_loss(torch.zeros(2, 4), torch.stack((errors, errors * 2)),
                                          beta=0.01, reduction='none')
        expected = (per_coordinate * torch.tensor([1, 1, 0.25, 0.25])).sum(-1).mean() * 0.1
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(shifted.grad[0], torch.zeros_like(shifted.grad[0]))
        self.assertGreater(shifted.grad[-1].abs().sum().item(), 0)

    def test_small_object_weighting_area_current_scale_clip_and_mean(self):
        plugin = MERT({'enabled': True})
        boxes = torch.tensor([[0.5, 0.5, 0.02, 0.02],
                               [0.5, 0.5, 0.08, 0.08],
                               [0.5, 0.5, 0.16, 0.16],
                               [0.5, 0.5, 0.5, 0.5]])
        # Areas 4,64,256,2500 pixels at 100x100 -> weights 4,2,1,1.
        torch.testing.assert_close(plugin._object_weights(boxes, (100, 100)),
                                   torch.tensor([4, 2, 1, 1], dtype=torch.float32))
        self.assertLess(plugin._object_weights(boxes, (200, 200))[1].item(), 2)
        target = make_target(boxes)
        batch = plugin.prepare(torch.zeros(1, 1, 100, 100), [target], self.model, shifts=[[1, 0]])
        original = boxes[None, None].expand(3, 1, 4, 4).clone()
        shifted = original.clone()
        shifted[1, ..., 0] += 0.02
        shifted[2, ..., 0] += 0.04
        shifted[..., 0] += 0.01
        indices = [(torch.arange(4), torch.arange(4))]
        matcher = Mock(side_effect=[indices, indices])
        loss = plugin.calculate_loss(make_outputs(original), make_outputs(shifted), batch, matcher)['loss_mert']
        expected = torch.tensor((0.02 - 0.005) * (4 + 2 + 1 + 1) / 4 * 0.10)
        torch.testing.assert_close(loss, expected)

    def test_concatenated_outputs_and_dn_metadata_split_without_mutation(self):
        trajectory = torch.rand(3, 4, 5, 4, requires_grad=True)
        output = make_outputs(trajectory)
        positive = tuple(torch.tensor([i]) for i in range(4))
        output['dn_meta'] = {'dn_positive_idx': positive, 'dn_num_group': 2,
                             'dn_num_split': [10, 5]}
        original, shifted = split_concatenated_outputs(output, 2)
        torch.testing.assert_close(decoder_box_trajectory(original), trajectory[:, :2])
        torch.testing.assert_close(decoder_box_trajectory(shifted), trajectory[:, 2:])
        self.assertEqual(len(original['dn_meta']['dn_positive_idx']), 2)
        self.assertEqual(original['dn_meta']['dn_positive_idx'][1].item(), 1)
        self.assertEqual(shifted['dn_meta']['dn_positive_idx'][0].item(), 2)
        self.assertEqual(len(output['dn_meta']['dn_positive_idx']), 4)
        self.assertEqual(shifted['dn_meta']['dn_num_group'], 2)
        self.assertEqual(shifted['dn_meta']['dn_num_split'], [10, 5])
        original['pred_boxes'].sum().backward()
        self.assertIsNotNone(trajectory.grad)

    def test_detection_losses_are_averaged_including_missing_dn_keys(self):
        original = {'loss_bbox': torch.tensor(2.0, requires_grad=True),
                     'loss_bbox_dn_0': torch.tensor(6.0, requires_grad=True)}
        shifted = {'loss_bbox': torch.tensor(4.0, requires_grad=True),
                    'loss_giou_aux_0': torch.tensor(8.0, requires_grad=True)}
        losses = average_loss_dicts(original, shifted)
        self.assertEqual(losses['loss_bbox'].item(), 3)
        self.assertEqual(losses['loss_bbox_dn_0'].item(), 3)
        self.assertEqual(losses['loss_giou_aux_0'].item(), 4)
        sum(losses.values()).backward()
        for values in (original, shifted):
            for value in values.values():
                self.assertEqual(value.grad.item(), 0.5)

    def test_real_decoder_matcher_criterion_concat_dn_backward(self):
        decoder = RTDETRTransformer(num_classes=1, hidden_dim=32, num_queries=8,
                                    feat_channels=[16, 16, 16], feat_strides=[8, 16, 32],
                                    num_decoder_layers=3, num_decoder_points=2,
                                    nhead=4, dim_feedforward=64, num_denoising=12).train()
        # Nonzero bbox heads expose a nontrivial refinement trajectory.
        for head in decoder.dec_bbox_head:
            nn.init.normal_(head.layers[-1].weight, std=0.01)
        matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                                    use_focal_loss=True)
        criterion = SetCriterion(matcher, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                                  ['vfl', 'boxes'], num_classes=1)
        samples = torch.randn(2, 3, 64, 64)
        targets = [make_target([[0.5, 0.5, 0.12, 0.14]]),
                   make_target([[0.3, 0.4, 0.08, 0.10], [0.7, 0.6, 0.14, 0.12]])]
        # Exercise the production augmentation metadata mismatch through real
        # DN generation, Hungarian matching, detection losses and backward.
        targets[0]['area'] = torch.arange(5, dtype=torch.float32)
        targets[0]['iscrowd'] = torch.zeros(5, dtype=torch.long)
        batch = self.plugin.prepare(samples, targets, self.model, shifts=[[1, 0], [0, -1]])
        self.assertEqual(batch.concatenated_samples.shape, (4, 3, 64, 64))
        features = [torch.randn(4, 16, size, size, requires_grad=True) for size in (8, 4, 2)]
        output = decoder(features, batch.concatenated_targets)
        original, shifted = split_concatenated_outputs(output, 2)
        self.assertEqual(decoder_box_trajectory(original).shape, (3, 2, 8, 4))
        for view in (original, shifted):
            self.assertEqual(len(view['dn_meta']['dn_positive_idx']), 2)
        losses = average_loss_dicts(criterion(original, batch.targets),
                                    criterion(shifted, batch.shifted_targets))
        losses.update(self.plugin.calculate_loss(original, shifted, batch, matcher))
        self.assertIn('loss_mert', losses)
        self.assertTrue(any('_dn_' in key for key in losses))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        self.assertTrue(all(feature.grad is not None for feature in features))
        self.assertGreater(sum(feature.grad.abs().sum().item() for feature in features), 0)
        self.assertIsNotNone(decoder.dec_bbox_head[-1].layers[-1].weight.grad)

    def test_mert_config_rejects_silent_protocol_changes(self):
        for options in ({'shift_pixels': 2}, {'trajectory_layers': 'all'},
                        {'forward_mode': 'invalid'}, {'schedule': {'enabled': True}},
                        {'beta': 0}, {'unknown_history_module': True}):
            with self.assertRaises(ValueError):
                MERT({'enabled': True, **options})


if __name__ == '__main__':
    unittest.main()
