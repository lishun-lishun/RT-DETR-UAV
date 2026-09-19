"""Training-only Micro-Shift Equivariant Refinement Training (MERT).

No module, parameter, buffer, or inference hook is added to the detector. Boxes
follow the original criterion's normalized ``cxcywh`` convention. For K late
transitions and N GTs matched in both views and fully visible in both views:

    loss_mert = loss_weight / (N*K) * sum_(g,k) w_g *
        [SL1(cx) + SL1(cy) + .25 * (SL1(w) + SL1(h))]

SL1 compares the two refinement deltas, not the absolute boxes. Coordinates
are summed; objects and transitions are averaged (there is no hidden /4).
Small-object weights use GT area at the sampled training resolution, and do
not change the denominator. Under DDP, N is the global pair count divided by
world size, matching the original criterion's gradient-normalization rule.
R18's three decoder outputs give two transitions.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MERTBatch:
    samples: torch.Tensor
    targets: list
    shifted_samples: torch.Tensor
    shifted_targets: list
    shifts: torch.Tensor  # [B, 2], integer (dx, dy) in sampled-image pixels
    spatial_size: Tuple[int, int]  # (height, width)

    @property
    def concatenated_samples(self):
        return torch.cat((self.samples, self.shifted_samples), dim=0)

    @property
    def concatenated_targets(self):
        return self.targets + self.shifted_targets


def _unwrap_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model


@contextmanager
def disable_internal_multiscale(model):
    """Temporarily prevent a second random resize, including through DDP.

    ``prepare`` already sampled the original multi_scale list with the same
    numpy RNG and nearest interpolation used by RTDETR.forward. Restore even
    when the forward fails; no persistent model change is made.
    """
    detector = _unwrap_model(model)
    if not hasattr(detector, 'multi_scale'):
        yield
        return
    previous = detector.multi_scale
    detector.multi_scale = None
    try:
        yield
    finally:
        detector.multi_scale = previous


def shift_images(samples, shifts):
    """Integer translation with zero-filled exposed pixels; never wraps."""
    if samples.ndim != 4 or shifts.shape != (samples.shape[0], 2):
        raise ValueError('Expected samples [B,C,H,W] and shifts [B,2].')
    result = torch.zeros_like(samples)
    height, width = samples.shape[-2:]
    for i, (dx, dy) in enumerate(shifts.detach().cpu().tolist()):
        dx, dy = int(dx), int(dy)
        if abs(dx) >= width or abs(dy) >= height:
            continue
        src_x, dst_x = max(-dx, 0), max(dx, 0)
        src_y, dst_y = max(-dy, 0), max(dy, 0)
        copy_w, copy_h = width - abs(dx), height - abs(dy)
        result[i, :, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
            samples[i, :, src_y:src_y + copy_h, src_x:src_x + copy_w]
    return result


def inverse_box_shift(boxes, shift, spatial_size):
    """Translate normalized cxcywh boxes back, without clipping predictions."""
    height, width = spatial_size
    shift = torch.as_tensor(shift, dtype=boxes.dtype, device=boxes.device)
    offset = boxes.new_zeros(4)
    offset[:2] = shift / boxes.new_tensor((width, height))
    return boxes - offset


def _xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def _cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), dim=-1)


def _fully_visible(corners):
    return ((corners >= 0) & (corners <= 1)).all(dim=-1) & \
        (corners[..., 2:] > corners[..., :2]).all(dim=-1)


def shift_targets(targets, shifts, spatial_size):
    """Keep all original supervision; clip shifted detection GT only.

    Partially cropped shifted GTs retain their IDs and detection supervision,
    but fully_visible=False excludes them from MERT. Entirely removed GTs are
    omitted from shifted detection targets. IDs are per-image, so (image in
    batch, origin_gt_id) uniquely identifies a GT throughout matching.
    """
    height, width = spatial_size
    originals, shifted = [], []
    instance_fields = {'labels', 'boxes', 'area', 'iscrowd', 'masks',
                       'keypoints', 'origin_gt_id', 'fully_visible'}
    for target, shift in zip(targets, shifts):
        original = dict(target)
        boxes = target['boxes']
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError('MERT requires normalized cxcywh target boxes [N,4].')
        count = len(boxes)
        if target['labels'].shape != (count,):
            raise ValueError('MERT target labels must have one entry per box.')
        if 'masks' in target and (target['masks'].ndim != 3 or
                                  target['masks'].shape[0] != count):
            raise ValueError('MERT target masks must have shape [N,H,W] aligned with boxes.')
        if 'fully_visible' in target and target['fully_visible'].shape != (count,):
            raise ValueError('MERT target fully_visible must have one entry per box.')

        # Original torchvision SanitizeBoundingBox filters boxes/labels/masks,
        # but can leave area/iscrowd at their pre-crop length. Reconstruct only
        # these auxiliary fields on this private view, BEFORE indexing by keep.
        # Never truncate: the surviving GT may not be the first annotation.
        if 'area' in original and original['area'].shape != (count,):
            original['area'] = boxes.float()[:, 2:].prod(-1) * width * height
        if 'iscrowd' in original and original['iscrowd'].shape != (count,):
            crowd = original['iscrowd']
            # The original CocoDetection converter excludes all crowd GTs.
            # Stale all-zero flags can therefore be rebuilt without guessing
            # instance correspondence; mixed/nonzero flags cannot.
            if bool((crowd != 0).any()):
                raise ValueError('MERT cannot realign mismatched nonzero iscrowd flags.')
            original['iscrowd'] = crowd.new_zeros(count)
        ids = target.get('origin_gt_id', torch.arange(count, device=boxes.device))
        ids = ids.to(device=boxes.device, dtype=torch.int64)
        if ids.shape != (count,) or ids.unique().numel() != count:
            raise ValueError('origin_gt_id must be a unique [N] integer tensor per image.')
        corners = _xyxy(boxes)
        original_visible = _fully_visible(corners)
        if 'fully_visible' in target:
            original_visible = original_visible & target['fully_visible'].bool()
        original['origin_gt_id'] = ids.clone()
        original['fully_visible'] = original_visible

        offset = boxes.new_tensor((width, height, width, height))
        translation = shift.to(boxes).repeat(2) / offset
        translated = corners + translation
        visible = original_visible & _fully_visible(translated)
        clipped = translated.clamp(0, 1)
        keep = (clipped[:, 2:] > clipped[:, :2]).all(dim=-1)
        shifted_target = dict(original)
        shifted_target['boxes'] = _cxcywh(clipped)
        shifted_target['fully_visible'] = visible
        # Do not filter metadata just because its first dimension equals N:
        # image_id/size/orig_size must remain unchanged, especially for N=1/2.
        for key in instance_fields & shifted_target.keys():
            shifted_target[key] = shifted_target[key][keep]
        if 'area' in shifted_target:
            shifted_target['area'] = shifted_target['boxes'][:, 2:].prod(-1) * width * height
        if 'masks' in shifted_target:
            masks = shifted_target['masks']
            if len(masks):
                masks = F.interpolate(masks[:, None].float(), size=(height, width),
                                      mode='nearest').to(masks.dtype)
                mask_shifts = shift[None].expand(len(masks), -1)
                shifted_target['masks'] = shift_images(masks, mask_shifts)[:, 0]
            else:
                shifted_target['masks'] = masks.new_empty((0, height, width))
        if 'keypoints' in shifted_target:
            # The original detection protocol is box-only. Do not silently
            # invent a normalized/absolute keypoint-coordinate convention.
            raise ValueError('MERT currently supports box/mask targets, not keypoints.')
        originals.append(original)
        shifted.append(shifted_target)
    return originals, shifted


def decoder_box_trajectory(outputs):
    """[L,B,Q,4] from decoder outputs only, never DN or encoder proposals.

    In the unmodified RT-DETR output protocol, the final aux_outputs entry is
    an encoder proposal. Decoder aux layers are therefore aux_outputs[:-1].
    Without aux outputs there is no observable transition, hence L=1.
    """
    decoder_aux = outputs.get('aux_outputs', [])[:-1]
    boxes = [layer['pred_boxes'] for layer in decoder_aux]
    boxes.append(outputs['pred_boxes'])
    return torch.stack(boxes, dim=0)


def split_concatenated_outputs(outputs, batch_size):
    """Split one DDP forward, preserving ordinary and DN criterion outputs."""
    def split(value, key=None):
        if key == 'dn_meta':
            first, second = dict(value), dict(value)
            positive = value['dn_positive_idx']
            if len(positive) != 2 * batch_size:
                raise ValueError('Concatenated DN metadata has an unexpected batch size.')
            first['dn_positive_idx'] = positive[:batch_size]
            second['dn_positive_idx'] = positive[batch_size:]
            return first, second
        if isinstance(value, torch.Tensor):
            if value.ndim and value.shape[0] == 2 * batch_size:
                return value[:batch_size], value[batch_size:]
            return value, value
        if isinstance(value, Mapping):
            first, second = {}, {}
            for name, item in value.items():
                first[name], second[name] = split(item, name)
            return first, second
        if isinstance(value, (list, tuple)):
            pairs = [split(item) for item in value]
            first = [pair[0] for pair in pairs]
            second = [pair[1] for pair in pairs]
            return (tuple(first), tuple(second)) if isinstance(value, tuple) else (first, second)
        return value, value

    if outputs['pred_boxes'].shape[0] != 2 * batch_size:
        raise ValueError('Expected original+shifted outputs with batch dimension 2B.')
    return split(outputs)


def average_loss_dicts(original, shifted):
    """Average views so shifted supervision does not double baseline weights.

    A missing DN branch is a differentiable zero contribution, not a change
    of the denominator. This also supports sequential batches with no shifted
    GTs, for which the original decoder does not generate a DN branch.
    """
    result = {}
    for key in original.keys() | shifted.keys():
        if key in original and key in shifted:
            result[key] = (original[key] + shifted[key]) * 0.5
        elif key in original:
            result[key] = original[key] * 0.5
        else:
            result[key] = shifted[key] * 0.5
    return result


class MERT:
    """Plain solver-side plugin; deliberately not an nn.Module."""

    def __init__(self, config=None, **overrides):
        options = dict(config or {})
        options.update(overrides)
        allowed = {'enabled', 'mode', 'shift_pixels', 'beta', 'xy_weight',
                   'wh_weight', 'trajectory_layers', 'loss_weight',
                   'shifted_detection_loss', 'forward_mode',
                   'small_object_weighting', 'schedule'}
        unknown = options.keys() - allowed
        if unknown:
            raise ValueError('Unknown MERT options: ' + ', '.join(sorted(unknown)))
        self.enabled = bool(options.get('enabled', False))
        self.mode = options.get('mode', 'late_xywh')
        self.shift_pixels = options.get('shift_pixels', 1)
        self.beta = float(options.get('beta', 0.01))
        self.xy_weight = float(options.get('xy_weight', 1.0))
        self.wh_weight = float(options.get('wh_weight', 0.25))
        self.trajectory_layers = options.get('trajectory_layers', 'last_2')
        self.loss_weight = float(options.get('loss_weight', 0.10))
        self.shifted_detection_loss = bool(options.get('shifted_detection_loss', True))
        self.forward_mode = options.get('forward_mode', 'concat')
        self.small_object_weighting = {'enabled': True, 'reference_area': 256.0,
                                      'gamma': 0.5, 'min_weight': 1.0,
                                      'max_weight': 4.0}
        small = dict(options.get('small_object_weighting') or {})
        if small.keys() - self.small_object_weighting.keys():
            raise ValueError('Unknown small_object_weighting options.')
        self.small_object_weighting.update(small)
        schedule = dict(options.get('schedule') or {})
        if schedule.keys() - {'enabled'} or schedule.get('enabled', False):
            raise ValueError('MERT uses the fixed verified loss weight; schedules are not supported.')
        if self.mode != 'late_xywh' or self.trajectory_layers != 'last_2':
            raise ValueError('Only the requested late_xywh / last_2 protocol is implemented.')
        if self.shift_pixels != 1:
            raise ValueError('The requested MERT protocol requires shift_pixels=1.')
        if self.beta <= 0 or min(self.xy_weight, self.wh_weight, self.loss_weight) < 0:
            raise ValueError('MERT beta must be positive and loss weights nonnegative.')
        if self.forward_mode not in {'concat', 'sequential'}:
            raise ValueError('forward_mode must be concat or sequential.')
        small = self.small_object_weighting
        if float(small['reference_area']) <= 0 or float(small['gamma']) < 0 or \
                not 0 < float(small['min_weight']) <= float(small['max_weight']):
            raise ValueError('Invalid MERT small-object weighting parameters.')

    def prepare(self, samples, targets, model, shifts=None) -> Optional[MERTBatch]:
        # This early return must precede RNG, resize, target copying, and shift.
        if not self.enabled or not model.training:
            return None
        if samples.ndim != 4 or len(targets) != samples.shape[0]:
            raise ValueError('MERT expects a dense image batch and one target dict per image.')
        detector = _unwrap_model(model)
        multi_scale = getattr(detector, 'multi_scale', None)
        if multi_scale:
            size = int(np.random.choice(multi_scale))
            samples = F.interpolate(samples, size=[size, size])
        batch_size = samples.shape[0]
        if shifts is None:
            choices = torch.tensor([(dx, dy) for dx in (-1, 0, 1)
                                    for dy in (-1, 0, 1) if dx or dy])
            shifts = choices[torch.randint(len(choices), (batch_size,))].to(samples.device)
        else:
            shifts = torch.as_tensor(shifts, device=samples.device)
            if shifts.shape != (batch_size, 2) or \
                    not bool(((shifts >= -1) & (shifts <= 1) &
                              (shifts == shifts.round())).all()) or \
                    not bool((shifts != 0).any(dim=-1).all()):
                raise ValueError('Each MERT shift must be a nonzero integer pair from {-1,0,1}^2.')
            shifts = shifts.long()
        spatial_size = tuple(samples.shape[-2:])
        originals, shifted = shift_targets(targets, shifts, spatial_size)
        return MERTBatch(samples, originals, shift_images(samples, shifts),
                         shifted, shifts, spatial_size)

    def _object_weights(self, boxes, spatial_size):
        options = self.small_object_weighting
        if not options['enabled']:
            return boxes.new_ones(len(boxes))
        height, width = spatial_size
        area = boxes[:, 2:].prod(-1) * width * height
        return (float(options['reference_area']) / (area + 1e-6)).pow(
            float(options['gamma'])).clamp(float(options['min_weight']),
                                          float(options['max_weight']))

    def calculate_loss(self, original_outputs, shifted_outputs, batch, matcher):
        if not self.enabled or batch is None:
            return {}
        original = decoder_box_trajectory(original_outputs).float()
        shifted = decoder_box_trajectory(shifted_outputs).float()
        if original.shape[:2] != shifted.shape[:2]:
            raise ValueError('MERT views must expose the same decoder layers and batch size.')
        zero = original.sum() * 0.0 + shifted.sum() * 0.0
        transitions = min(2, len(original) - 1)
        if transitions == 0:
            return {'loss_mert': zero}
        distributed = torch.distributed.is_available() and \
            torch.distributed.is_initialized()
        # In single-process mode, retain the cheap empty-target path. Under
        # DDP every rank must continue to the pair-count collective even when
        # its local batch has no valid pair, otherwise ranks can deadlock or
        # normalize their gradients with different denominators.
        if not distributed and (not any(len(t['boxes']) for t in batch.targets) or
                                not any(len(t['boxes']) for t in batch.shifted_targets)):
            return {'loss_mert': zero}
        # Independently use the original final-layer Hungarian matcher. Never
        # assume equal query indices, and never rematch intermediate layers.
        with torch.no_grad():
            original_matches = matcher({key: original_outputs[key].float()
                                        for key in ('pred_logits', 'pred_boxes')}, batch.targets)
            shifted_matches = matcher({key: shifted_outputs[key].float()
                                       for key in ('pred_logits', 'pred_boxes')}, batch.shifted_targets)
        coordinate_weights = original.new_tensor((self.xy_weight, self.xy_weight,
                                                  self.wh_weight, self.wh_weight))
        total, pair_count = zero, 0
        for i, ((oq, og), (sq, sg)) in enumerate(zip(original_matches, shifted_matches)):
            target, shifted_target = batch.targets[i], batch.shifted_targets[i]
            original_ids = target['origin_gt_id'].detach().cpu().tolist()
            shifted_ids = shifted_target['origin_gt_id'].detach().cpu().tolist()
            original_visible = target['fully_visible'].detach().cpu().tolist()
            shifted_visible = shifted_target['fully_visible'].detach().cpu().tolist()
            by_id = {original_ids[g]: (q, g) for q, g in zip(oq.tolist(), og.tolist())
                     if original_visible[g]}
            pairs = [(by_id[shifted_ids[g]][0], q, by_id[shifted_ids[g]][1])
                     for q, g in zip(sq.tolist(), sg.tolist())
                     if shifted_visible[g] and shifted_ids[g] in by_id]
            if not pairs:
                continue
            oi, si, gi = (torch.tensor(index, device=original.device, dtype=torch.long)
                          for index in zip(*pairs))
            original_boxes = original[-(transitions + 1):, i, oi]
            shifted_boxes = inverse_box_shift(shifted[-(transitions + 1):, i, si],
                                              batch.shifts[i], batch.spatial_size)
            delta_original = original_boxes[1:] - original_boxes[:-1]
            delta_shifted = shifted_boxes[1:] - shifted_boxes[:-1]
            coordinate_loss = F.smooth_l1_loss(delta_original, delta_shifted,
                                               beta=self.beta, reduction='none')
            weights = self._object_weights(target['boxes'][gi].float(), batch.spatial_size)
            total = total + ((coordinate_loss * coordinate_weights).sum(-1) * weights).sum()
            pair_count += len(pairs)
        normalizer = original.new_tensor(float(pair_count))
        if distributed:
            torch.distributed.all_reduce(normalizer)
            normalizer = normalizer / torch.distributed.get_world_size()
        if normalizer.item() == 0:
            return {'loss_mert': zero}
        return {'loss_mert': self.loss_weight * total / (normalizer * transitions)}


MERTTrainingPlugin = MERT
