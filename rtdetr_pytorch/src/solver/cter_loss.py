"""Cross-Stage Target Evidence Relay: a parameter-free, training-only loss.

Inputs are existing backbone features, original normalized cxcywh GT boxes,
and the actual post-multiscale image sizes. No decoder or shifted-view input
is accepted. Strides come from the backbone, not from a fixed 8/16/32 table.
"""

import torch
import torch.distributed as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.misc.amp import autocast_context


def _mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError('{} must be a mapping'.format(name))
    return value


def _cell_bounds(image_size, feature_shape, stride, reference):
    height, width = image_size
    feature_h, feature_w = feature_shape
    x1 = torch.arange(feature_w, device=reference.device, dtype=torch.float32) * stride
    y1 = torch.arange(feature_h, device=reference.device, dtype=torch.float32) * stride
    x2 = (x1 + stride).clamp(max=float(width))
    y2 = (y1 + stride).clamp(max=float(height))
    area = (y2 - y1).clamp_min(0)[:, None] * (x2 - x1).clamp_min(0)[None, :]
    return x1, y1, x2, y2, area


def soft_cell_occupancy(boxes_xyxy, image_size, feature_shape, stride):
    """Exact rectangle/cell intersection area divided by the cell area.

    All boxes and all spatial cells are processed with tensor broadcasting.
    Cells are clipped at image borders; no integer floor/ceil GT crop is used.
    """
    boxes = boxes_xyxy.float()
    x1, y1, x2, y2, area = _cell_bounds(
        image_size, feature_shape, float(stride), boxes
    )
    overlap_x = (
        torch.minimum(boxes[:, 2, None], x2[None])
        - torch.maximum(boxes[:, 0, None], x1[None])
    ).clamp_min(0)
    overlap_y = (
        torch.minimum(boxes[:, 3, None], y2[None])
        - torch.maximum(boxes[:, 1, None], y1[None])
    ).clamp_min(0)
    return (
        overlap_y[:, :, None] * overlap_x[:, None, :]
        / area.clamp_min(1.0e-12)[None]
    ).clamp(0, 1)


def context_ring_occupancy(
    target_box, all_boxes, image_size, feature_shape, stride, context_scale=1.5
):
    """Area occupancy of expanded context minus the UNION of every GT box.

    A vectorized rectangle partition gives exact exclusion, including
    overlapping GT boxes. There is no Python loop over feature-grid cells and
    no hard rejection of a complete cell just because a small GT touches it.
    """
    height, width = image_size
    center = (target_box[:2] + target_box[2:]) * 0.5
    half_size = (target_box[2:] - target_box[:2]) * (float(context_scale) * 0.5)
    limits = target_box.new_tensor([float(width), float(height)])
    lower = (center - half_size).clamp_min(0)
    upper = torch.minimum(center + half_size, limits)

    # All clipped GT boundaries partition context into rectangles. Membership
    # is constant inside each rectangle, so union exclusion is exact.
    clipped_lower = torch.minimum(torch.maximum(all_boxes[:, :2], lower), upper)
    clipped_upper = torch.minimum(torch.maximum(all_boxes[:, 2:], lower), upper)
    x_edges = torch.cat([
        lower[0:1], upper[0:1], clipped_lower[:, 0], clipped_upper[:, 0]
    ]).sort().values
    y_edges = torch.cat([
        lower[1:2], upper[1:2], clipped_lower[:, 1], clipped_upper[:, 1]
    ]).sort().values
    part_x1, part_x2 = x_edges[:-1], x_edges[1:]
    part_y1, part_y2 = y_edges[:-1], y_edges[1:]
    mid_x = (part_x1 + part_x2) * 0.5
    mid_y = (part_y1 + part_y2) * 0.5
    inside_x = (mid_x[:, None] >= all_boxes[None, :, 0]) \
        & (mid_x[:, None] < all_boxes[None, :, 2])
    inside_y = (mid_y[:, None] >= all_boxes[None, :, 1]) \
        & (mid_y[:, None] < all_boxes[None, :, 3])
    covered = (inside_y[:, None, :] & inside_x[None, :, :]).any(dim=-1)
    free_partition = (~covered).to(dtype=torch.float32)

    cell_x1, cell_y1, cell_x2, cell_y2, cell_area = _cell_bounds(
        image_size, feature_shape, float(stride), target_box
    )
    overlap_x = (
        torch.minimum(part_x2[:, None], cell_x2[None])
        - torch.maximum(part_x1[:, None], cell_x1[None])
    ).clamp_min(0)
    overlap_y = (
        torch.minimum(part_y2[:, None], cell_y2[None])
        - torch.maximum(part_y1[:, None], cell_y1[None])
    ).clamp_min(0)
    background_area = overlap_y.transpose(0, 1) @ free_partition @ overlap_x
    return (background_area / cell_area.clamp_min(1.0e-12)).clamp(0, 1)


def target_evidence_margin(normalized_feature, positive, background, temperature=0.1, eps=1e-6):
    """Instantaneous instance prototype, positive similarity and hard margin.

    Background uses occupancy-weighted log-mean-exp. Binary occupancy exactly
    recovers tau*log(sum(exp(q/tau))/N). Empty supports remain finite and are
    marked invalid instead of producing infinities/NaNs in the loss graph.
    """
    feature = normalized_feature.flatten(1)
    positive_weights = positive.flatten()
    background_weights = background.flatten()
    positive_area = positive_weights.sum()
    background_area = background_weights.sum()
    prototype = (feature @ positive_weights) / (positive_area + eps)
    prototype = F.normalize(prototype, p=2, dim=0, eps=eps)
    similarity = prototype @ feature
    positive_similarity = (similarity * positive_weights).sum() / (positive_area + eps)

    # Do not inflate fractional background weights smaller than eps.
    tiny = torch.finfo(background_weights.dtype).tiny
    terms = similarity / temperature + background_weights.clamp_min(tiny).log()
    terms = torch.where(
        background_weights > 0, terms, torch.full_like(terms, float('-inf'))
    )
    has_background = background_area > eps
    # Inactive all-empty supports must not enter logsumexp as all -inf.
    terms = torch.where(has_background, terms, torch.zeros_like(terms))
    background_similarity = temperature * (
        torch.logsumexp(terms, dim=0) - background_area.clamp_min(eps).log()
    )
    background_similarity = torch.where(
        has_background, background_similarity, torch.zeros_like(background_similarity)
    )
    valid = (positive_area > eps) & has_background
    return positive_similarity - background_similarity, positive_area, valid


def evidence_relay_loss(source_margin, destination_margin, relay_ratio=0.9):
    """Only destination margin receives a gradient from this direct relay."""
    reference = source_margin.relu().detach() * float(relay_ratio)
    return (reference - destination_margin).relu().square()


class CTERLoss(nn.Module):
    def __init__(self, config, feature_strides, stage_names=None):
        super().__init__()
        config = _mapping(config, 'CTER')
        self.feature_strides = tuple(float(stride) for stride in feature_strides)
        if any(stride <= 0 for stride in self.feature_strides):
            raise ValueError('CTER feature strides must be positive')
        if stage_names is None:
            stage_names = tuple('s{}'.format(index + 3) for index in range(len(feature_strides)))
        self.stage_names = tuple(stage_names)
        if len(self.stage_names) != len(self.feature_strides):
            raise ValueError('CTER feature strides and stage names differ in length')
        self.stage_indices = {name: index for index, name in enumerate(self.stage_names)}
        self.transitions = tuple(config.get('transitions', ['3to4', '4to5']))
        supported = {'3to4': ('s3', 's4'), '4to5': ('s4', 's5')}
        if not self.transitions or len(set(self.transitions)) != len(self.transitions):
            raise ValueError('CTER.transitions must be non-empty and unique')
        self.transition_stages = {}
        for transition in self.transitions:
            if transition not in supported:
                raise ValueError('Unsupported CTER transition: {}'.format(transition))
            stages = supported[transition]
            if any(stage not in self.stage_indices for stage in stages):
                raise ValueError('Backbone does not expose stages for {}'.format(transition))
            self.transition_stages[transition] = stages
        self.debug = bool(config.get('debug', False))
        self.required_stages = tuple(
            name for name in self.stage_names
            if self.debug or any(name in pair for pair in self.transition_stages.values())
        )
        self.context_scale = float(config.get('context_scale', 1.5))
        self.eps = float(config.get('eps', 1.0e-6))
        hard_negative = _mapping(config.get('hard_negative'), 'CTER.hard_negative')
        self.temperature = float(hard_negative.get('temperature', 0.10))
        self.relay_ratio = float(config.get('relay_ratio', 0.90))
        occupancy = _mapping(config.get('occupancy'), 'CTER.occupancy')
        if occupancy.get('mode', 'soft_overlap') != 'soft_overlap':
            raise ValueError('CTER supports only occupancy.mode=soft_overlap')
        validity = _mapping(config.get('stage_validity'), 'CTER.stage_validity')
        self.stage_validity_enabled = bool(validity.get('enabled', True))
        if self.context_scale <= 1 or self.eps <= 0 or self.temperature <= 0:
            raise ValueError('Invalid CTER context_scale/eps/temperature')
        if not 0 <= self.relay_ratio <= 1:
            raise ValueError('CTER.relay_ratio must be in [0, 1]')

    def forward(self, features, targets, image_sizes):
        if len(features) != len(self.feature_strides):
            raise ValueError('CTER backbone feature count differs from stride metadata')
        if features[0].shape[0] != len(targets):
            raise ValueError('CTER original-view batch and GT count differ')
        if len(image_sizes) == 2 and isinstance(image_sizes[0], int):
            image_sizes = [tuple(image_sizes)] * len(targets)
        if len(image_sizes) != len(targets):
            raise ValueError('CTER needs one actual image size per image')

        with autocast_context(features[0].device, enabled=False):
            numerator = features[0].float().sum() * 0.0
            total_targets = sum(len(target['boxes']) for target in targets)
            debug_sums = {}

            def record(key, value):
                if self.debug:
                    value = value.detach()
                    debug_sums[key] = debug_sums.get(key, torch.zeros_like(value)) + value

            for image_index, (target, image_size) in enumerate(zip(targets, image_sizes)):
                num_targets = len(target['boxes'])
                if num_targets == 0:
                    continue
                height, width = image_size
                boxes = target['boxes'].detach().to(
                    device=features[0].device, dtype=torch.float32
                )
                centers, sizes = boxes[:, :2], boxes[:, 2:]
                boxes_xyxy = torch.cat([centers - sizes * 0.5, centers + sizes * 0.5], dim=-1)
                boxes_xyxy = boxes_xyxy * boxes.new_tensor([width, height, width, height])
                boxes_xyxy[:, 0::2].clamp_(0, width)
                boxes_xyxy[:, 1::2].clamp_(0, height)
                margins, occupancies, validities = {}, {}, {}

                for stage in self.required_stages:
                    index = self.stage_indices[stage]
                    feature = features[index][image_index].float()
                    stride = self.feature_strides[index]
                    normalized = F.normalize(feature, p=2, dim=0, eps=self.eps)
                    positives = soft_cell_occupancy(
                        boxes_xyxy, image_size, feature.shape[-2:], stride
                    )
                    stage_margins, stage_occupancies, stage_valid = [], [], []
                    for target_index in range(num_targets):
                        background = context_ring_occupancy(
                            boxes_xyxy[target_index], boxes_xyxy, image_size,
                            feature.shape[-2:], stride, self.context_scale,
                        )
                        margin, occupancy, valid = target_evidence_margin(
                            normalized, positives[target_index], background,
                            self.temperature, self.eps,
                        )
                        stage_margins.append(margin)
                        stage_occupancies.append(occupancy)
                        stage_valid.append(valid)
                    margins[stage] = torch.stack(stage_margins)
                    occupancies[stage] = torch.stack(stage_occupancies)
                    validities[stage] = torch.stack(stage_valid)
                    if self.debug:
                        valid_float = validities[stage].float()
                        record('margin_{}_sum'.format(stage), (margins[stage] * valid_float).sum())
                        record('margin_{}_count'.format(stage), valid_float.sum())
                        record('occupancy_{}_sum'.format(stage), occupancies[stage].sum())
                        record('occupancy_{}_count'.format(stage), feature.new_tensor(num_targets))

                for transition, (source, destination) in self.transition_stages.items():
                    valid = validities[source] & validities[destination]
                    validity_weight = occupancies[destination].clamp(0, 1) \
                        if self.stage_validity_enabled else torch.ones_like(occupancies[destination])
                    weight = valid.float() * validity_weight
                    relay = evidence_relay_loss(margins[source], margins[destination], self.relay_ratio)
                    weighted_sum = (weight * relay).sum()
                    numerator = numerator + weighted_sum
                    if self.debug:
                        suffix = transition.replace('to', '')
                        record('loss_{}_sum'.format(suffix), weighted_sum)
                        record('loss_{}_count'.format(suffix), weighted_sum.new_tensor(num_targets))
                        record('valid_targets_{}'.format(suffix), valid.float().sum())
                        reliable_source = valid & (margins[source] > self.eps)
                        observed = margins[destination] / margins[source].relu().clamp_min(self.eps)
                        record('relay_ratio_{}_sum'.format(suffix), (observed * reliable_source.float()).sum())
                        record('relay_ratio_{}_count'.format(suffix), reliable_source.float().sum())

            denominator = numerator.new_tensor(float(total_targets))
            world_size = 1
            if tdist.is_available() and tdist.is_initialized():
                # Every rank participates, including ranks with empty GT batches.
                tdist.all_reduce(denominator)
                world_size = tdist.get_world_size()
            return numerator * world_size / denominator.clamp_min(1.0), debug_sums


class CTERTrainingPlugin(object):
    """Loss and deferred debug aggregation; never attached to the detector."""
    def __init__(self, config, feature_strides, stage_names):
        self.loss_weight = float(config.get('loss_weight', 0.05))
        if self.loss_weight < 0:
            raise ValueError('CTER.loss_weight must be non-negative')
        self.debug = bool(config.get('debug', False))
        self.loss_module = CTERLoss(config, feature_strides, stage_names)
        self.debug_sums = {}

    def calculate_loss(self, features, targets, image_sizes):
        loss, stats = self.loss_module(features, targets, image_sizes)
        if self.debug:
            # Pre-create keys on every rank so empty-GT ranks participate in
            # the same epoch-end collective with the same packed shape.
            if not self.debug_sums:
                zero = loss.detach().new_zeros(())
                for stage in self.loss_module.required_stages:
                    for metric in ('margin', 'occupancy'):
                        for suffix in ('sum', 'count'):
                            self.debug_sums['{}_{}_{}'.format(metric, stage, suffix)] = zero.clone()
                for transition in self.loss_module.transitions:
                    name = transition.replace('to', '')
                    for metric in ('loss', 'relay_ratio'):
                        for suffix in ('sum', 'count'):
                            self.debug_sums['{}_{}_{}'.format(metric, name, suffix)] = zero.clone()
                    self.debug_sums['valid_targets_{}'.format(name)] = zero.clone()
            for key, value in stats.items():
                self.debug_sums[key].add_(value.detach())
        return self.loss_weight * loss

    def debug_summary(self):
        if not self.debug or not self.debug_sums:
            return {}
        keys = sorted(self.debug_sums)
        packed = torch.stack([self.debug_sums[key] for key in keys])
        if tdist.is_available() and tdist.is_initialized():
            tdist.all_reduce(packed)
        values = dict(zip(keys, packed.cpu().tolist()))
        summary = {}
        for stage in self.loss_module.required_stages:
            for metric in ('margin', 'occupancy'):
                prefix = '{}_{}'.format(metric, stage)
                summary[prefix + '_mean'] = values[prefix + '_sum'] / max(values[prefix + '_count'], 1.0)
        for transition in self.loss_module.transitions:
            name = transition.replace('to', '')
            for metric in ('loss', 'relay_ratio'):
                prefix = '{}_{}'.format(metric, name)
                output = prefix if metric == 'loss' else prefix + '_observed'
                summary[output] = values[prefix + '_sum'] / max(values[prefix + '_count'], 1.0)
            summary['valid_targets_' + name] = values['valid_targets_' + name]
        return summary
