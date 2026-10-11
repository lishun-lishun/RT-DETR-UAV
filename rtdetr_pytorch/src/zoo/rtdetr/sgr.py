"""Small-object Geometry-aware Regression helpers for RT-DETR.

SGR is a training-only criterion helper.  It does not change Hungarian
assignment, detector forward outputs, inference, or checkpoint structure.
The helper supplies target-area weights and a normalized centre-error term;
the owning criterion keeps the original loss reduction and DDP normalizer.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn


__all__ = ['SmallObjectGeometryRegression']


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be boolean')
    return value


class SmallObjectGeometryRegression(nn.Module):
    """Compute SGR geometry terms without introducing trainable parameters.

    ``area_ref_mode`` is resolved once per criterion forward and then reused
    by the main and auxiliary losses.  Batch statistics are synchronized over
    all distributed ranks so every rank optimizes with the same reference.
    """

    def __init__(self, use_scale_weight=True, gamma=0.25,
                 weight_max=2.0, area_ref_mode='fixed', area_ref=0.01,
                 eps=1e-6, use_center_loss=True, center_loss_coef=0.25,
                 min_box_size=0.01, center_loss_max=10.0,
                 apply_to_aux=True, apply_to_dn=False):
        super().__init__()
        self.use_scale_weight = _boolean(
            'SGR.use_scale_weight', use_scale_weight)
        self.use_center_loss = _boolean(
            'SGR.use_center_loss', use_center_loss)
        self.apply_to_aux = _boolean('SGR.apply_to_aux', apply_to_aux)
        self.apply_to_dn = _boolean('SGR.apply_to_dn', apply_to_dn)

        self.gamma = _finite_float('SGR.gamma', gamma)
        self.weight_max = _finite_float('SGR.weight_max', weight_max)
        self.area_ref = _finite_float('SGR.area_ref', area_ref)
        self.eps = _finite_float('SGR.eps', eps)
        self.center_loss_coef = _finite_float(
            'SGR.center_loss_coef', center_loss_coef)
        self.min_box_size = _finite_float(
            'SGR.min_box_size', min_box_size)
        self.center_loss_max = _finite_float(
            'SGR.center_loss_max', center_loss_max)

        if self.gamma < 0.0:
            raise ValueError('SGR.gamma must be non-negative')
        if self.weight_max < 1.0:
            raise ValueError('SGR.weight_max must be at least 1')
        if self.area_ref <= 0.0:
            raise ValueError('SGR.area_ref must be positive')
        if self.eps <= 0.0:
            raise ValueError('SGR.eps must be positive')
        if self.center_loss_coef < 0.0:
            raise ValueError('SGR.center_loss_coef must be non-negative')
        if self.min_box_size <= 0.0:
            raise ValueError('SGR.min_box_size must be positive')
        if self.center_loss_max <= 0.0:
            raise ValueError('SGR.center_loss_max must be positive')

        if not isinstance(area_ref_mode, str):
            raise ValueError('SGR.area_ref_mode must be a string')
        self.area_ref_mode = area_ref_mode.lower()
        if self.area_ref_mode not in ('fixed', 'batch_mean', 'batch_median'):
            raise ValueError(
                'SGR.area_ref_mode must be fixed, batch_mean or '
                'batch_median')

    def should_apply(self, stage):
        """Return whether SGR applies to a criterion output stage."""
        if stage == 'main':
            return True
        if stage == 'aux':
            return self.apply_to_aux
        if stage == 'encoder_aux':
            # RT-DETR appends its encoder proposal loss as the final member of
            # aux_outputs; apply_to_aux intentionally means decoder aux only.
            return False
        if stage == 'dn':
            return self.apply_to_dn
        raise ValueError(f'Unknown SGR loss stage: {stage!r}')

    @staticmethod
    def _dist_enabled():
        return dist.is_available() and dist.is_initialized()

    @torch.no_grad()
    def _target_areas(self, targets, device):
        boxes = [target['boxes'].detach().to(
            device=device, dtype=torch.float32) for target in targets]
        if not boxes:
            return torch.empty(0, device=device, dtype=torch.float32)
        boxes = torch.cat(boxes, dim=0)
        if boxes.numel() == 0:
            return boxes.new_empty((0,))
        return boxes[:, 2].clamp_min(0.0) * boxes[:, 3].clamp_min(0.0)

    @torch.no_grad()
    def resolve_area_ref(self, targets, device):
        """Resolve one scalar area reference, globally consistent under DDP."""
        fixed = torch.tensor(
            self.area_ref, dtype=torch.float32, device=device)
        if self.area_ref_mode == 'fixed':
            return fixed

        areas = self._target_areas(targets, device)
        if self.area_ref_mode == 'batch_mean':
            stats = torch.stack((
                areas.sum(), areas.new_tensor(float(areas.numel()))))
            if self._dist_enabled():
                dist.all_reduce(stats)
            if stats[1].item() == 0:
                return fixed
            return stats[0] / stats[1]

        # Variable-length, device-local all-gather avoids object collectives
        # and works with both NCCL/CUDA and Gloo/CPU process groups.
        local_count = torch.tensor(
            [areas.numel()], dtype=torch.int64, device=device)
        if not self._dist_enabled():
            return areas.median() if areas.numel() else fixed

        world_size = dist.get_world_size()
        gathered_counts = [torch.zeros_like(local_count)
                           for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(value.item()) for value in gathered_counts]
        maximum = max(counts)
        if maximum == 0:
            return fixed

        padded = areas.new_zeros(maximum)
        if areas.numel():
            padded[:areas.numel()] = areas
        gathered_areas = [torch.zeros_like(padded)
                          for _ in range(world_size)]
        dist.all_gather(gathered_areas, padded)
        all_areas = torch.cat([
            value[:count] for value, count in zip(gathered_areas, counts)
            if count > 0
        ])
        return all_areas.median()

    def scale_weights(self, target_boxes, resolved_area_ref):
        """Return one clipped target-area weight for every matched box."""
        target = target_boxes.detach().float()
        areas = (target[:, 2].clamp_min(0.0)
                 * target[:, 3].clamp_min(0.0))
        reference = torch.as_tensor(
            resolved_area_ref, device=areas.device, dtype=areas.dtype)
        weights = (reference / (areas + self.eps)).pow(self.gamma)
        return weights.clamp(min=1.0, max=self.weight_max)

    def center_error(self, source_boxes, target_boxes):
        """Return clipped normalized centre error for each matched pair."""
        source = source_boxes.float()
        target = target_boxes.detach().float()
        denominator_w = target[:, 2].clamp_min(self.min_box_size)
        denominator_h = target[:, 3].clamp_min(self.min_box_size)
        center = (
            (source[:, 0] - target[:, 0]).abs() / denominator_w
            + (source[:, 1] - target[:, 1]).abs() / denominator_h)
        return center.clamp(max=self.center_loss_max)
