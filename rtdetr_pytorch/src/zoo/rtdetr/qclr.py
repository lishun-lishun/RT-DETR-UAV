"""Query-conditioned local correlation refinement for small RT-DETR boxes."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class QueryConditionedLocalRefinement(nn.Module):
    """Refine only box centers using vectorized local feature correlation."""

    def __init__(self, query_dim=256, high_res_channels=18,
                 small_area_threshold=0.01, window_size=5, corr_dim=64,
                 temperature=0.1, learnable_beta=True, init_beta=0.0,
                 center_only=True, max_refine_queries=100,
                 feature_source='hrnet_highest', align_corners=False,
                 debug=False):
        super().__init__()
        if feature_source != 'hrnet_highest':
            raise ValueError(
                "QCLR.feature_source must be 'hrnet_highest' in this HRNet implementation")
        for name, value in (('query_dim', query_dim),
                            ('high_res_channels', high_res_channels),
                            ('window_size', window_size), ('corr_dim', corr_dim),
                            ('max_refine_queries', max_refine_queries)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'QCLR.{name} must be a positive integer')
        if window_size % 2 == 0:
            raise ValueError('QCLR.window_size must be odd')
        for name, value in (('small_area_threshold', small_area_threshold),
                            ('temperature', temperature), ('init_beta', init_beta)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f'QCLR.{name} must be finite')
        if small_area_threshold <= 0 or temperature <= 0:
            raise ValueError('QCLR area threshold and temperature must be positive')
        if not center_only:
            raise ValueError('QCLR first version supports center_only: true')
        if not isinstance(learnable_beta, bool) or not isinstance(align_corners, bool) \
                or not isinstance(debug, bool):
            raise ValueError('QCLR boolean options must be YAML booleans')

        self.query_dim = query_dim
        self.high_res_channels = high_res_channels
        self.small_area_threshold = float(small_area_threshold)
        self.window_size = window_size
        self.corr_dim = corr_dim
        self.temperature = float(temperature)
        self.max_refine_queries = max_refine_queries
        self.align_corners = align_corners
        self.debug = debug
        self._debug_printed = False

        self.query_projection = nn.Linear(query_dim, corr_dim, bias=False)
        self.feature_projection = nn.Linear(
            high_res_channels, corr_dim, bias=False)
        beta = torch.tensor(float(init_beta))
        if learnable_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer('beta', beta)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.feature_projection.weight)

        radius = window_size // 2
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer(
            'pixel_offsets',
            torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1),
            persistent=False)
        self.last_patch_shape = None
        self.last_selected_shape = None
        self.last_refined_count = None

    def forward(self, queries, boxes, class_logits, high_res_feature):
        if queries.ndim != 3 or queries.shape[-1] != self.query_dim:
            raise RuntimeError(
                f'QCLR queries must have shape [B,Q,{self.query_dim}]')
        if boxes.shape != queries.shape[:2] + (4,):
            raise RuntimeError('QCLR boxes must have shape [B,Q,4]')
        if class_logits.ndim != 3 or class_logits.shape[:2] != queries.shape[:2]:
            raise RuntimeError('QCLR class logits must align with [B,Q]')
        if (high_res_feature.ndim != 4
                or high_res_feature.shape[0] != queries.shape[0]
                or high_res_feature.shape[1] != self.high_res_channels):
            raise RuntimeError(
                'QCLR high-resolution feature has incompatible B/C dimensions')
        batch, query_count = boxes.shape[:2]
        selected_count = min(self.max_refine_queries, query_count)
        area = boxes[..., 2] * boxes[..., 3]
        small = area < self.small_area_threshold
        confidence = class_logits.float().sigmoid().amax(dim=-1)
        ranked = confidence.masked_fill(~small, float('-inf'))
        indices = ranked.topk(selected_count, dim=1).indices
        selected_valid = small.gather(1, indices)

        gather4 = indices.unsqueeze(-1).expand(-1, -1, 4)
        selected_boxes = boxes.gather(1, gather4)
        gatherq = indices.unsqueeze(-1).expand(-1, -1, queries.shape[-1])
        selected_queries = queries.gather(1, gatherq)

        height, width = high_res_feature.shape[-2:]
        offsets = self.pixel_offsets.to(device=boxes.device, dtype=boxes.dtype)
        scale = boxes.new_tensor([width, height])
        sample_xy = selected_boxes[..., None, :2] + offsets[None, None] / scale
        grid = sample_xy.mul(2.0).sub(1.0)
        patches = F.grid_sample(
            high_res_feature, grid, mode='bilinear', padding_mode='border',
            align_corners=self.align_corners)
        # grid_sample result: [B,C,N,K] -> [B,N,K,C].
        patches = patches.permute(0, 2, 3, 1).contiguous()
        query_embedding = F.normalize(
            self.query_projection(selected_queries).float(), dim=-1)
        patch_embedding = F.normalize(
            self.feature_projection(patches).float(), dim=-1)
        correlation = (patch_embedding * query_embedding.unsqueeze(2)).sum(-1)
        weights = (correlation / self.temperature).softmax(dim=-1)
        expected_pixels = torch.sum(
            weights.unsqueeze(-1) * offsets.float()[None, None], dim=2)
        normalized_delta = expected_pixels / expected_pixels.new_tensor(
            [width, height])
        normalized_delta = normalized_delta.to(boxes.dtype)
        normalized_delta = normalized_delta * selected_valid.unsqueeze(-1)

        deltas = boxes.new_zeros(batch, query_count, 2)
        deltas.scatter_add_(
            1, indices.unsqueeze(-1).expand(-1, -1, 2), normalized_delta)
        centers = (boxes[..., :2] + self.beta.to(boxes.dtype) * deltas).clamp(0, 1)
        refined = torch.cat((centers, boxes[..., 2:]), dim=-1)
        self.last_patch_shape = tuple(patches.shape)
        self.last_selected_shape = tuple(indices.shape)
        self.last_refined_count = selected_valid.detach().sum()
        if self.debug and not self._debug_printed and _rank_zero():
            if not torch.isfinite(refined).all():
                raise FloatingPointError('QCLR output contains NaN or Inf')
            print('QCLR enabled: '
                  f'feature={tuple(high_res_feature.shape)}, '
                  f'selected={tuple(indices.shape)}, '
                  f'valid={int(self.last_refined_count)}, '
                  f'patches={self.last_patch_shape}, beta={float(self.beta.detach())}')
            self._debug_printed = True
        return refined


QCLR = QueryConditionedLocalRefinement
