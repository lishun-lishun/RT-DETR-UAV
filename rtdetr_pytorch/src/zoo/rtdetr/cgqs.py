"""Consensus-guided global query selection for RT-DETR."""

from __future__ import annotations

import math
import warnings

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class ConsensusGuidedQuerySelector(nn.Module):
    """Re-score encoder proposals while retaining one global Top-K."""

    def __init__(self, num_queries=300, use_consensus=True,
                 consensus_weight=0.5, use_local_peak=True, peak_weight=0.25,
                 peak_kernel=3, score_mode='additive', eps=1e-6,
                 align_mode='bilinear', align_corners=False, debug=False):
        super().__init__()
        if not isinstance(num_queries, int) or num_queries < 1:
            raise ValueError('CGQS.num_queries must be a positive integer')
        if not isinstance(use_consensus, bool) or not isinstance(use_local_peak, bool):
            raise ValueError('CGQS switches must be YAML booleans')
        if not isinstance(peak_kernel, int) or peak_kernel < 1 or peak_kernel % 2 == 0:
            raise ValueError('CGQS.peak_kernel must be a positive odd integer')
        if score_mode != 'additive':
            raise ValueError("CGQS.score_mode currently supports only 'additive'")
        for name, value in (('consensus_weight', consensus_weight),
                            ('peak_weight', peak_weight), ('eps', eps)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f'CGQS.{name} must be finite')
        if eps <= 0:
            raise ValueError('CGQS.eps must be positive')
        if align_mode not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('CGQS.align_mode must be nearest, bilinear or bicubic')
        if not isinstance(align_corners, bool) or not isinstance(debug, bool):
            raise ValueError('CGQS align_corners/debug must be YAML booleans')

        self.num_queries = num_queries
        self.use_consensus = use_consensus
        self.consensus_weight = float(consensus_weight)
        self.use_local_peak = use_local_peak
        self.peak_weight = float(peak_weight)
        self.peak_kernel = peak_kernel
        self.eps = float(eps)
        self.align_mode = align_mode
        self.align_corners = align_corners
        self.debug = debug
        self._warned_missing_consensus = False
        self._debug_printed = False
        self.last_selected_shape = None
        self.last_score_stats = None

    @staticmethod
    def _validate_shapes(class_logits, spatial_shapes, valid_mask):
        if class_logits.ndim != 3 or class_logits.shape[-1] < 1:
            raise RuntimeError('CGQS class logits must have shape [B,L,C]')
        if not isinstance(spatial_shapes, (list, tuple)):
            raise RuntimeError('CGQS spatial_shapes must be a list or tuple')
        lengths = []
        for level, shape in enumerate(spatial_shapes):
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or any(not isinstance(v, int) or v < 1 for v in shape)):
                raise RuntimeError(
                    f'CGQS spatial_shapes[{level}] must be positive [H,W]')
            lengths.append(shape[0] * shape[1])
        if sum(lengths) != class_logits.shape[1]:
            raise RuntimeError('CGQS spatial shapes do not match flattened logits')
        if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)
        if valid_mask.ndim != 2 or valid_mask.shape[1] != class_logits.shape[1]:
            raise RuntimeError('CGQS valid_mask must have shape [1|B,L,1] or [1|B,L]')
        return lengths, valid_mask.bool().expand(class_logits.shape[0], -1)

    def _resize_consensus(self, consensus, size):
        kwargs = {'size': size, 'mode': self.align_mode}
        if self.align_mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = self.align_corners
        return F.interpolate(consensus.float(), **kwargs)

    def forward(self, class_logits, spatial_shapes, valid_mask, consensus=None):
        lengths, valid = self._validate_shapes(
            class_logits, spatial_shapes, valid_mask)
        if consensus is not None and (consensus.ndim != 4
                                      or consensus.shape[1] != 1
                                      or consensus.shape[0] != class_logits.shape[0]):
            raise RuntimeError('CGQS consensus must have shape [B,1,H,W]')
        raw = class_logits.max(dim=-1).values.float()
        ranking = torch.log(raw.sigmoid() + self.eps)

        level_scores = torch.split(raw.sigmoid(), lengths, dim=1)
        peak_parts = []
        consensus_parts = []
        for probability, (height, width) in zip(level_scores, spatial_shapes):
            score_map = probability.reshape(-1, 1, height, width)
            if self.use_local_peak:
                local_mean = F.avg_pool2d(
                    score_map, self.peak_kernel, stride=1,
                    padding=self.peak_kernel // 2, count_include_pad=False)
                peak = (score_map - local_mean).relu()
                peak = peak / peak.flatten(1).amax(dim=1, keepdim=True).clamp_min(
                    self.eps).reshape(-1, 1, 1, 1)
                peak_parts.append(peak.flatten(1))
            if self.use_consensus and consensus is not None:
                resized = self._resize_consensus(consensus, (height, width))
                consensus_parts.append(resized.flatten(1))
        if self.use_local_peak:
            ranking = ranking + self.peak_weight * torch.cat(peak_parts, dim=1)
        consensus_active = self.use_consensus and consensus is not None
        if consensus_active:
            ranking = ranking + self.consensus_weight * torch.cat(
                consensus_parts, dim=1)
        elif self.use_consensus and not self._warned_missing_consensus:
            if _rank_zero():
                warnings.warn(
                    'CGQS consensus disabled because CRCG is off.',
                    RuntimeWarning, stacklevel=2)
            self._warned_missing_consensus = True

        ranking = ranking.masked_fill(~valid, float('-inf'))
        if class_logits.shape[1] < self.num_queries:
            raise RuntimeError('CGQS has fewer proposals than num_queries')
        indices = torch.topk(ranking, self.num_queries, dim=1).indices
        self.last_selected_shape = tuple(indices.shape)
        if self.debug and not self._debug_printed and _rank_zero():
            finite = ranking[valid]
            self.last_score_stats = (
                float(finite.detach().min()), float(finite.detach().max()),
                float(finite.detach().mean()))
            print('CGQS enabled: '
                  f'global_topk={self.num_queries}, selected={tuple(indices.shape)}, '
                  f'consensus={consensus_active}, local_peak={self.use_local_peak}, '
                  f'score[min,max,mean]={self.last_score_stats}')
            self._debug_printed = True
        return indices


CGQS = ConsensusGuidedQuerySelector
