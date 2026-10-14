"""Cross-resolution consensus gating for HRNet high-resolution features."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class CrossResolutionConsensusGating(nn.Module):
    """Gate HRNet branch 0 with evidence agreed on by branches 1..3.

    The residual scale is initialized to zero, so enabling CRCG starts from
    the exact HRNet feature path while keeping the new path trainable.
    """

    def __init__(self, branch_channels, embed_dim=64, gate_type='sigmoid',
                 temperature=2.0, gate_bias=0.0, consensus_reduce='mean',
                 mode='gate_self', align_mode='bilinear',
                 align_corners=False, init_alpha=0.0, debug=False):
        super().__init__()
        if len(branch_channels) < 2:
            raise ValueError('CRCG requires at least two resolution branches')
        if not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError('CRCG.embed_dim must be a positive integer')
        for name, value in (('temperature', temperature),
                            ('gate_bias', gate_bias),
                            ('init_alpha', init_alpha)):
            if (not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise ValueError(f'CRCG.{name} must be finite')
        if temperature <= 0:
            raise ValueError('CRCG.temperature must be positive')
        if gate_type not in ('sigmoid', 'linear'):
            raise ValueError("CRCG.gate_type must be 'sigmoid' or 'linear'")
        if consensus_reduce not in ('mean', 'learnable'):
            raise ValueError("CRCG.consensus_reduce must be 'mean' or 'learnable'")
        if mode not in ('gate_self', 'gate_cross'):
            raise ValueError("CRCG.mode must be 'gate_self' or 'gate_cross'")
        if align_mode not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('CRCG.align_mode must be nearest, bilinear or bicubic')
        if not isinstance(align_corners, bool) or not isinstance(debug, bool):
            raise ValueError('CRCG align_corners/debug must be YAML booleans')

        self.branch_channels = tuple(branch_channels)
        self.embed_dim = embed_dim
        self.gate_type = gate_type
        self.temperature = float(temperature)
        self.gate_bias = float(gate_bias)
        self.consensus_reduce = consensus_reduce
        self.mode = mode
        self.align_mode = align_mode
        self.align_corners = align_corners
        self.debug = debug
        self._debug_printed = False

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embed_dim, 1, bias=False)
            for channels in self.branch_channels)
        self.detail = nn.Conv2d(
            self.branch_channels[0], self.branch_channels[0], 3, padding=1,
            groups=self.branch_channels[0], bias=False)
        if mode == 'gate_cross':
            self.cross_out = nn.Conv2d(
                embed_dim, self.branch_channels[0], 1, bias=False)
        if consensus_reduce == 'learnable':
            self.consensus_logits = nn.Parameter(
                torch.zeros(len(self.branch_channels) - 1))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')

        self.last_branch_shapes = None
        self.last_reliability_shape = None
        self.last_reliability_stats = None

    def _resize(self, tensor, size):
        kwargs = {'size': size, 'mode': self.align_mode}
        if self.align_mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = self.align_corners
        return F.interpolate(tensor, **kwargs)

    def forward(self, features):
        if len(features) != len(self.branch_channels):
            raise ValueError(
                f'CRCG expected {len(self.branch_channels)} branches, '
                f'got {len(features)}')
        for index, (feature, channels) in enumerate(
                zip(features, self.branch_channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f'CRCG branch {index} expected BCHW with {channels} '
                    f'channels, got {tuple(feature.shape)}')

        reference_size = features[0].shape[-2:]
        projected = [projection(feature)
                     for projection, feature in zip(self.projections, features)]
        aligned = [projected[0]] + [
            self._resize(feature, reference_size) for feature in projected[1:]
        ]
        normalized_reference = F.normalize(aligned[0].float(), dim=1)
        similarities = [
            (normalized_reference * F.normalize(feature.float(), dim=1))
            .sum(dim=1, keepdim=True)
            for feature in aligned[1:]
        ]
        if self.gate_type == 'sigmoid':
            reliabilities = [
                torch.sigmoid(self.temperature * score + self.gate_bias)
                for score in similarities]
        else:
            reliabilities = [(score + 1.0) * 0.5 for score in similarities]

        stacked = torch.stack(reliabilities, dim=0)
        if self.consensus_reduce == 'mean':
            reliability = stacked.mean(dim=0)
        else:
            weights = self.consensus_logits.softmax(dim=0)
            reliability = (stacked * weights[:, None, None, None, None]).sum(0)
        reliability = reliability.to(dtype=features[0].dtype)
        if self.mode == 'gate_self':
            residual = reliability * self.detail(features[0])
        else:
            cross = torch.stack([
                gate.to(aligned_feature.dtype) * aligned_feature
                for gate, aligned_feature in zip(reliabilities, aligned[1:])
            ], dim=0).mean(dim=0)
            residual = self.cross_out(cross.to(features[0].dtype))
        refined = features[0] + self.alpha.to(features[0].dtype) * residual

        outputs = list(features)
        outputs[0] = refined
        self.last_branch_shapes = [tuple(feature.shape) for feature in features]
        self.last_reliability_shape = tuple(reliability.shape)
        if self.debug and not self._debug_printed and _rank_zero():
            if not torch.isfinite(reliability).all():
                raise FloatingPointError('CRCG reliability contains NaN or Inf')
            self.last_reliability_stats = (
                float(reliability.detach().min()),
                float(reliability.detach().max()),
                float(reliability.detach().mean()),
            )
            strides = [4 * (2 ** index) for index in range(len(features))]
            print('CRCG enabled: '
                  f'branches={self.last_branch_shapes}, strides={strides}, '
                  f'embed_dim={self.embed_dim}, '
                  f'reliability[min,max,mean]={self.last_reliability_stats}, '
                  f'alpha={float(self.alpha.detach())}')
            self._debug_printed = True
        return outputs, reliability
