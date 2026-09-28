"""Sub-cell Localization Relay (SLR) neck for RT-DETR.

SLR consumes a stride-4 detail source only after the original HybridEncoder
has produced N3/N4/N5.  It modifies N3 with one localization residual and does
not create a detector level, alter CCFF, or touch N4/N5.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SLRNeck']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _positive_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or float(value) <= 0.0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


class SLRNeck(nn.Module):
    """Relay phase-local stride-4 detail into the original stride-8 N3."""

    def __init__(self, hidden_dim, detail_source_channels, query_dim=32,
                 position_dim=16, alpha_max=0.3, alpha_init=0.05):
        super().__init__()
        hidden_dim = _positive_int('SLR.hidden_dim', hidden_dim)
        detail_source_channels = _positive_int(
            'SLR.detail_source_channels', detail_source_channels)
        query_dim = min(_positive_int('SLR.query_dim', query_dim), hidden_dim)
        position_dim = _positive_int('SLR.position_dim', position_dim)
        alpha_max = _positive_float('SLR.alpha_max', alpha_max)
        alpha_init = _positive_float('SLR.alpha_init', alpha_init)
        if alpha_init >= alpha_max:
            raise ValueError('SLR.alpha_init must be smaller than alpha_max')

        self.hidden_dim = hidden_dim
        self.detail_source_channels = detail_source_channels
        self.query_dim = query_dim
        self.alpha_max = alpha_max

        self.query_projection = nn.Conv2d(hidden_dim, query_dim, 1)
        self.key_projection = nn.Conv2d(
            detail_source_channels, query_dim, 1)
        self.value_projection = nn.Conv2d(
            detail_source_channels, query_dim, 1)
        self.position_projection = nn.Sequential(
            nn.Conv2d(2, position_dim, 1),
            nn.SiLU(inplace=True),
        )
        self.output_projection = nn.Conv2d(
            query_dim + position_dim, hidden_dim, 1)

        probability = alpha_init / alpha_max
        raw_init = math.log(probability / (1.0 - probability))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init))

        # Phase order follows pixel_unshuffle: top-left, top-right,
        # bottom-left, bottom-right. Coordinates are explicitly (x, y).
        positions_xy = torch.tensor([
            [-0.25, -0.25],
            [+0.25, -0.25],
            [-0.25, +0.25],
            [+0.25, +0.25],
        ]).view(1, 4, 2, 1, 1)
        self.register_buffer('positions_xy', positions_xy, persistent=False)

    def effective_alpha(self):
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def _phase_features(self, detail):
        if detail.ndim != 4:
            raise ValueError('SLR detail source must be a BCHW tensor')
        if detail.shape[1] != self.detail_source_channels:
            raise RuntimeError(
                f'SLR detail source has {detail.shape[1]} channels, expected '
                f'{self.detail_source_channels}')
        if detail.shape[-2] % 2 or detail.shape[-1] % 2:
            raise RuntimeError(
                'SLR stride-4 detail height and width must both be even for '
                'lossless 2x2 pixel unshuffle')
        batch, channels, _, _ = detail.shape
        unshuffled = F.pixel_unshuffle(detail, 2)
        height, width = unshuffled.shape[-2:]
        # pixel_unshuffle groups the four offsets inside each source channel;
        # transpose them into an explicit four-phase dimension.
        return (unshuffled.reshape(batch, channels, 4, height, width)
                .permute(0, 2, 1, 3, 4).contiguous())

    @staticmethod
    def _project_phases(projection, phases):
        batch, count, channels, height, width = phases.shape
        projected = projection(phases.reshape(
            batch * count, channels, height, width))
        return projected.reshape(
            batch, count, projected.shape[1], height, width)

    def forward(self, n3, detail, return_aux=False):
        if n3.ndim != 4 or n3.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f'SLR N3 must be BCHW with {self.hidden_dim} channels')
        phases = self._phase_features(detail)
        if phases.shape[0] != n3.shape[0] or phases.shape[-2:] != n3.shape[-2:]:
            raise RuntimeError(
                'SLR detail/N3 alignment mismatch after pixel_unshuffle: '
                f'detail phases={tuple(phases.shape)}, N3={tuple(n3.shape)}')

        shared = phases.mean(dim=1, keepdim=True)
        local_residuals = phases - shared

        query = self.query_projection(n3)
        keys = self._project_phases(self.key_projection, phases)
        values = self._project_phases(
            self.value_projection, local_residuals)

        # Dot-product statistics and four-way softmax remain FP32 under AMP.
        scores = (query.float().unsqueeze(1) * keys.float()).sum(dim=2)
        scores = scores / math.sqrt(self.query_dim)
        attention = torch.softmax(scores, dim=1)
        local_detail = (attention.unsqueeze(2).to(values.dtype) * values).sum(dim=1)

        positions = self.positions_xy.to(device=attention.device,
                                         dtype=attention.dtype)
        delta_position = (attention.unsqueeze(2) * positions).sum(dim=1)
        position_embedding = self.position_projection(
            delta_position.to(dtype=n3.dtype))
        localization = self.output_projection(torch.cat(
            [local_detail, position_embedding], dim=1))
        alpha = self.effective_alpha()
        enhanced = n3 + alpha.to(dtype=localization.dtype) * localization

        if not return_aux:
            return enhanced
        return enhanced, {
            'attention': attention,
            'delta_position_xy': delta_position,
            'local_detail': local_detail,
            'localization_residual': localization,
            'alpha_eff': alpha,
        }
