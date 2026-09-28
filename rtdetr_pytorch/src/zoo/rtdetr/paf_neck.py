"""Phase-adaptive top-down alignment for the RT-DETR HybridEncoder.

The original HybridEncoder still performs nearest-neighbour upsampling.  This
module only selects, independently at every spatial location, a convex
combination of four zero-padded spatial phases of that upsampled feature.  It
does not gate feature amplitude, alter the feature shape, or replace the
existing CSPRep fusion block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['PhaseAdaptiveFusion']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class PhaseAdaptiveFusion(nn.Module):
    """Align one upsampled deep feature to a same-resolution shallow feature.

    Phase indices are ``(dy, dx)`` in the fixed order ``00, 01, 10, 11``.
    Positive offsets shift the upsampled feature right/down with zeros entering
    from the left/top.  Padding followed by cropping is deliberately used
    instead of ``torch.roll`` so values never wrap around an image boundary.
    """

    def __init__(self, hidden_dim, query_dim=32, debug=False,
                 debug_interval=100):
        super().__init__()
        self.hidden_dim = _positive_int('PAF.hidden_dim', hidden_dim)
        self.query_dim = _positive_int('PAF.query_dim', query_dim)
        if not isinstance(debug, bool):
            raise ValueError('PAF.debug must be boolean')
        self.debug_interval = _positive_int(
            'PAF.debug_interval', debug_interval)
        self.debug = debug

        # No normalization is added: PAF only needs compact query/key
        # projections to choose a spatial phase.  Values remain the original
        # hidden-dimensional upsampled features.
        self.query_projection = nn.Conv2d(
            self.hidden_dim, self.query_dim, kernel_size=1)
        self.key_projection = nn.Conv2d(
            self.hidden_dim, self.query_dim, kernel_size=1)

        self._debug_step = 0
        self.last_debug_stats = None

    @staticmethod
    def _zero_shift(value, dy, dx):
        """Shift right/down using padding and crop, without wrap-around."""
        if dy not in (0, 1) or dx not in (0, 1):
            raise ValueError('PAF phase offsets must be zero or one')
        if dy == 0 and dx == 0:
            return value
        height, width = value.shape[-2:]
        padded = F.pad(value, (dx, 0, dy, 0), mode='constant', value=0.0)
        return padded[..., :height, :width]

    @classmethod
    def phase_candidates(cls, upsampled):
        """Return ``[B, 4, C, H, W]`` phases in 00/01/10/11 order."""
        if not torch.is_tensor(upsampled) or upsampled.ndim != 4:
            raise ValueError('PAF upsampled feature must be a BCHW tensor')
        return torch.stack([
            cls._zero_shift(upsampled, 0, 0),
            cls._zero_shift(upsampled, 0, 1),
            cls._zero_shift(upsampled, 1, 0),
            cls._zero_shift(upsampled, 1, 1),
        ], dim=1)

    def _project_keys(self, candidates):
        batch, count, channels, height, width = candidates.shape
        keys = self.key_projection(candidates.reshape(
            batch * count, channels, height, width))
        return keys.reshape(batch, count, self.query_dim, height, width)

    @staticmethod
    def _debug_statistics(attention):
        detached = attention.detach().float()
        entropy = -(detached * detached.clamp_min(1e-12).log()).sum(dim=1)
        means = detached.mean(dim=(0, 2, 3))
        return {
            'phase00_mean': means[0].item(),
            'phase01_mean': means[1].item(),
            'phase10_mean': means[2].item(),
            'phase11_mean': means[3].item(),
            'phase_entropy': entropy.mean().item(),
        }

    def forward(self, shallow, upsampled, return_aux=False):
        if (not torch.is_tensor(shallow) or not torch.is_tensor(upsampled)
                or shallow.ndim != 4 or upsampled.ndim != 4):
            raise ValueError('PAF inputs must be BCHW tensors')
        if shallow.shape != upsampled.shape:
            raise RuntimeError(
                'PAF requires same-shaped shallow and upsampled features, '
                f'got {tuple(shallow.shape)} and {tuple(upsampled.shape)}')
        if shallow.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f'PAF expected {self.hidden_dim} channels, got '
                f'{shallow.shape[1]}')

        candidates = self.phase_candidates(upsampled)
        query = self.query_projection(shallow)
        keys = self._project_keys(candidates)

        # Similarity statistics and the four-way softmax stay FP32 under AMP.
        scores = (query.float().unsqueeze(1) * keys.float()).sum(dim=2)
        scores = scores / math.sqrt(self.query_dim)
        attention = torch.softmax(scores, dim=1)
        aligned = (attention.to(dtype=candidates.dtype).unsqueeze(2)
                   * candidates).sum(dim=1)

        if self.debug:
            if self._debug_step % self.debug_interval == 0:
                self.last_debug_stats = self._debug_statistics(attention)
            self._debug_step += 1

        if not return_aux:
            return aligned
        return aligned, {
            'attention': attention,
            'scores': scores,
            'candidates': candidates,
        }

