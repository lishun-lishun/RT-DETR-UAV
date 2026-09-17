"""Sparse-Evidence Concentration Downsampling stage-transition bypass."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormLayer


class SECDTransition(nn.Module):
    """Preserve sparse four-phase residuals without replacing a backbone stage.

    ``forward`` returns only the scaled bypass; the caller adds it to the
    original, complete stage output. Odd right/bottom edges are replicated so
    that the spatial result agrees with the backbone's ceil stride-2 shape.
    BatchNorm follows PResNet's normal recursive freeze_norm conversion.
    """

    def __init__(self, ch_in, ch_out, groups=8, temperature=0.1,
                 alpha_init=0.0, alpha_max=0.20, eps=1e-6,
                 alpha_mode='bounded_tanh'):
        super().__init__()
        if not isinstance(groups, int) or groups <= 0 or ch_in % groups:
            raise ValueError('SECD requires positive groups and ch_in % groups == 0')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('SECD temperature must be finite and positive')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('SECD eps must be finite and positive')
        if not math.isfinite(alpha_init) or not math.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError('SECD alpha_init must be finite and alpha_max positive')
        if alpha_mode != 'bounded_tanh':
            raise ValueError('SECD supports only alpha_mode=bounded_tanh')

        self.ch_in = ch_in
        self.groups = groups
        self.temperature = temperature
        self.eps = eps
        self.alpha_max = alpha_max
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.projection = ConvNormLayer(ch_in, ch_out, 1, 1, act=None)

    @property
    def alpha_eff(self):
        return self.alpha_max * self.raw_alpha.tanh()

    def sparse_evidence(self, x):
        """Return unprojected S and group concentration kappa for diagnostics.

        S has shape [B,C,ceil(H/2),ceil(W/2)] and kappa [B,G,h,w].
        Low-precision energy/softmax math is promoted to float32. Keeping eps
        inside sqrt avoids undefined gradients at uniform/zero residuals.
        """
        if x.ndim != 4 or x.shape[1] != self.ch_in:
            raise ValueError('SECD input must be NCHW with the configured channels')
        if not x.is_floating_point():
            raise ValueError('SECD input must be floating point')
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        height, width = work.shape[-2:]
        if height % 2 or width % 2:
            work = F.pad(work, (0, width % 2, 0, height % 2), mode='replicate')

        phases = torch.stack((work[:, :, 0::2, 0::2],
                              work[:, :, 0::2, 1::2],
                              work[:, :, 1::2, 0::2],
                              work[:, :, 1::2, 1::2]), dim=2)
        residual = phases - phases.mean(dim=2, keepdim=True)
        batch, channels, _, height, width = residual.shape
        residual = residual.reshape(batch, self.groups, channels // self.groups,
                                    4, height, width)
        energy = (residual.square().mean(dim=2) + self.eps).sqrt()
        probability = (energy / self.temperature).softmax(dim=2)
        kappa = ((4 * probability.square().sum(dim=2) - 1) / 3).clamp(0.0, 1.0)
        dense_residual = (probability.unsqueeze(2) * residual).sum(dim=3)
        sparse = (kappa.unsqueeze(2) * dense_residual).reshape(batch, channels,
                                                            height, width)
        return sparse.to(dtype=x.dtype), kappa

    def forward(self, x):
        sparse, _ = self.sparse_evidence(x)
        evidence = self.projection(sparse)
        return self.alpha_eff * evidence
