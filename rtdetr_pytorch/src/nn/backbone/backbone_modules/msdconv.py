"""Multi-scale difference-of-Gaussian feature refinement for PResNet."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..common import ConvNormLayer


__all__ = ['MSDConv']


def _binomial_kernel(coefficients):
    vector = torch.tensor(coefficients, dtype=torch.float32)
    vector = vector / vector.sum()
    kernel = vector[:, None] * vector[None, :]
    return kernel.reshape(1, 1, len(coefficients), len(coefficients))


class MSDConv(nn.Module):
    """Route fixed scale-space H/M bands while using L only as context.

    G1 uses normalized separable 3x3 binomial [1,2,1]. G2 uses normalized
    5x5 [1,4,6,4,1]. G3 applies the same 5x5 low-pass once more to G2.
    """

    def __init__(self, channels, groups=8, eps=1e-6, beta_init=0.1,
                 beta_max=0.5, debug=False):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError('MSDConv channels must be a positive integer')
        if not isinstance(groups, int) or isinstance(groups, bool) or groups < 1:
            raise ValueError('MSDConv groups must be a positive integer')
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError('MSDConv eps must be finite and positive')
        if not math.isfinite(float(beta_max)) or float(beta_max) <= 0:
            raise ValueError('MSDConv beta_max must be finite and positive')
        if not 0.0 < float(beta_init) < float(beta_max):
            raise ValueError('MSDConv beta_init must be in (0, beta_max)')
        if not isinstance(debug, bool):
            raise ValueError('MSDConv debug must be boolean')

        self.channels = channels
        self.groups = max(divisor for divisor in range(1, min(groups, channels) + 1)
                          if channels % divisor == 0)
        self.eps = float(eps)
        self.beta_max = float(beta_max)
        ratio = float(beta_init) / float(beta_max)
        self.raw_beta = nn.Parameter(torch.tensor(math.atanh(ratio)))
        self.register_buffer('kernel3', _binomial_kernel([1, 2, 1]))
        self.register_buffer('kernel5', _binomial_kernel([1, 4, 6, 4, 1]))
        self.context_router = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.projection = ConvNormLayer(channels, channels, 1, 1, act=None)
        self.debug = debug
        self.last_debug_stats = None
        self.last_debug_tensors = None

    @property
    def beta(self):
        return self.beta_max * self.raw_beta.float().tanh()

    @staticmethod
    def _lowpass(x, kernel):
        radius = kernel.shape[-1] // 2
        padded = F.pad(x, (radius, radius, radius, radius), mode='replicate')
        weight = kernel.to(dtype=x.dtype).expand(x.shape[1], 1, -1, -1)
        return F.conv2d(padded, weight, groups=x.shape[1])

    def scale_space(self, x):
        """Return differentiable G/H/M/L bands and group-wise routing maps."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError('MSDConv expects NCHW input with configured channels')
        if x.shape[-2] < 1 or x.shape[-1] < 1:
            raise ValueError('MSDConv requires nonempty spatial dimensions')

        g1 = self._lowpass(x, self.kernel3)
        g2 = self._lowpass(x, self.kernel5)
        g3 = self._lowpass(g2, self.kernel5)
        high = g1 - g2
        medium = g2 - g3
        low = g3

        batch, channels, height, width = high.shape
        per_group = channels // self.groups
        high_group = high.float().reshape(batch, self.groups, per_group, height, width)
        medium_group = medium.float().reshape(batch, self.groups, per_group, height, width)
        energy_h = (high_group.square().mean(dim=2, keepdim=True) + self.eps).sqrt()
        energy_m = (medium_group.square().mean(dim=2, keepdim=True) + self.eps).sqrt()
        denominator = energy_h + energy_m + self.eps
        weight_h = energy_h / denominator
        weight_m = energy_m / denominator
        target = (weight_h * high_group + weight_m * medium_group).reshape(
            batch, channels, height, width).to(dtype=x.dtype)

        target_strength = target.abs().mean(dim=1, keepdim=True)
        context_strength = low.abs().mean(dim=1, keepdim=True)
        eta = self.context_router(torch.cat((target_strength, context_strength), dim=1)).sigmoid()
        routed = eta * target
        return {'G1': g1, 'G2': g2, 'G3': g3, 'H': high, 'M': medium, 'L': low,
                'eH': energy_h, 'eM': energy_m, 'wH': weight_h, 'wM': weight_m,
                'T': target, 'eta': eta, 'routed': routed}

    def forward(self, x):
        fields = self.scale_space(x)
        projected = self.projection(fields['routed'])
        beta = self.beta.to(dtype=projected.dtype)
        output = x + beta * projected

        if self.debug:
            with torch.no_grad():
                eta = fields['eta'].detach().float().flatten()
                input_norm = torch.linalg.vector_norm(x.detach().float())
                residual_norm = torch.linalg.vector_norm((beta * projected).detach().float())
                self.last_debug_stats = {
                    'H_norm': torch.linalg.vector_norm(fields['H'].detach().float()),
                    'M_norm': torch.linalg.vector_norm(fields['M'].detach().float()),
                    'L_norm': torch.linalg.vector_norm(fields['L'].detach().float()),
                    'eH_mean': fields['eH'].detach().float().mean(),
                    'eM_mean': fields['eM'].detach().float().mean(),
                    'wH_mean': fields['wH'].detach().float().mean(),
                    'wM_mean': fields['wM'].detach().float().mean(),
                    'eta_mean': eta.mean(),
                    'eta_p10': torch.quantile(eta, 0.10),
                    'eta_p50': torch.quantile(eta, 0.50),
                    'eta_p90': torch.quantile(eta, 0.90),
                    'beta': self.beta.detach(),
                    'residual_input_norm_ratio': residual_norm / (input_norm + self.eps),
                }
                self.last_debug_tensors = {
                    'H_energy': fields['H'].detach().float().square().mean(1).sqrt(),
                    'M_energy': fields['M'].detach().float().square().mean(1).sqrt(),
                    'wH': fields['wH'].detach().float().mean(2),
                    'wM': fields['wM'].detach().float().mean(2),
                    'eta': fields['eta'].detach().float().squeeze(1),
                }
        return output

