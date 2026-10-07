"""Frequency-Decoupled Context Residual block for RT-DETR's neck.

FDCR is a same-resolution post-fusion refinement block.  It decomposes an
input feature into a 3x3 average-pooled low-frequency component and its
high-frequency residual, processes the two components independently, and
injects their fused response through a bounded per-channel LayerScale.

The block deliberately contains no normalization, attention, or gate.  It is
also backbone-agnostic: callers can create independent instances for the N3
and N4 outputs of any HybridEncoder.
"""

import math

import torch
import torch.nn as nn


__all__ = ['FrequencyDecoupledContextResidual', 'FDCRNeck']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _bounded_raw_init(maximum, initial):
    maximum = _finite_float('FDCR.gamma_max', maximum)
    initial = _finite_float('FDCR.gamma_init', initial)
    if maximum <= 0.0:
        raise ValueError('FDCR.gamma_max must be positive')
    if abs(initial) >= maximum:
        raise ValueError(
            'abs(FDCR.gamma_init) must be smaller than FDCR.gamma_max')
    return maximum, math.atanh(initial / maximum)


class FrequencyDecoupledContextResidual(nn.Module):
    """Refine one feature map through same-scale frequency decomposition.

    Given ``F`` with shape ``[B, C, H, W]``, the block computes::

        F_low = AvgPool3x3(F)
        F_high = F - F_low
        L_context = Conv1x1(DWConv5x5(F_low))
        H_detail = Conv1x1(DWConv3x3(F_high))
        residual = Conv1x1(cat([H_detail, L_context]))
        output = F + gamma * residual

    Here ``gamma = gamma_max * tanh(raw_gamma)`` is independently learnable
    for every channel and has shape ``[1, C, 1, 1]``.  Setting ``raw_gamma``
    to zero therefore restores the input exactly.
    """

    def __init__(self, hidden_dim=256, gamma_max=0.30, gamma_init=0.05):
        super().__init__()
        self.hidden_dim = _positive_int('FDCR.hidden_dim', hidden_dim)
        self.gamma_max, raw_init = _bounded_raw_init(
            gamma_max, gamma_init)

        self.low_pass = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.low_depthwise = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1,
            padding=2, groups=self.hidden_dim, bias=True)
        self.low_project = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, bias=True)
        self.high_depthwise = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1,
            padding=1, groups=self.hidden_dim, bias=True)
        self.high_project = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, bias=True)
        self.fuse = nn.Conv2d(
            2 * self.hidden_dim, self.hidden_dim, kernel_size=1, bias=True)

        self.raw_gamma = nn.Parameter(torch.full(
            (1, self.hidden_dim, 1, 1), raw_init))

    def effective_gamma(self):
        """Return the signed channel-wise scale bounded by ``gamma_max``."""
        return self.gamma_max * torch.tanh(self.raw_gamma)

    def _validate_input(self, features):
        if not torch.is_tensor(features) or features.ndim != 4:
            raise RuntimeError('FDCR input must be a BCHW tensor')
        if features.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f'FDCR input has {features.shape[1]} channels, '
                f'expected {self.hidden_dim}')
        if not features.is_floating_point() or features.is_complex():
            raise RuntimeError(
                'FDCR input must use a real floating-point dtype')

    def forward(self, features, return_aux=False):
        self._validate_input(features)

        low_frequency = self.low_pass(features)
        high_frequency = features - low_frequency
        low_context = self.low_project(
            self.low_depthwise(low_frequency))
        high_detail = self.high_project(
            self.high_depthwise(high_frequency))
        residual = self.fuse(torch.cat(
            (high_detail, low_context), dim=1))

        # Keep the feature computation in its active (possibly autocast)
        # dtype while retaining an FP32 master LayerScale parameter.
        gamma = self.effective_gamma().to(dtype=residual.dtype)
        output = features + gamma * residual

        if not return_aux:
            return output
        return output, {
            'low_frequency': low_frequency,
            'high_frequency': high_frequency,
            'low_context': low_context,
            'high_detail': high_detail,
            'residual': residual,
            'gamma': gamma,
        }


# Concise alias matching the experiment/configuration name.
FDCRNeck = FrequencyDecoupledContextResidual
