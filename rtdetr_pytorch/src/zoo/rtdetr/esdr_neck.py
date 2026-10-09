"""Extrema-Sensitive Downsampling Residual block for RT-DETR.

ESDR augments an already-computed, original bottom-up downsampling result
without replacing or otherwise changing that path.  Given a source feature
``X`` and the corresponding original result ``D_base``, it computes::

    D_ext = Conv1x1(MaxPool2d(kernel_size=2, stride=2)(X))
    D = D_base + beta * D_ext

``beta`` is a bounded, learnable scale for every channel.  The block contains
no normalization, attention, or gating operation.  Callers should create two
independent instances for the P3-to-P4 and P4-to-P5 bottom-up transitions.
"""

import math

import torch
import torch.nn as nn


__all__ = ['ExtremaSensitiveDownsample', 'ESDRNeck']


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
    maximum = _finite_float('ESDR.beta_max', maximum)
    initial = _finite_float('ESDR.beta_init', initial)
    if maximum <= 0.0:
        raise ValueError('ESDR.beta_max must be positive')
    if abs(initial) >= maximum:
        raise ValueError(
            'abs(ESDR.beta_init) must be smaller than ESDR.beta_max')
    return maximum, math.atanh(initial / maximum)


class ExtremaSensitiveDownsample(nn.Module):
    """Inject max-pooled extrema into an original downsampling result.

    Args:
        hidden_dim: Channel count of both ``source`` and ``base``.
        beta_max: Strict bound used by the per-channel LayerScale.
        beta_init: Effective initial LayerScale value.  Its raw parameter is
            initialized with the exact inverse hyperbolic tangent.

    ``forward`` requires the source spatial dimensions to be even.  ``base``
    must have the exact shape ``[B, C, H // 2, W // 2]`` and share the source
    tensor's device and dtype.  These checks prevent accidental broadcasting
    or an unintended replacement of the original bottom-up path.
    """

    def __init__(self, hidden_dim=256, beta_max=0.20, beta_init=0.02):
        super().__init__()
        self.hidden_dim = _positive_int('ESDR.hidden_dim', hidden_dim)
        self.beta_max, raw_init = _bounded_raw_init(beta_max, beta_init)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.project = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.raw_beta = nn.Parameter(torch.full(
            (1, self.hidden_dim, 1, 1), raw_init))

    def effective_beta(self):
        """Return the signed per-channel scale bounded by ``beta_max``."""
        return self.beta_max * torch.tanh(self.raw_beta)

    def _validate_inputs(self, source, base):
        if not torch.is_tensor(source) or source.ndim != 4:
            raise RuntimeError('ESDR source must be a BCHW tensor')
        if source.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f'ESDR source has {source.shape[1]} channels, '
                f'expected {self.hidden_dim}')
        if not source.is_floating_point() or source.is_complex():
            raise RuntimeError(
                'ESDR source must use a real floating-point dtype')

        height, width = source.shape[-2:]
        if height < 2 or width < 2 or height % 2 or width % 2:
            raise RuntimeError(
                'ESDR source height and width must be positive even '
                'dimensions of at least 2')

        if not torch.is_tensor(base) or base.ndim != 4:
            raise RuntimeError('ESDR base must be a BCHW tensor')
        expected_shape = (
            source.shape[0], self.hidden_dim, height // 2, width // 2)
        if tuple(base.shape) != expected_shape:
            raise RuntimeError(
                f'ESDR base shape must be {expected_shape}, '
                f'got {tuple(base.shape)}')
        if base.device != source.device:
            raise RuntimeError(
                'ESDR source and base must be on the same device')
        if base.dtype != source.dtype:
            raise RuntimeError(
                'ESDR source and base must use the same dtype')
        if not base.is_floating_point() or base.is_complex():
            raise RuntimeError(
                'ESDR base must use a real floating-point dtype')

    def forward(self, source, base, return_aux=False):
        """Return ``base + beta * projected_max_pool(source)``.

        ``base`` is supplied by the unchanged original stride-2 convolution;
        ESDR never computes ``D_ext - D_base``.
        """
        self._validate_inputs(source, base)

        max_pooled = self.max_pool(source)
        extrema = self.project(max_pooled)
        # Retain an FP32 master LayerScale while respecting autocast output.
        beta = self.effective_beta().to(dtype=extrema.dtype)
        output = base + beta * extrema

        if not return_aux:
            return output
        return output, {
            'max_pooled': max_pooled,
            'extrema': extrema,
            'beta': beta,
        }


# Concise alias matching the experiment/configuration name.
ESDRNeck = ExtremaSensitiveDownsample
