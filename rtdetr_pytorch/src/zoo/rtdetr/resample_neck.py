"""Learnable residual resampling blocks for RT-DETR's HybridEncoder.

The two blocks in this module deliberately preserve the original resampling
result as an explicit ``base`` path:

* :class:`LearnablePixelReassemblyUpsample` augments nearest-neighbour
  upsampling with a PixelShuffle reconstruction residual.
* :class:`SubpixelPreservingDownsample` augments the original stride-2
  convolution with a PixelUnshuffle information-preserving residual.

Neither block contains normalization, attention, a spatial gate, nor a
backbone-specific assumption.  Setting its effective channel-wise residual
scale to zero restores the supplied base tensor exactly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'LearnablePixelReassemblyUpsample',
    'SubpixelPreservingDownsample',
]


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _bounded_raw_init(name, maximum, initial):
    maximum = _finite_float(f'{name}_max', maximum)
    initial = _finite_float(f'{name}_init', initial)
    if maximum <= 0.0:
        raise ValueError(f'{name}_max must be positive')
    if abs(initial) >= maximum:
        raise ValueError(
            f'abs({name}_init) must be smaller than {name}_max')
    return maximum, math.atanh(initial / maximum)


def _validate_feature(name, value, channels):
    if not torch.is_tensor(value) or value.ndim != 4:
        raise RuntimeError(f'{name} must be a BCHW tensor')
    if value.shape[1] != channels:
        raise RuntimeError(
            f'{name} has {value.shape[1]} channels, expected {channels}')
    if not value.is_floating_point() or value.is_complex():
        raise RuntimeError(f'{name} must use a real floating-point dtype')


def _validate_base(name, base, source, channels, spatial_size):
    _validate_feature(name, base, channels)
    if base.shape[0] != source.shape[0]:
        raise RuntimeError(f'{name} and source must share a batch size')
    if tuple(base.shape[-2:]) != tuple(spatial_size):
        raise RuntimeError(
            f'{name} spatial size {tuple(base.shape[-2:])} does not match '
            f'the learned path {tuple(spatial_size)}')
    if base.device != source.device:
        raise RuntimeError(f'{name} and source must be on the same device')
    if base.dtype != source.dtype:
        raise RuntimeError(f'{name} and source must use the same dtype')


class LearnablePixelReassemblyUpsample(nn.Module):
    """Nearest baseline plus a bounded PixelShuffle reconstruction residual.

    For ``x`` with shape ``[B, C, H, W]``, the learned path is::

        Conv1x1(C, 4C) -> PixelShuffle(2)
        -> DepthwiseConv3x3(C) -> Conv1x1(C, C)

    and the final output is ``base + alpha * (learned - base)``, where
    ``alpha = alpha_max * tanh(raw_alpha)`` has shape ``[1, C, 1, 1]``.
    Passing HybridEncoder's already-computed nearest tensor as ``base`` keeps
    the original RT-DETR path explicit.  If it is omitted, this module creates
    the same ``scale_factor=2, mode='nearest'`` baseline itself.
    """

    def __init__(self, channels=256, alpha_max=0.5, alpha_init=0.05,
                 debug=False):
        super().__init__()
        self.channels = _positive_int('LPRU.channels', channels)
        self.alpha_max, raw_init = _bounded_raw_init(
            'LPRU.alpha', alpha_max, alpha_init)
        if not isinstance(debug, bool):
            raise ValueError('LPRU.debug must be a boolean')
        self.debug = debug

        self.expand = nn.Conv2d(
            self.channels, 4 * self.channels, kernel_size=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.depthwise = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, stride=1,
            padding=1, groups=self.channels, bias=True)
        self.project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=True)
        self.raw_alpha = nn.Parameter(torch.full(
            (1, self.channels, 1, 1), raw_init))
        self.last_debug_stats = {}

    def effective_alpha(self):
        """Return the signed channel-wise scale bounded by ``alpha_max``."""
        return self.alpha_max * torch.tanh(self.raw_alpha)

    def forward(self, x, base=None, return_aux=False):
        _validate_feature('LPRU source', x, self.channels)
        if base is None:
            base = F.interpolate(x, scale_factor=2.0, mode='nearest')

        rearranged = self.pixel_shuffle(self.expand(x))
        learned = self.project(self.depthwise(rearranged))
        _validate_base(
            'LPRU base', base, x, self.channels, learned.shape[-2:])

        residual = learned - base
        # Keep autocast features in their compute dtype. The FP32 master raw
        # parameter still receives gradients through this differentiable cast.
        alpha = self.effective_alpha().to(dtype=residual.dtype)
        output = base + alpha * residual

        if self.debug:
            with torch.no_grad():
                eps = torch.finfo(output.dtype).eps
                self.last_debug_stats = {
                    'alpha_mean': alpha.detach().mean(),
                    'alpha_min': alpha.detach().amin(),
                    'alpha_max': alpha.detach().amax(),
                    'residual_to_base_norm_ratio': (
                        residual.detach().float().norm()
                        / base.detach().float().norm().clamp_min(eps)),
                }

        if not return_aux:
            return output
        return output, {
            'base': base,
            'rearranged': rearranged,
            'learned': learned,
            'residual': residual,
            'alpha': alpha,
        }


class SubpixelPreservingDownsample(nn.Module):
    """Stride-2 convolution baseline plus a PixelUnshuffle residual path.

    The learned information-preserving path is::

        PixelUnshuffle(2) -> Conv1x1(4C, C)
        -> DepthwiseConv3x3(C) -> Conv1x1(C, C)

    and the output is ``base + beta * (preserved - base)``, where ``base`` is
    the unmodified result of HybridEncoder's original ``downsample_conv`` and
    ``beta = beta_max * tanh(raw_beta)`` is channel-wise.  Even input spatial
    dimensions are required because PixelUnshuffle performs a lossless 2x2
    space-to-channel rearrangement without cropping or padding.
    """

    def __init__(self, channels=256, beta_max=0.5, beta_init=0.05,
                 debug=False):
        super().__init__()
        self.channels = _positive_int('SPDR.channels', channels)
        self.beta_max, raw_init = _bounded_raw_init(
            'SPDR.beta', beta_max, beta_init)
        if not isinstance(debug, bool):
            raise ValueError('SPDR.debug must be a boolean')
        self.debug = debug

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.compress = nn.Conv2d(
            4 * self.channels, self.channels, kernel_size=1, bias=True)
        self.depthwise = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, stride=1,
            padding=1, groups=self.channels, bias=True)
        self.project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=True)
        self.raw_beta = nn.Parameter(torch.full(
            (1, self.channels, 1, 1), raw_init))
        self.last_debug_stats = {}

    def effective_beta(self):
        """Return the signed channel-wise scale bounded by ``beta_max``."""
        return self.beta_max * torch.tanh(self.raw_beta)

    def forward(self, x, base, return_aux=False):
        _validate_feature('SPDR source', x, self.channels)
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise RuntimeError(
                'SPDR source height and width must be even; cropping and '
                'padding are intentionally forbidden')

        rearranged = self.pixel_unshuffle(x)
        preserved = self.project(self.depthwise(self.compress(rearranged)))
        _validate_base(
            'SPDR base', base, x, self.channels, preserved.shape[-2:])

        residual = preserved - base
        # Avoid promoting the complete feature map back to FP32 under AMP.
        beta = self.effective_beta().to(dtype=residual.dtype)
        output = base + beta * residual

        if self.debug:
            with torch.no_grad():
                eps = torch.finfo(output.dtype).eps
                self.last_debug_stats = {
                    'beta_mean': beta.detach().mean(),
                    'beta_min': beta.detach().amin(),
                    'beta_max': beta.detach().amax(),
                    'residual_to_base_norm_ratio': (
                        residual.detach().float().norm()
                        / base.detach().float().norm().clamp_min(eps)),
                }

        if not return_aux:
            return output
        return output, {
            'base': base,
            'rearranged': rearranged,
            'preserved': preserved,
            'residual': residual,
            'beta': beta,
        }
