"""Partial Spatial Context Attention for RT-DETR's fused neck features.

PSCA keeps most channels on an identity path and applies compressed
single-head spatial attention only to the final context-channel slice.  It is
designed as a same-resolution post-CCFF refinement: HybridEncoder can create
independent instances for N3 and N4 with different key/value pooling strides.

Only the QK logits and softmax are evaluated in FP32 for numerical stability.
The projections, AV product, residual update, and optional fixed channel
shuffle remain in the active (possibly autocast) dtype.
"""

import math

import torch
import torch.nn as nn


__all__ = ['PartialSpatialContextAttention', 'PSCANeck']


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
    maximum = _finite_float('PSCA.alpha_max', maximum)
    initial = _finite_float('PSCA.alpha_init', initial)
    if maximum <= 0.0:
        raise ValueError('PSCA.alpha_max must be positive')
    if abs(initial) >= maximum:
        raise ValueError(
            'abs(PSCA.alpha_init) must be smaller than PSCA.alpha_max')
    return maximum, math.atanh(initial / maximum)


def _channel_shuffle(features, groups):
    """Deterministically interleave channel groups without parameters."""
    batch, channels, height, width = features.shape
    features = features.reshape(
        batch, groups, channels // groups, height, width)
    features = features.transpose(1, 2).contiguous()
    return features.reshape(batch, channels, height, width)


class PartialSpatialContextAttention(nn.Module):
    """Refine a contiguous context-channel slice with pooled spatial attention.

    For the default ``context_ratio=0.25``, an input ``F`` is split in channel
    order as ``P=F[:, :3C/4]`` and ``Ctx=F[:, 3C/4:]``.  Queries use the
    original spatial grid, while keys and values use an average-pooled grid::

        A = softmax(Q(Ctx)^T K(pool(Ctx)) / sqrt(d))
        C_att = A V(pool(Ctx))
        Ctx' = Ctx + alpha * W_o(C_att)

    ``alpha = alpha_max * tanh(raw_alpha)`` is a bounded per-context-channel
    LayerScale with shape ``[1, Cc, 1, 1]``.
    The untouched partial slice and refined context slice are concatenated;
    an optional fixed two-group channel shuffle then interleaves both slices.
    """

    def __init__(self, hidden_dim=256, context_ratio=0.25,
                 attention_dim=32, pool_stride=4, alpha_max=0.20,
                 alpha_init=0.02, channel_shuffle=True, shuffle_groups=2):
        super().__init__()
        self.hidden_dim = _positive_int('PSCA.hidden_dim', hidden_dim)
        self.attention_dim = _positive_int(
            'PSCA.attention_dim', attention_dim)
        self.pool_stride = _positive_int('PSCA.pool_stride', pool_stride)

        context_ratio = _finite_float(
            'PSCA.context_ratio', context_ratio)
        if not 0.0 < context_ratio < 1.0:
            raise ValueError('PSCA.context_ratio must be between 0 and 1')
        self.context_ratio = context_ratio
        self.context_channels = int(self.hidden_dim * self.context_ratio)
        if not 0 < self.context_channels < self.hidden_dim:
            raise ValueError(
                'PSCA.context_ratio must allocate at least one channel to '
                'both the partial and context paths')
        self.partial_channels = self.hidden_dim - self.context_channels

        if not isinstance(channel_shuffle, bool):
            raise ValueError('PSCA.channel_shuffle must be a boolean')
        self.channel_shuffle = channel_shuffle
        self.shuffle_groups = _positive_int(
            'PSCA.shuffle_groups', shuffle_groups)
        if self.channel_shuffle and self.hidden_dim % self.shuffle_groups != 0:
            raise ValueError(
                'PSCA.hidden_dim must be divisible by shuffle_groups when '
                'channel_shuffle is enabled')

        self.alpha_max, raw_init = _bounded_raw_init(
            alpha_max, alpha_init)
        self.scale = self.attention_dim ** -0.5

        self.context_pool = nn.AvgPool2d(
            kernel_size=self.pool_stride, stride=self.pool_stride)
        # Bias-free Q/K avoids a redundant key bias whose contribution is a
        # row-wise softmax constant and therefore cannot receive gradients.
        self.query = nn.Conv2d(
            self.context_channels, self.attention_dim,
            kernel_size=1, bias=False)
        self.key = nn.Conv2d(
            self.context_channels, self.attention_dim,
            kernel_size=1, bias=False)
        self.value = nn.Conv2d(
            self.context_channels, self.context_channels,
            kernel_size=1, bias=True)
        self.output = nn.Conv2d(
            self.context_channels, self.context_channels,
            kernel_size=1, bias=True)
        self.raw_alpha = nn.Parameter(torch.full(
            (1, self.context_channels, 1, 1), raw_init))

    def effective_alpha(self):
        """Return the signed per-channel scale bounded by ``alpha_max``."""
        return self.alpha_max * torch.tanh(self.raw_alpha)

    def _validate_input(self, features):
        if not torch.is_tensor(features) or features.ndim != 4:
            raise RuntimeError('PSCA input must be a BCHW tensor')
        if features.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f'PSCA input has {features.shape[1]} channels, '
                f'expected {self.hidden_dim}')
        if not features.is_floating_point() or features.is_complex():
            raise RuntimeError(
                'PSCA input must use a real floating-point dtype')
        if (features.shape[-2] < self.pool_stride
                or features.shape[-1] < self.pool_stride):
            raise RuntimeError(
                'PSCA input spatial dimensions must be at least pool_stride')

    def forward(self, features, return_aux=False):
        self._validate_input(features)
        batch, _, height, width = features.shape
        partial, context = torch.split(
            features, (self.partial_channels, self.context_channels), dim=1)

        pooled_context = self.context_pool(context)
        query = self.query(context).flatten(2).transpose(1, 2)
        key = self.key(pooled_context).flatten(2)
        value = self.value(pooled_context).flatten(2).transpose(1, 2)

        # Keep only QK/softmax in FP32.  Casting attention back before AV
        # preserves the active dtype for the expensive context aggregation.
        with torch.autocast(device_type=features.device.type, enabled=False):
            attention_logits = (
                torch.bmm(query.float(), key.float()) * self.scale)
            attention = torch.softmax(attention_logits, dim=-1)
        attended = torch.bmm(attention.to(dtype=value.dtype), value)
        attended = attended.transpose(1, 2).reshape(
            batch, self.context_channels, height, width)
        projected = self.output(attended)

        alpha = self.effective_alpha().to(dtype=projected.dtype)
        refined_context = context + alpha * projected
        output = torch.cat((partial, refined_context), dim=1)
        if self.channel_shuffle:
            output = _channel_shuffle(output, self.shuffle_groups)

        if not return_aux:
            return output
        return output, {
            'partial': partial,
            'context': context,
            'pooled_context': pooled_context,
            'query': query,
            'key': key,
            'value': value,
            'attention': attention,
            'attended': attended,
            'projected': projected,
            'refined_context': refined_context,
            'alpha': alpha,
        }


PSCANeck = PartialSpatialContextAttention
