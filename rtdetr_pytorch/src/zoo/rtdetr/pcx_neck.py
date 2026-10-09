"""Partial-channel cross-scale exchange for post-CCFF feature pyramids.

PCX keeps the first ``Cp`` channels of every pyramid level on a parameter-free
route and exchanges only the final ``Ce`` channels between adjacent levels.
It accepts the three same-width CCFF outputs ``[N3, N4, N5]`` and returns a
three-element list with their spatial sizes unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['PartialChannelCrossScaleExchange', 'PCXNeck']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _exchange_width(hidden_dim, exchange_ratio):
    if (not isinstance(exchange_ratio, (int, float))
            or isinstance(exchange_ratio, bool)
            or not math.isfinite(float(exchange_ratio))):
        raise ValueError('PCX.exchange_ratio must be a finite number')
    exchange_ratio = float(exchange_ratio)
    if not 0.0 < exchange_ratio < 1.0:
        raise ValueError('PCX.exchange_ratio must be between zero and one')

    exchange_channels = int(hidden_dim * exchange_ratio)
    preserved_channels = hidden_dim - exchange_channels
    if exchange_channels < 1 or preserved_channels < 1:
        raise ValueError(
            'PCX.exchange_ratio must leave at least one channel in each part')
    return exchange_ratio, preserved_channels, exchange_channels


class _ExchangeRefinement(nn.Module):
    """Independent spatial and pointwise refinement for one exchange part."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=True)
        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, bias=True)

    def forward(self, features):
        return self.pointwise(self.depthwise(features))


class PartialChannelCrossScaleExchange(nn.Module):
    """Exchange a configurable tail of the N3/N4/N5 channel dimension.

    For each input ``Ni``, its contiguous channel split is ``[Pi, Ei]``.
    ``Pi`` is concatenated directly into the corresponding output.  The three
    exchange sums are::

        S3 = E3 + resize(E4, size(E3))
        S4 = E4 + resize(conv3s2(E3), size(E4)) + resize(E5, size(E4))
        S5 = E5 + resize(conv3s2(E4), size(E5))

    Separate depthwise 3x3 and pointwise 1x1 pairs transform S3/S4/S5 before
    concatenation.  An optional parameter-free two-group channel permutation
    is applied last.
    """

    def __init__(
            self,
            hidden_dim=256,
            exchange_ratio=0.25,
            channel_shuffle=True):
        super().__init__()
        self.hidden_dim = _positive_int('PCX.hidden_dim', hidden_dim)
        (self.exchange_ratio,
         self.preserved_channels,
         self.exchange_channels) = _exchange_width(
             self.hidden_dim, exchange_ratio)
        if not isinstance(channel_shuffle, bool):
            raise ValueError('PCX.channel_shuffle must be a boolean')
        if channel_shuffle and self.hidden_dim % 2 != 0:
            raise ValueError(
                'PCX.hidden_dim must be divisible by two when '
                'channel_shuffle is enabled')
        self.channel_shuffle = channel_shuffle

        exchanged = self.exchange_channels
        self.down_3_to_4 = nn.Conv2d(
            exchanged, exchanged, kernel_size=3, stride=2, padding=1,
            bias=True)
        self.down_4_to_5 = nn.Conv2d(
            exchanged, exchanged, kernel_size=3, stride=2, padding=1,
            bias=True)

        self.refine3 = _ExchangeRefinement(exchanged)
        self.refine4 = _ExchangeRefinement(exchanged)
        self.refine5 = _ExchangeRefinement(exchanged)

    @staticmethod
    def _resize(features, spatial_size):
        if tuple(features.shape[-2:]) == tuple(spatial_size):
            return features
        return F.interpolate(features, size=spatial_size, mode='nearest')

    @staticmethod
    def _shuffle_two_groups(features):
        batch, channels, height, width = features.shape
        features = features.reshape(batch, 2, channels // 2, height, width)
        features = features.transpose(1, 2).contiguous()
        return features.reshape(batch, channels, height, width)

    def _validate_inputs(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise RuntimeError(
                'PCX input must be a three-element list or tuple [N3, N4, N5]')

        first = features[0]
        if not torch.is_tensor(first) or first.ndim != 4:
            raise RuntimeError('PCX N3 must be a BCHW tensor')
        batch = first.shape[0]
        dtype = first.dtype
        device = first.device

        for index, feature in enumerate(features, start=3):
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise RuntimeError(f'PCX N{index} must be a BCHW tensor')
            if feature.shape[0] != batch:
                raise RuntimeError('PCX inputs must use the same batch size')
            if feature.shape[1] != self.hidden_dim:
                raise RuntimeError(
                    f'PCX N{index} has {feature.shape[1]} channels, '
                    f'expected {self.hidden_dim}')
            if not feature.is_floating_point() or feature.is_complex():
                raise RuntimeError(
                    f'PCX N{index} must use a real floating-point dtype')
            if feature.dtype != dtype or feature.device != device:
                raise RuntimeError(
                    'PCX inputs must use the same dtype and device')

        for high, low in zip(features, features[1:]):
            if (high.shape[-2] < low.shape[-2]
                    or high.shape[-1] < low.shape[-1]):
                raise RuntimeError(
                    'PCX spatial sizes must be non-increasing from N3 to N5')

    def _split(self, features):
        return torch.split(
            features,
            (self.preserved_channels, self.exchange_channels),
            dim=1)

    def forward(self, features, return_aux=False):
        self._validate_inputs(features)
        n3, n4, n5 = features
        p3, e3 = self._split(n3)
        p4, e4 = self._split(n4)
        p5, e5 = self._split(n5)

        sum3 = e3 + self._resize(e4, e3.shape[-2:])
        sum4 = (
            e4
            + self._resize(self.down_3_to_4(e3), e4.shape[-2:])
            + self._resize(e5, e4.shape[-2:]))
        sum5 = (
            e5
            + self._resize(self.down_4_to_5(e4), e5.shape[-2:]))

        exchanged3 = self.refine3(sum3)
        exchanged4 = self.refine4(sum4)
        exchanged5 = self.refine5(sum5)
        outputs = [
            torch.cat((p3, exchanged3), dim=1),
            torch.cat((p4, exchanged4), dim=1),
            torch.cat((p5, exchanged5), dim=1),
        ]
        if self.channel_shuffle:
            outputs = [self._shuffle_two_groups(output) for output in outputs]

        if not return_aux:
            return outputs
        return outputs, {
            'preserved': (p3, p4, p5),
            'exchange': (e3, e4, e5),
            'exchange_sums': (sum3, sum4, sum5),
            'refined_exchange': (exchanged3, exchanged4, exchanged5),
        }


PCXNeck = PartialChannelCrossScaleExchange
