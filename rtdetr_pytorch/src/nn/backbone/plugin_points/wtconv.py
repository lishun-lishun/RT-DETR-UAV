"""WTConv2d port from reference ultralytics/nn/extra_modules/wtconv2d.py.

Source YAML: rtdetr-WTConv.yaml. Only same-scale WTConv is ported; the two
Neck WTConv downsamplers and the Ultralytics framework are not copied.
Haar/db1 taps are exactly the source default wavelet. This minimal first
version needs neither PyWavelets nor the unused dill import. Native autograd
replaces the source custom Functions, preserving fixed transforms and avoiding
FP16 backward filter dtype mismatches. Non-db1 wavelets are not supported.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Scale(nn.Module):
    def __init__(self, channels, init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))

    def forward(self, x):
        return self.weight.to(dtype=x.dtype) * x


class WTConv2d(nn.Module):
    def __init__(self, channels, kernel_size=5, wt_levels=1, wt_type='db1'):
        super().__init__()
        if wt_type not in ('db1', 'haar'):
            raise ValueError('First-round P4-WTConv supports the source default db1/Haar only')
        if not isinstance(wt_levels, int) or wt_levels < 1:
            raise ValueError('wt_levels must be a positive integer')
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('WTConv kernel_size must be positive and odd')
        self.channels, self.wt_levels = channels, wt_levels
        low = torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)])
        high = torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)])
        taps = torch.stack([low[None, :] * low[:, None], low[None, :] * high[:, None],
                            high[None, :] * low[:, None], high[None, :] * high[:, None]])
        # Preserve source Parameter accounting, including non-trainable taps.
        self.wt_filter = nn.Parameter(taps[:, None].repeat(channels, 1, 1, 1), requires_grad=False)
        self.iwt_filter = nn.Parameter(self.wt_filter.detach().clone(), requires_grad=False)
        self.base_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                                  groups=channels, bias=True)
        self.base_scale = Scale(channels)
        self.wavelet_convs = nn.ModuleList(nn.Conv2d(channels * 4, channels * 4, kernel_size,
                                                    padding=kernel_size // 2,
                                                    groups=channels * 4, bias=False)
                                           for _ in range(wt_levels))
        self.wavelet_scale = nn.ModuleList(Scale(channels * 4, .1) for _ in range(wt_levels))

    def forward(self, x):
        levels = []
        current = x
        for convolution, scale in zip(self.wavelet_convs, self.wavelet_scale):
            shape = current.shape
            if shape[-2] % 2 or shape[-1] % 2:
                current = F.pad(current, (0, shape[-1] % 2, 0, shape[-2] % 2))
            b, c, h, w = current.shape
            transformed = F.conv2d(current, self.wt_filter.to(dtype=current.dtype),
                                    stride=2, groups=c).reshape(b, c, 4, h // 2, w // 2)
            current = transformed[:, :, 0]
            tagged = scale(convolution(transformed.flatten(1, 2))).reshape_as(transformed)
            levels.append((tagged[:, :, 0], tagged[:, :, 1:], shape))
        reconstructed = 0
        for low, high, shape in reversed(levels):
            merged = torch.cat([(low + reconstructed).unsqueeze(2), high], dim=2).flatten(1, 2)
            reconstructed = F.conv_transpose2d(merged, self.iwt_filter.to(dtype=merged.dtype),
                                               stride=2, groups=self.channels)
            reconstructed = reconstructed[:, :, :shape[-2], :shape[-1]]
        return self.base_scale(self.base_conv(x)) + reconstructed
