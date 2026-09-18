"""Adapted from E:\\code\\RT-DETR\\RTDETR-main\\ultralytics\\nn\\extra_modules\\block.py.

Source project: E:\\code\\RT-DETR\\RTDETR-main
Source classes: SRFD, Cut. Source YAML: rtdetr-SRFD.yaml / rtdetr-r50-SRFD.yaml.
Adaptation: RT-DETR PResNet18 plugin only; NO Neck DRFD is imported.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.batch_norm(self.conv_fusion(torch.cat([
            x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]], dim=1)))


class SRFDPlugin(nn.Module):
    """Two-stage 4x downsampling: convolution / phase-cut / max-pool fusion."""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        if out_channels % 4:
            raise ValueError('SRFD out_channels must be divisible by four')
        quarter, half = out_channels // 4, out_channels // 2
        self.conv_init = nn.Conv2d(in_channels, quarter, 7, padding=3)
        self.conv_1 = nn.Conv2d(quarter, half, 3, padding=1, groups=quarter)
        self.conv_x1 = nn.Conv2d(half, half, 3, stride=2, padding=1, groups=half)
        self.batch_norm_x1 = nn.BatchNorm2d(half)
        self.cut_c = Cut(quarter, half)
        self.fusion1 = nn.Conv2d(out_channels, half, 1)
        self.conv_2 = nn.Conv2d(half, out_channels, 3, padding=1, groups=half)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, 3, stride=2,
                                 padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(2, 2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(half, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, x):
        # All training scales are multiples of 32. Right/bottom zero padding
        # also makes odd-size inputs agree with the stock stem's ceil(H/4).
        pad_h, pad_w = (-x.shape[-2]) % 4, (-x.shape[-1]) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.conv_init(x)
        cut = self.cut_c(x)
        x = self.batch_norm_x1(self.conv_x1(self.conv_1(x)))
        x = self.fusion1(torch.cat([x, cut], dim=1))
        cut = self.cut_r(x)
        x = self.conv_2(x)
        pooled = self.batch_norm_m(self.max_m(x))
        x = self.batch_norm_x2(self.conv_x2(x))
        return self.fusion2(torch.cat([x, cut, pooled], dim=1))
