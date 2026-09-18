"""Adapted from E:\\code\\RT-DETR\\RTDETR-main\\ultralytics\\nn\\extra_modules\\RFAConv.py.

Source project: E:\\code\\RT-DETR\\RTDETR-main
Source class: RFAConv. Source YAML: rtdetr-RFAConv.yaml / rtdetr-r50-RFAConv.yaml.
Adaptation: RT-DETR PResNet18 plugin only. Native reshape replaces einops;
external PResNet BN/activation replace the source terminal Conv wrapper.
Intrinsic feature-generation BN/ReLU are retained. NO RFCA/RFCBAM/SE classes.
"""

import torch
from torch import nn


class RFAConvPlugin(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        if kernel_size != 3:
            raise ValueError('This first-round RFAConv uses the source 3x3 setting')
        self.kernel_size = kernel_size
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channels, in_channels * kernel_size ** 2, 1,
                      groups=in_channels, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * kernel_size ** 2, kernel_size,
                      padding=kernel_size // 2, stride=stride,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * kernel_size ** 2), nn.ReLU())
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=kernel_size, padding=0, bias=False)

    def forward(self, x):
        batch, channels = x.shape[:2]
        weight = self.get_weight(x)
        height, width = weight.shape[-2:]
        weight = weight.reshape(batch, channels, self.kernel_size ** 2,
                                height, width).softmax(dim=2)
        feature = self.generate_feature(x).reshape_as(weight)
        feature = (feature * weight).reshape(batch, channels, self.kernel_size,
                                             self.kernel_size, height, width)
        feature = feature.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, channels, height * self.kernel_size, width * self.kernel_size)
        return self.conv(feature)

    @torch.no_grad()
    def init_from_conv(self, conv):
        self.conv.weight.copy_(conv.weight)
        return 'output 3x3 kernel copied; attention/feature generation newly initialized (not function-equivalent)'
