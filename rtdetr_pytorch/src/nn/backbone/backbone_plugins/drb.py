# UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
# Github source: https://github.com/AILab-CVC/UniRepLKNet
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on RepLKNet, ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/DingXiaoH/RepLKNet-pytorch
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
"""Adapted from E:\\code\\RT-DETR\\RTDETR-main\\ultralytics\\nn\\extra_modules\\block.py.

Source project: E:\\code\\RT-DETR\\RTDETR-main
Source class: DilatedReparamBlock. Helpers: nn/backbone/UniRepLKNet.py.
Source YAML: rtdetr-DRB.yaml / rtdetr-r50-DRB.yaml.
Adaptation: RT-DETR PResNet18 plugin only. Native PyTorch depthwise conv and
small BN/kernel-fusion helpers; no timm, iGEMM, SE, GRN or complete backbone.
"""

import torch
from torch import nn
from torch.nn import functional as F


def fuse_bn(conv, bn):
    std = (bn.running_var + bn.eps).sqrt()
    bias = 0 if conv.bias is None else conv.bias
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
            bn.bias + (bias - bn.running_mean) * bn.weight / std)


def merge_dilated_kernel(large, small, dilation):
    identity = small.new_ones(1, 1, 1, 1)
    equivalent = F.conv_transpose2d(small, identity, stride=dilation)
    pad = (large.shape[-1] - equivalent.shape[-1]) // 2
    return large + F.pad(equivalent, [pad] * 4)


class DRBPlugin(nn.Module):
    """Depthwise large kernel plus source-prescribed dilated Conv+BN branches."""

    SETTINGS = {
        5: ([3, 3], [1, 2]), 7: ([5, 3, 3], [1, 2, 3]),
        9: ([5, 5, 3, 3], [1, 2, 3, 4]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
        15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
        17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
    }

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            raise ValueError('The source DRB requires equal channels and stride=1')
        if kernel_size not in self.SETTINGS:
            raise ValueError('DRB kernel_size must be one of 5,7,9,11,13,15,17')
        self.kernel_sizes, self.dilates = self.SETTINGS[kernel_size]
        self.lk_origin = nn.Conv2d(in_channels, in_channels, kernel_size,
                                  padding=kernel_size // 2, groups=in_channels,
                                  bias=False)
        self.origin_bn = nn.BatchNorm2d(in_channels)
        for size, rate in zip(self.kernel_sizes, self.dilates):
            setattr(self, f'dil_conv_k{size}_{rate}', nn.Conv2d(
                in_channels, in_channels, size, padding=rate * (size - 1) // 2,
                dilation=rate, groups=in_channels, bias=False))
            setattr(self, f'dil_bn_k{size}_{rate}', nn.BatchNorm2d(in_channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for size, rate in zip(self.kernel_sizes, self.dilates):
            out = out + getattr(self, f'dil_bn_k{size}_{rate}')(
                getattr(self, f'dil_conv_k{size}_{rate}')(x))
        return out

    def init_from_conv(self, conv):
        return 'new source-style initialization; dense 3x3 cannot map exactly to depthwise DRB'

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            weight, bias = fuse_bn(self.lk_origin, self.origin_bn)
            for size, rate in zip(self.kernel_sizes, self.dilates):
                branch_weight, branch_bias = fuse_bn(
                    getattr(self, f'dil_conv_k{size}_{rate}'),
                    getattr(self, f'dil_bn_k{size}_{rate}'))
                weight = merge_dilated_kernel(weight, branch_weight, rate)
                bias = bias + branch_bias
            conv = nn.Conv2d(weight.shape[0], weight.shape[0], weight.shape[-1],
                             padding=weight.shape[-1] // 2, groups=weight.shape[0],
                             bias=True).to(device=weight.device, dtype=weight.dtype)
            conv.weight.copy_(weight)
            conv.bias.copy_(bias)
            self.lk_origin = conv
            del self.origin_bn
            for size, rate in zip(self.kernel_sizes, self.dilates):
                delattr(self, f'dil_conv_k{size}_{rate}')
                delattr(self, f'dil_bn_k{size}_{rate}')

    convert_to_deploy = switch_to_deploy
