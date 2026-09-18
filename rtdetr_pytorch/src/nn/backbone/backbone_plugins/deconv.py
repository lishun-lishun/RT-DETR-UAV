"""Adapted from E:\\code\\RT-DETR\\RTDETR-main\\ultralytics\\nn\\extra_modules\\deconv.py.

Source project: E:\\code\\RT-DETR\\RTDETR-main
Source classes: Conv2d_cd/hd/vd/ad, DEConv.
Source YAML: rtdetr-DEConv.yaml / rtdetr-r50-DEConv.yaml.
Adaptation: RT-DETR PResNet18 plugin only. Keep the original external BN/act;
use native reshape and device/dtype-safe allocations instead of einops/CUDA
FloatTensor. The unused radial-difference class is deliberately not ported.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DifferenceConv(nn.Module):
    def __init__(self, in_channels, out_channels, kind):
        super().__init__()
        self.kind = kind
        if kind in ('hd', 'vd'):
            self.conv = nn.Conv1d(in_channels, out_channels, 3, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=True)

    def get_weight(self):
        weight = self.conv.weight.flatten(2)
        if self.kind == 'cd':
            result = weight.clone()
            result[:, :, 4] = weight[:, :, 4] - weight.sum(dim=2)
        elif self.kind == 'ad':
            result = weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        else:
            result = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
            positive, negative = (([0, 3, 6], [2, 5, 8]) if self.kind == 'hd'
                                  else ([0, 1, 2], [6, 7, 8]))
            result[:, :, positive] = weight
            result[:, :, negative] = -weight
        return result.reshape(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class DEConvPlugin(nn.Module):
    """Sum four difference kernels and one ordinary 3x3 kernel before conv."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1_1 = DifferenceConv(in_channels, out_channels, 'cd')
        self.conv1_2 = DifferenceConv(in_channels, out_channels, 'hd')
        self.conv1_3 = DifferenceConv(in_channels, out_channels, 'vd')
        self.conv1_4 = DifferenceConv(in_channels, out_channels, 'ad')
        self.conv1_5 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                                padding=1, bias=True)

    def equivalent_kernel(self):
        weight, bias = self.conv1_5.weight, self.conv1_5.bias
        if hasattr(self, 'conv1_1'):
            for index in range(1, 5):
                branch_weight, branch_bias = getattr(self, f'conv1_{index}').get_weight()
                weight, bias = weight + branch_weight, bias + branch_bias
        return weight, bias

    def forward(self, x):
        weight, bias = self.equivalent_kernel()
        return F.conv2d(x, weight, bias, stride=self.stride, padding=1)

    @torch.no_grad()
    def init_from_conv(self, conv):
        self.conv1_5.weight.copy_(conv.weight)
        self.conv1_5.bias.zero_()
        if conv.bias is not None:
            self.conv1_5.bias.copy_(conv.bias)
        # Preserve the loaded operator initially, without disabling gradients
        # in the four difference branches. No extra residual or loss is added.
        for index in range(1, 5):
            branch = getattr(self, f'conv1_{index}').conv
            branch.weight.zero_()
            branch.bias.zero_()
        return 'ordinary kernel copied; difference kernels/biases initialized to zero'

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'conv1_1'):
            weight, bias = self.equivalent_kernel()
            self.conv1_5.weight.copy_(weight)
            self.conv1_5.bias.copy_(bias)
            for index in range(1, 5):
                delattr(self, f'conv1_{index}')

    convert_to_deploy = switch_to_deploy
