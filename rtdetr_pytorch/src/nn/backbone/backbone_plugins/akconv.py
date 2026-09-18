"""Adapted from E:\\code\\RT-DETR\\RTDETR-main\\ultralytics\\nn\\extra_modules\\block.py.

Source project: E:\\code\\RT-DETR\\RTDETR-main
Source class: AKConv. Source YAML: rtdetr-AKConv.yaml / rtdetr-r50-AKConv.yaml.
Adaptation: RT-DETR PResNet18 plugin only. Preserve source sampling/interpolation;
retain external PResNet BN/act instead of the source terminal BN/SiLU. Native
reshape replaces einops. Coordinates use FP32 on the input device under AMP.
The source _set_lr hook only assigns local generators and returns None: it has
no gradient-scaling effect, so that ineffective hook is not copied.
"""

import math
import torch
from torch import nn


class AKConvPlugin(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_param=5):
        super().__init__()
        if not isinstance(num_param, int) or num_param < 1:
            raise ValueError('AKConv num_param must be a positive integer')
        self.num_param, self.stride = num_param, stride
        self.conv = nn.Conv2d(in_channels, out_channels, (num_param, 1),
                              stride=(num_param, 1), bias=False)
        self.p_conv = nn.Conv2d(in_channels, 2 * num_param, 3,
                                padding=1, stride=stride)
        nn.init.zeros_(self.p_conv.weight)  # source leaves offset bias at default
        width = round(math.sqrt(num_param))
        rows, remainder = divmod(num_param, width)
        point_y, point_x = torch.meshgrid(torch.arange(rows), torch.arange(width),
                                          indexing='ij')
        point_y, point_x = point_y.flatten(), point_x.flatten()
        if remainder:
            point_y = torch.cat([point_y, torch.full((remainder,), rows)])
            point_x = torch.cat([point_x, torch.arange(remainder)])
        self.register_buffer('_points', torch.cat([point_y, point_x]).float().view(
            1, 2 * num_param, 1, 1), persistent=False)

    @staticmethod
    def _gather(x, q, count):
        batch, height, width, _ = q.shape
        index = q[..., :count] * x.shape[-1] + q[..., count:]
        index = index.reshape(batch, 1, -1).expand(-1, x.shape[1], -1)
        return x.flatten(2).gather(2, index).reshape(
            batch, x.shape[1], height, width, count)

    def forward(self, x):
        offset = self.p_conv(x)
        height, width = offset.shape[-2:]
        # Do not use half-precision coordinates: AMP at larger feature sizes
        # otherwise loses subpixel offsets even though the conv itself is safe.
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=torch.float32) * self.stride,
            torch.arange(width, device=x.device, dtype=torch.float32) * self.stride,
            indexing='ij')
        base = torch.cat([yy[None].expand(self.num_param, -1, -1),
                          xx[None].expand(self.num_param, -1, -1)], dim=0)[None]
        p = (base + self._points.float() + offset.float()).permute(0, 2, 3, 1)
        count = self.num_param
        lower = p.detach().floor()
        upper = lower + 1

        def clamp(q):
            return torch.cat([q[..., :count].clamp(0, x.shape[-2] - 1),
                              q[..., count:].clamp(0, x.shape[-1] - 1)], dim=-1)

        lower, upper = clamp(lower).long(), clamp(upper).long()
        left_bottom = torch.cat([lower[..., :count], upper[..., count:]], dim=-1)
        right_top = torch.cat([upper[..., :count], lower[..., count:]], dim=-1)
        p = clamp(p)
        py, px = p[..., :count], p[..., count:]
        ly, lx = lower[..., :count], lower[..., count:]
        uy, ux = upper[..., :count], upper[..., count:]
        weights = ((1 + ly - py) * (1 + lx - px),
                   (1 - uy + py) * (1 - ux + px),
                   (1 + ly - py) * (1 - ux + px),
                   (1 - uy + py) * (1 + lx - px))
        sampled = sum(weight.to(x.dtype).unsqueeze(1) * self._gather(x, corner, count)
                      for weight, corner in zip(weights, (lower, upper, left_bottom, right_top)))
        # b c h w n -> b c (h n) w, exactly as the source column aggregation.
        sampled = sampled.permute(0, 1, 2, 4, 3).reshape(
            x.shape[0], x.shape[1], height * count, width)
        return self.conv(sampled)

    def init_from_conv(self, conv):
        return 'new source-style initialization; irregular five-point aggregation has no exact dense 3x3 mapping'
