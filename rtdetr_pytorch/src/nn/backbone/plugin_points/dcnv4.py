# --------------------------------------------------------
# Deformable Convolution v4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see upstream LICENSE for details]
# --------------------------------------------------------
"""Minimal DCNv4 port from the bundled third_party/DCNv4_op/DCNv4.

Sources: modules/dcnv4.py (DCNv4) and functions/dcnv4_func.py.
Source YAML: rtdetr-DCNV4.yaml. No YOLO or FlashDeformAttn module is ported.
The real, separately compiled DCNv4.ext remains REQUIRED: there is no fallback.
Only the needed configuration is supported: 3x3, stride1, pad1, no center
removal/scale, pointwise value/output projections, native NCHW adapter.
"""

import importlib
import math

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def load_backend():
    try:
        ext = importlib.import_module('DCNv4.ext')
    except (ImportError, OSError) as error:
        raise RuntimeError('P2 requires the REAL DCNv4 CUDA extension (DCNv4.ext). '
                           'Build the bundled third_party/DCNv4_op in the server '
                           'training environment; see third_party/DCNv4_op/'
                           'README_RTDETR.md. No DCNv2/3/AKConv fallback is used.') from error
    if not all(hasattr(ext, name) for name in ('dcnv4_forward', 'dcnv4_backward')):
        raise RuntimeError('Incompatible DCNv4.ext: missing forward/backward symbols')
    return ext


def launch_spec(batch, height, width, groups, channels, backward=False):
    # Same shape-derived fallback as reference findspec/find_spec_bwd; the
    # optional giant performance lookup table is not a mathematical dependency.
    d_stride = (2 if channels >= 64 else 1) if backward else 8
    limit = 256 if backward else 512
    multiplier = max(m for m in range(1, 65)
                     if batch * height * width % m == 0
                     and m * groups * channels // d_stride <= limit)
    return d_stride, multiplier * groups * channels // d_stride


class DCNv4Function(Function):
    @staticmethod
    def forward(ctx, value, offset_mask, groups, offset_scale):
        ext = load_backend()
        n, h, w, c = value.shape
        group_channels = c // groups
        fw_stride, fw_threads = launch_spec(n, h, w, groups, group_channels)
        ctx.bw_spec = launch_spec(n, h, w, groups, group_channels, backward=True)
        ctx.groups, ctx.group_channels, ctx.offset_scale = groups, group_channels, offset_scale
        ctx.save_for_backward(value, offset_mask)
        return ext.dcnv4_forward(value, offset_mask, 3, 3, 1, 1, 1, 1, 1, 1,
                                 groups, group_channels, offset_scale, 256, 0,
                                 fw_stride, fw_threads, False)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, offset_mask = ctx.saved_tensors
        grad_value, grad_offset_mask = load_backend().dcnv4_backward(
            value, offset_mask, 3, 3, 1, 1, 1, 1, 1, 1,
            ctx.groups, ctx.group_channels, ctx.offset_scale, 256,
            grad_output.contiguous(), 0, *ctx.bw_spec, False)
        return grad_value, grad_offset_mask, None, None


class DCNv4Adapter(nn.Module):
    def __init__(self, channels, groups=1, offset_scale=1.0, mask_init=1.0 / 9.0):
        super().__init__()
        if (not isinstance(groups, int) or groups <= 0 or channels % groups
                or (channels // groups) % 16):
            raise ValueError('DCNv4 requires positive groups and channels/groups divisible by 16')
        if not math.isfinite(offset_scale) or offset_scale <= 0:
            raise ValueError('DCNv4 offset_scale must be finite and positive')
        if not math.isfinite(mask_init) or mask_init == 0:
            raise ValueError('DCNv4 mask_init must be finite and nonzero for zero-gate learning')
        self.channels, self.groups, self.offset_scale = channels, groups, offset_scale
        self.offset_mask = nn.Linear(channels, math.ceil(groups * 27 / 8) * 8)
        self.value_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm2d(channels)  # source DCNV4_YOLO(..., act=None)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        # Source CUDA layout per group: 18 offsets followed by 9 raw weights.
        # Source init=0 makes DCNv4(x)=0 and kills a zero external gate forever.
        # Uniform initial raw weights avoid that; NO softmax/attention is added.
        with torch.no_grad():
            for group in range(groups):
                self.offset_mask.bias[group * 27 + 18:group * 27 + 27].fill_(mask_init)
        for projection in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError('P2-DCNv4 is a native CUDA-only plugin; no CPU fallback')
        load_backend()
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        value = self.value_proj(tokens).reshape(n, h, w, c).contiguous()
        offset_mask = self.offset_mask(tokens).reshape(n, h, w, -1).contiguous()
        # Explicitly match native inputs under the surrounding train AMP context.
        offset_mask = offset_mask.to(dtype=value.dtype)
        if value.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(f'DCNv4 supports this port in FP16/FP32, got {value.dtype}')
        output = DCNv4Function.apply(value, offset_mask, self.groups, self.offset_scale)
        output = self.output_proj(output.reshape(n, h * w, c))
        return self.bn(output.transpose(1, 2).reshape(n, c, h, w))
