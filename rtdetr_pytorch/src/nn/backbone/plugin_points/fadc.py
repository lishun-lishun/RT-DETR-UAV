# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of the upstream source tree.
"""Default AdaptiveDilatedConv path from reference extra_modules/fadc.py.

Source YAML: rtdetr-fadc.yaml; classes FrequencySelection/AdaptiveDilatedConv.
Keep FFT band selection, learned dilation offsets and modulated sampling.
Only the source YAML's actual path is ported: pre_fs, freq bands [3,5,7,9],
sigmoid*2 high-band gates, source linear low-band gate, spatial_group1,
kernel_decompose=None, no extra OmniAttention.
The SAME modulated deformable operation uses torchvision's native backend
instead of MMCV. This is NOT a substitute for the separate DCNv4 plugin.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrequencySelection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.k_list = (3, 5, 7, 9)
        self.freq_weight_conv_list = nn.ModuleList(
            nn.Conv2d(channels, 1, 3, padding=1, bias=True) for _ in range(5))
        for conv in self.freq_weight_conv_list:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        # cuFFT half cannot handle S4's usual 40x40 shape. Only FFT/band math
        # is FP32; learned convolutions retain the original autocast mechanism.
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.float() if dtype in (torch.float16, torch.bfloat16) else x
            spectrum = torch.fft.fftshift(torch.fft.fft2(work, norm='ortho'), dim=(-2, -1))
            h, w = x.shape[-2:]
            previous = work
            bands = []
            for frequency in self.k_list:
                mask = work.new_zeros(1, 1, h, w)
                mask[:, :, round(h / 2 - h / (2 * frequency)):round(h / 2 + h / (2 * frequency)),
                     round(w / 2 - w / (2 * frequency)):round(w / 2 + w / (2 * frequency))] = 1
                low = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)),
                                      norm='ortho').real
                bands.append((previous - low).to(dtype=dtype))
                previous = low
            bands.append(previous.to(dtype=dtype))
        high = sum((conv(x).sigmoid() * 2) * band
                   for conv, band in zip(self.freq_weight_conv_list[:-1], bands[:-1]))
        # The reference freq path notably does NOT call sp_act on its last,
        # low-frequency gate. Preserve that actual implementation, not the more
        # intuitive sigmoid gate used in some other FADC variants.
        return high + self.freq_weight_conv_list[-1](x) * bands[-1]


class FADC(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('FADC epsilon must be positive to avoid a dead initial ReLU offset')
        self.channels, self.epsilon = channels, epsilon
        self.weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
        # MMCV ModulatedDeformConv2d default bias=False.
        self.register_parameter('bias', None)
        self.conv_offset = nn.Conv2d(channels, 1, 3, padding=1, bias=True)
        self.conv_mask = nn.Conv2d(channels, 9, 3, padding=1, bias=True)
        self.FS = FrequencySelection(channels)
        self.register_buffer('dilated_offset', torch.tensor(
            [-1., -1., -1., 0., -1., 1., 0., -1., 0., 0., 0., 1., 1., -1., 1., 0., 1., 1.]
        ).reshape(1, 18, 1, 1))
        # Matches MMCV's default uniform convolution initialization.
        bound = 1 / math.sqrt(channels * 9)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.constant_(self.conv_offset.bias, epsilon)
        nn.init.zeros_(self.conv_mask.weight)
        nn.init.zeros_(self.conv_mask.bias)

    def forward(self, x):
        try:
            from torchvision.ops import deform_conv2d
        except (ImportError, OSError) as error:
            raise RuntimeError('P4-FADC requires torchvision with native deform_conv2d') from error
        selected = self.FS(x)
        offset = F.relu(self.conv_offset(selected)) * self.dilated_offset.to(dtype=selected.dtype)
        mask = self.conv_mask(selected).sigmoid()
        # Native torchvision AMP support varies by release. Only sampling is
        # promoted, not the original backbone nor the selection/offset convs.
        dtype = selected.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = deform_conv2d(selected.float(), offset.float(), self.weight.float(),
                                   bias=None, stride=(1, 1), padding=(1, 1),
                                   dilation=(1, 1), mask=mask.float())
        return output.to(dtype=dtype)
