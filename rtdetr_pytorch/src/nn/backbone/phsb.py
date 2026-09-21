"""Persistent high-resolution semantic branch for PResNet18-d.

The branch consumes the original stride-8 S3 output.  It never replaces an
original stage and returns two independently scaled residuals for P3 and P4.
"""

import math

import torch
from torch import nn

from .common import ConvNormLayer


class PHSBBranch(nn.Module):
    def __init__(self, ch_in, ch_p4, block_type, branch_ratio=0.75,
                 num_blocks=3, alpha_init=0.0, alpha_max=0.20,
                 debug=False, debug_interval=100):
        super().__init__()
        if not isinstance(ch_in, int) or ch_in <= 0 or not isinstance(ch_p4, int) or ch_p4 <= 0:
            raise ValueError('PHSB channels must be positive integers')
        if not math.isfinite(branch_ratio) or not 0 < branch_ratio <= 1:
            raise ValueError('PHSB branch_ratio must be in (0, 1]')
        if not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError('PHSB num_blocks must be a positive integer')
        if not all(math.isfinite(v) for v in (alpha_init, alpha_max)) or alpha_max <= 0:
            raise ValueError('PHSB alpha_init must be finite and alpha_max positive')
        if not isinstance(debug, bool) or not isinstance(debug_interval, int) or debug_interval < 1:
            raise ValueError('PHSB debug must be boolean and debug_interval positive')

        # Eight-channel alignment keeps the branch hardware-friendly without
        # hard-coding 96.  R18 C3=128 and ratio=.75 resolve to Ch=96.
        self.hidden_channels = max(8, int(round(ch_in * branch_ratio / 8)) * 8)
        self.reduction = ConvNormLayer(ch_in, self.hidden_channels, 1, 1, act='relu')
        self.hr_blocks = nn.Sequential(*[
            block_type(self.hidden_channels, self.hidden_channels, stride=1,
                       shortcut=True, act='relu', variant='d')
            for _ in range(num_blocks)
        ])
        self.p3_projection = ConvNormLayer(self.hidden_channels, ch_in, 1, 1, act=None)
        self.p4_projection = ConvNormLayer(self.hidden_channels, ch_p4, 3, 2, act=None)
        self.raw_alpha3 = nn.Parameter(torch.tensor(float(alpha_init)))
        self.raw_alpha4 = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = float(alpha_max)
        self.debug = debug
        self.debug_interval = debug_interval
        self._debug_iteration = 0

    @property
    def alpha3_effective(self):
        return self.alpha_max * self.raw_alpha3.tanh()

    @property
    def alpha4_effective(self):
        return self.alpha_max * self.raw_alpha4.tanh()

    def _print_debug(self, hidden, e3, e4, f3_base, f4_base):
        eps = 1e-12
        e3_norm, e4_norm = e3.float().norm(), e4.float().norm()
        p3_norm, p4_norm = f3_base.float().norm(), f4_base.float().norm()
        values = {
            'alpha3_effective': self.alpha3_effective,
            'alpha4_effective': self.alpha4_effective,
            'E3_norm': e3_norm, 'P3_base_norm': p3_norm,
            'E3/P3': e3_norm / p3_norm.clamp_min(eps),
            'E4_norm': e4_norm, 'P4_base_norm': p4_norm,
            'E4/P4': e4_norm / p4_norm.clamp_min(eps),
            'HR_mean': hidden.float().mean(),
            'HR_std': hidden.float().std(unbiased=False),
        }
        message = ' '.join(f'{name}={float(value.detach()):.6g}'
                           for name, value in values.items())
        print(f'[PHSB iter={self._debug_iteration}] {message}')

    def forward(self, f3_base, f4_base):
        hidden = self.hr_blocks(self.reduction(f3_base))
        e3 = self.p3_projection(hidden)
        e4 = self.p4_projection(hidden)
        if e3.shape != f3_base.shape or e4.shape != f4_base.shape:
            raise ValueError('PHSB projection must match original P3/P4 shapes')
        if self.debug and self._debug_iteration % self.debug_interval == 0:
            self._print_debug(hidden, e3, e4, f3_base, f4_base)
        self._debug_iteration += 1
        return (self.alpha3_effective.to(dtype=e3.dtype) * e3,
                self.alpha4_effective.to(dtype=e4.dtype) * e4)
