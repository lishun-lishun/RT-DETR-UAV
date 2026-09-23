"""Base-detail phase-preserving stride-two downsampling for PResNet."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..common import ConvNormLayer, get_activation


__all__ = ['BDPDDownsample']


class BDPDDownsample(nn.Module):
    """Replace a stride-two ConvNormLayer while retaining all four 2x2 phases.

    Phase splitting produces ``[B, 4, C, H/2, W/2]`` after minimal
    right/bottom replicate padding for odd inputs. Base and detail projections
    both produce ``[B, C_out, H/2, W/2]``.
    """

    def __init__(self, in_channels, out_channels, local_kernel=3,
                 detail_floor=0.25, eps=1e-6, alpha_init=0.5,
                 alpha_max=1.0, act='relu', debug=False):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError('BDPD in_channels must be a positive integer')
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError('BDPD out_channels must be a positive integer')
        if local_kernel != 3:
            raise ValueError('BDPD v1 requires local_kernel=3')
        if not 0.0 <= float(detail_floor) <= 1.0:
            raise ValueError('BDPD detail_floor must be in [0, 1]')
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError('BDPD eps must be finite and positive')
        if not math.isfinite(float(alpha_max)) or float(alpha_max) <= 0:
            raise ValueError('BDPD alpha_max must be finite and positive')
        if not 0.0 < float(alpha_init) < float(alpha_max):
            raise ValueError('BDPD alpha_init must be in (0, alpha_max)')
        if not isinstance(debug, bool):
            raise ValueError('BDPD debug must be boolean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_kernel = local_kernel
        self.detail_floor = float(detail_floor)
        self.eps = float(eps)
        self.alpha_max = float(alpha_max)
        ratio = float(alpha_init) / float(alpha_max)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        self.base_projection = ConvNormLayer(in_channels, out_channels, 1, 1, act=None)
        self.detail_projection = ConvNormLayer(4 * in_channels, out_channels, 1, 1,
                                               act=None)
        self.activation = get_activation(act)
        self.debug = debug
        self.last_debug_stats = None
        self.last_debug_tensors = None

    @property
    def alpha(self):
        return self.alpha_max * self.raw_alpha.float().sigmoid()

    def phase_components(self, x):
        """Return phase/base/detail/consistency tensors for tests and analysis."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('BDPD expects NCHW input with configured in_channels')
        if x.shape[-2] < 1 or x.shape[-1] < 1:
            raise ValueError('BDPD requires nonempty spatial dimensions')
        pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        phases = torch.stack((x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2],
                              x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]), dim=1)
        base = phases.mean(dim=1)
        detail = phases - base.unsqueeze(1)

        # Ratio arithmetic remains FP32 under AMP. count_include_pad=False
        # implements same-size local means without injecting artificial zeros.
        shape = detail.shape
        detail32 = detail.float().reshape(shape[0] * 4, shape[2], shape[3], shape[4])
        signed_mean = F.avg_pool2d(detail32, self.local_kernel, stride=1,
                                   padding=self.local_kernel // 2,
                                   count_include_pad=False)
        absolute_mean = F.avg_pool2d(detail32.abs(), self.local_kernel, stride=1,
                                     padding=self.local_kernel // 2,
                                     count_include_pad=False)
        consistency = (signed_mean.abs() / (absolute_mean + self.eps)).clamp_(0.0, 1.0)
        consistency = consistency.reshape(shape[0], 4, shape[2], shape[3], shape[4])
        weight = self.detail_floor + (1.0 - self.detail_floor) * consistency
        weighted_detail = (weight * detail.float()).to(dtype=detail.dtype)
        return {'phases': phases, 'base': base, 'detail': detail,
                'consistency': consistency, 'detail_weight': weight,
                'weighted_detail': weighted_detail,
                'padding': (pad_h, pad_w)}

    def forward(self, x):
        fields = self.phase_components(x)
        base_output = self.base_projection(fields['base'])
        weighted = fields['weighted_detail']
        detail_input = weighted.reshape(weighted.shape[0], -1,
                                        weighted.shape[-2], weighted.shape[-1])
        detail_output = self.detail_projection(detail_input)
        alpha = self.alpha.to(dtype=detail_output.dtype)
        output = self.activation(base_output + alpha * detail_output)

        if self.debug:
            with torch.no_grad():
                consistency = fields['consistency'].detach().float().flatten()
                weight = fields['detail_weight'].detach().float().flatten()
                base_norm = torch.linalg.vector_norm(base_output.detach().float())
                raw_norm = torch.linalg.vector_norm(fields['detail'].detach().float())
                weighted_norm = torch.linalg.vector_norm(weighted.detach().float())
                detail_norm = torch.linalg.vector_norm(detail_output.detach().float())
                output_norm = torch.linalg.vector_norm(output.detach().float())
                self.last_debug_stats = {
                    'base_norm': base_norm,
                    'detail_raw_norm': raw_norm,
                    'detail_weighted_norm': weighted_norm,
                    'consistency_mean': consistency.mean(),
                    'consistency_p10': torch.quantile(consistency, 0.10),
                    'consistency_p50': torch.quantile(consistency, 0.50),
                    'consistency_p90': torch.quantile(consistency, 0.90),
                    'detail_weight_mean': weight.mean(),
                    'alpha': self.alpha.detach(),
                    'base_output_ratio': base_norm / (output_norm + self.eps),
                    'detail_output_ratio': (self.alpha.detach() * detail_norm /
                                            (output_norm + self.eps)),
                }
                self.last_debug_tensors = {
                    'base_energy': fields['base'].detach().float().square().mean(1).sqrt(),
                    'detail_energy': fields['weighted_detail'].detach().float().square()
                                     .mean((1, 2)).sqrt(),
                    'consistency': fields['consistency'].detach().float().mean((1, 2)),
                }
        return output

