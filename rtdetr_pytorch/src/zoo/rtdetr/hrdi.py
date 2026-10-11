"""High-resolution detail injection for completed RT-DETR CCFF outputs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['HighResolutionDetailInjection', 'HRDI']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class _DetailProjection(nn.Module):
    """One independent detail projection for one detector pyramid level."""

    def __init__(self, source_channels, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            source_channels, source_channels, kernel_size,
            stride=1, padding=padding, groups=source_channels, bias=False)
        self.norm = nn.BatchNorm2d(source_channels)
        self.activation = nn.SiLU(inplace=True)
        self.projection = nn.Conv2d(
            source_channels, hidden_dim, kernel_size=1, bias=True)

    def forward(self, detail):
        detail = self.depthwise(detail)
        detail = self.norm(detail)
        detail = self.activation(detail)
        return self.projection(detail)


class HighResolutionDetailInjection(nn.Module):
    """Inject a stride-4 backbone detail source into post-CCFF P3 and P4.

    Each target owns an independent ``DWConv3 -> BN -> SiLU -> Conv1`` route
    and a scalar residual coefficient initialized strictly to zero.  P5 is
    returned as the original tensor.
    """

    _LEVEL_TO_INDEX = {'P3': 0, 'P4': 1}

    def __init__(
            self,
            hidden_dim=256,
            source='c2',
            source_channels=64,
            source_stride=4,
            targets=('P3', 'P4'),
            use_dwconv=True,
            kernel_size=3,
            norm='BN',
            activation='SiLU',
            resize_mode='bilinear',
            align_corners=False,
            init_alpha=0.0,
            per_level_alpha=True):
        super().__init__()
        self.hidden_dim = _positive_int('HRDI.hidden_dim', hidden_dim)
        if source not in ('c2', 'highest_resolution'):
            raise ValueError(
                "HRDI.source must be 'c2' or 'highest_resolution'")
        self.source = source
        self.source_channels = _positive_int(
            'HRDI.source_channels', source_channels)
        self.source_stride = _positive_int(
            'HRDI.source_stride', source_stride)
        if self.source_stride != 4:
            raise ValueError('HRDI.source_stride must be 4 in the first version')

        if not isinstance(targets, (list, tuple)):
            raise ValueError('HRDI.targets must be a list or tuple')
        self.targets = tuple(targets)
        if self.targets != ('P3', 'P4'):
            raise ValueError(
                "HRDI.targets must be exactly ['P3', 'P4'] in the first version")

        if use_dwconv is not True:
            raise ValueError('HRDI.use_dwconv must be true in the first version')
        self.use_dwconv = True
        self.kernel_size = _positive_int('HRDI.kernel_size', kernel_size)
        if self.kernel_size != 3:
            raise ValueError('HRDI.kernel_size must be 3 in the first version')
        if not isinstance(norm, str) or norm.upper() != 'BN':
            raise ValueError("HRDI.norm must be 'BN' in the first version")
        if not isinstance(activation, str) or activation.lower() != 'silu':
            raise ValueError(
                "HRDI.activation must be 'SiLU' in the first version")
        self.norm_name = 'BN'
        self.activation_name = 'SiLU'

        if resize_mode not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError(
                'HRDI.resize_mode must be nearest, bilinear or bicubic')
        if not isinstance(align_corners, bool):
            raise ValueError('HRDI.align_corners must be a boolean')
        self.resize_mode = resize_mode
        self.align_corners = align_corners

        if (not isinstance(init_alpha, (int, float))
                or isinstance(init_alpha, bool)
                or not math.isfinite(float(init_alpha))):
            raise ValueError('HRDI.init_alpha must be a finite number')
        if float(init_alpha) != 0.0:
            raise ValueError(
                'HRDI.init_alpha must be exactly 0 for strict identity initialization')
        if per_level_alpha is not True:
            raise ValueError(
                'HRDI.per_level_alpha must be true in the first version')
        self.per_level_alpha = True

        self.projections = nn.ModuleDict({
            level: _DetailProjection(
                self.source_channels, self.hidden_dim, self.kernel_size)
            for level in self.targets
        })
        self.alphas = nn.ParameterDict({
            level: nn.Parameter(torch.zeros(1)) for level in self.targets
        })

    def _resize(self, detail, spatial_size):
        if tuple(detail.shape[-2:]) == tuple(spatial_size):
            return detail
        kwargs = {}
        if self.resize_mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = self.align_corners
        return F.interpolate(
            detail, size=spatial_size, mode=self.resize_mode, **kwargs)

    def _validate_inputs(self, features, detail):
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise RuntimeError(
                'HRDI features must be a three-element list or tuple [P3, P4, P5]')
        if not torch.is_tensor(detail) or detail.ndim != 4:
            raise RuntimeError('HRDI detail source must be a BCHW tensor')
        if detail.shape[1] != self.source_channels:
            raise RuntimeError(
                f'HRDI detail source has {detail.shape[1]} channels, '
                f'expected {self.source_channels}')
        if not detail.is_floating_point() or detail.is_complex():
            raise RuntimeError('HRDI detail source must use a real floating-point dtype')

        batch = detail.shape[0]
        for index, feature in enumerate(features, start=3):
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise RuntimeError(f'HRDI P{index} must be a BCHW tensor')
            if feature.shape[0] != batch:
                raise RuntimeError('HRDI inputs must use the same batch size')
            if feature.shape[1] != self.hidden_dim:
                raise RuntimeError(
                    f'HRDI P{index} has {feature.shape[1]} channels, '
                    f'expected {self.hidden_dim}')
            if not feature.is_floating_point() or feature.is_complex():
                raise RuntimeError(
                    f'HRDI P{index} must use a real floating-point dtype')
            if feature.device != detail.device or feature.dtype != detail.dtype:
                raise RuntimeError(
                    'HRDI detail and pyramid features must use the same dtype and device')

        for level in self.targets:
            target = features[self._LEVEL_TO_INDEX[level]]
            if (detail.shape[-2] < target.shape[-2]
                    or detail.shape[-1] < target.shape[-1]):
                raise RuntimeError(
                    f'HRDI stride-{self.source_stride} detail source must not '
                    f'be smaller than target {level}')

    def forward(self, features, detail, return_aux=False):
        self._validate_inputs(features, detail)
        outputs = list(features)
        projected_details = {}
        resized_residuals = {}
        effective_alphas = {}

        for level in self.targets:
            index = self._LEVEL_TO_INDEX[level]
            target = features[index]
            # Preserve the specified HRDI order: first extract/project the
            # native stride-4 detail, then resize that projected residual to
            # the target pyramid level.
            projected = self.projections[level](detail)
            residual = self._resize(projected, target.shape[-2:])
            alpha = self.alphas[level].to(dtype=residual.dtype)
            outputs[index] = target + alpha * residual
            projected_details[level] = projected
            resized_residuals[level] = residual
            effective_alphas[level] = alpha

        if not return_aux:
            return outputs
        return outputs, {
            'projected_detail': projected_details,
            'resized_residual': resized_residuals,
            'alpha': effective_alphas,
        }


HRDI = HighResolutionDetailInjection
