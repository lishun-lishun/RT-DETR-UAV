"""Reparameterized Directional Context Fusion blocks for RT-DETR.

RDCF is deliberately applied *after* HybridEncoder has produced N3/N4/N5.
It refines N3 and N4 independently while returning N5 unchanged.  During
training, each refinement block sums three depthwise branches (3x3, 1x9 and
9x1).  :meth:`ReparamDirectionalContextBlock.switch_to_deploy` embeds and
sums those kernels into one depthwise 9x9 convolution with a bias.

No normalization, attention, spatial gate, or backbone-specific operation is
used here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'ReparamDirectionalContextBlock',
    'RDCFNeck',
]


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _bounded_raw_init(maximum, initial):
    maximum = _finite_float('RDCF.eta_max', maximum)
    initial = _finite_float('RDCF.eta_init', initial)
    if maximum <= 0.0:
        raise ValueError('RDCF.eta_max must be positive')
    if abs(initial) >= maximum:
        raise ValueError('abs(RDCF.eta_init) must be smaller than eta_max')
    return maximum, math.atanh(initial / maximum)


def _validate_feature(name, value, channels):
    if not torch.is_tensor(value) or value.ndim != 4:
        raise RuntimeError(f'{name} must be a BCHW tensor')
    if value.shape[1] != channels:
        raise RuntimeError(
            f'{name} has {value.shape[1]} channels, expected {channels}')
    if not value.is_floating_point() or value.is_complex():
        raise RuntimeError(f'{name} must use a real floating-point dtype')


class ReparamDirectionalContextBlock(nn.Module):
    """Directional context residual with an exactly fusible DWConv core.

    Training structure::

        B = DWConv3x3(x) + DWConv1x9(x) + DWConv9x1(x)
        residual = Conv1x1(B)
        y = x + eta * residual

    where ``eta = eta_max * tanh(raw_eta)`` is independently learned for
    every channel.  In deploy form the three depthwise branches are replaced
    by one depthwise 9x9 convolution.  ``deploy=True`` constructs that compact
    form directly, so a converted deploy state dict can be loaded strictly.
    """

    def __init__(self, channels=256, eta_max=0.30, eta_init=0.05,
                 deploy=False):
        super().__init__()
        self.channels = _positive_int('RDCF.channels', channels)
        self.eta_max, raw_init = _bounded_raw_init(eta_max, eta_init)
        if not isinstance(deploy, bool):
            raise ValueError('RDCF.deploy must be a boolean')
        self.deploy = deploy

        if deploy:
            self.reparam_conv = self._make_reparam_conv()
        else:
            self.dw_3x3 = nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1,
                padding=1, groups=self.channels, bias=True)
            self.dw_1x9 = nn.Conv2d(
                self.channels, self.channels, kernel_size=(1, 9), stride=1,
                padding=(0, 4), groups=self.channels, bias=True)
            self.dw_9x1 = nn.Conv2d(
                self.channels, self.channels, kernel_size=(9, 1), stride=1,
                padding=(4, 0), groups=self.channels, bias=True)

        self.project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.raw_eta = nn.Parameter(torch.full(
            (1, self.channels, 1, 1), raw_init))

    def _make_reparam_conv(self, reference=None):
        conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=9, stride=1,
            padding=4, groups=self.channels, bias=True)
        if reference is not None:
            conv = conv.to(device=reference.device, dtype=reference.dtype)
        return conv

    def effective_eta(self):
        """Return the signed channel-wise scale bounded by ``eta_max``."""
        return self.eta_max * torch.tanh(self.raw_eta)

    def get_equivalent_kernel_bias(self):
        """Return the exact 9x9 depthwise kernel and bias for deployment."""
        if hasattr(self, 'reparam_conv'):
            return self.reparam_conv.weight, self.reparam_conv.bias

        # Conv2d weights are [C, 1, H, W] because all branches are depthwise.
        local = F.pad(self.dw_3x3.weight, (3, 3, 3, 3))
        horizontal = F.pad(self.dw_1x9.weight, (0, 0, 4, 4))
        vertical = F.pad(self.dw_9x1.weight, (4, 4, 0, 0))
        kernel = local + horizontal + vertical
        bias = self.dw_3x3.bias + self.dw_1x9.bias + self.dw_9x1.bias
        return kernel, bias

    def forward(self, x, return_aux=False):
        _validate_feature('RDCF input', x, self.channels)
        if hasattr(self, 'reparam_conv'):
            directional = self.reparam_conv(x)
        else:
            directional = (
                self.dw_3x3(x) + self.dw_1x9(x) + self.dw_9x1(x))
        residual = self.project(directional)
        eta = self.effective_eta().to(dtype=residual.dtype)
        output = x + eta * residual

        if not return_aux:
            return output
        return output, {
            'directional': directional,
            'residual': residual,
            'eta': eta,
        }

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse the three training branches into one depthwise 9x9 Conv2d.

        The method is idempotent and returns ``self`` to make checkpoint
        conversion code concise.
        """
        if hasattr(self, 'reparam_conv'):
            self.deploy = True
            return self

        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = self._make_reparam_conv(reference=kernel)
        reparam_conv.weight.copy_(kernel)
        reparam_conv.bias.copy_(bias)
        self.reparam_conv = reparam_conv

        del self.dw_3x3
        del self.dw_1x9
        del self.dw_9x1
        self.deploy = True
        return self

    convert_to_deploy = switch_to_deploy


class RDCFNeck(nn.Module):
    """Apply two independent RDCF blocks to N3/N4 and preserve N5 exactly."""

    def __init__(self, hidden_dim=256, eta_max=0.30, eta_init=0.05,
                 deploy=False):
        super().__init__()
        self.hidden_dim = _positive_int('RDCF.hidden_dim', hidden_dim)
        if not isinstance(deploy, bool):
            raise ValueError('RDCF.deploy must be a boolean')
        self.deploy = deploy
        self.rdcf3 = ReparamDirectionalContextBlock(
            channels=self.hidden_dim, eta_max=eta_max, eta_init=eta_init,
            deploy=deploy)
        self.rdcf4 = ReparamDirectionalContextBlock(
            channels=self.hidden_dim, eta_max=eta_max, eta_init=eta_init,
            deploy=deploy)

    def _validate_features(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise RuntimeError('RDCF expects exactly [N3, N4, N5]')
        for level, feature in zip((3, 4, 5), features):
            _validate_feature(f'RDCF N{level}', feature, self.hidden_dim)
        reference = features[0]
        for level, feature in zip((4, 5), features[1:]):
            if feature.shape[0] != reference.shape[0]:
                raise RuntimeError(
                    f'RDCF N{level} must share the N3 batch size')
            if feature.device != reference.device:
                raise RuntimeError(
                    f'RDCF N{level} must be on the same device as N3')
            if feature.dtype != reference.dtype:
                raise RuntimeError(
                    f'RDCF N{level} must use the same dtype as N3')

    def forward(self, features, return_aux=False):
        self._validate_features(features)
        n3, n4, n5 = features
        if return_aux:
            out3, aux3 = self.rdcf3(n3, return_aux=True)
            out4, aux4 = self.rdcf4(n4, return_aux=True)
            return [out3, out4, n5], {'N3': aux3, 'N4': aux4}
        return [self.rdcf3(n3), self.rdcf4(n4), n5]

    @torch.no_grad()
    def switch_to_deploy(self):
        """Convert both independent RDCF levels to deploy form in-place."""
        self.rdcf3.switch_to_deploy()
        self.rdcf4.switch_to_deploy()
        self.deploy = True
        return self

    convert_to_deploy = switch_to_deploy
