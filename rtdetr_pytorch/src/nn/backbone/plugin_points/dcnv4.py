"""Self-contained UAV-oriented DCNv4-style aggregation.

This is an in-project implementation, not a Python wrapper around the upstream
``DCNv4.ext`` CUDA package. It keeps the useful DCNv4 ideas -- grouped learned
offsets and input-dependent aggregation weights -- while using torchvision's
already-installed deformable sampler. No additional CUDA extension is needed.

For small UAVs, the branch predicts offsets from a locally contrast-enhanced
feature and explicitly returns high-frequency evidence. The enclosing
``ResidualPlugin`` applies a zero-initialized bounded gate, so enabling this
module starts exactly from the original RT-DETR backbone output.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def load_backend():
    """Return the packaged deformable sampler with a focused error message."""
    try:
        from torchvision.ops import deform_conv2d
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            'UAV-DCNv4 requires torchvision.ops.deform_conv2d from the '
            'torchvision already paired with this RT-DETR environment. No '
            'separate DCNv4 CUDA extension is required.'
        ) from error
    return deform_conv2d


def _validate_positive(name, value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive, got {value!r}')


class UAVDCNv4(nn.Module):
    """Grouped dynamic 3x3 aggregation with small-target detail preservation.

    Offsets are measured in feature-map pixels and bounded by ``max_offset``.
    For every offset group, the nine dynamic weights are softmax-normalized.
    The depthwise deformable kernel then performs per-channel aggregation while
    a pair of pointwise projections mixes channels before and after sampling.

    The evidence returned to the residual plugin is

    ``(dynamic_sample - local_mean) + detail_gain * (value - local_mean)``.

    The first term learns shape-adaptive evidence. The second preserves the
    high-frequency response of tiny targets instead of smoothing it away.
    """

    kernel_size = 3
    kernel_points = 9

    def __init__(self, channels, groups=4, max_offset=1.5, temperature=1.0,
                 detail_gain=0.5):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f'channels must be a positive integer, got {channels!r}')
        if not isinstance(groups, int) or groups <= 0 or channels % groups:
            raise ValueError('UAV-DCNv4 groups must be a positive divisor of channels')
        _validate_positive('max_offset', max_offset)
        _validate_positive('temperature', temperature)
        if (not isinstance(detail_gain, (int, float)) or not math.isfinite(detail_gain)
                or not 0 < detail_gain < 1):
            raise ValueError('detail_gain must be finite and strictly between 0 and 1')

        self.channels = channels
        self.groups = groups
        self.max_offset = float(max_offset)
        self.temperature = float(temperature)

        # Identity initialization gives the zero gate a meaningful detail
        # signal on its first update without perturbing the Baseline output.
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Conv2d(channels, channels, 1)

        # Cheap local context: depthwise 3x3 plus a pointwise offset/mask head.
        # A zero head means zero displacement and uniform dynamic weights.
        self.offset_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.SiLU(),
        )
        self.offset_mask = nn.Conv2d(
            channels, groups * self.kernel_points * 3, 1, bias=True)

        # Dynamic masks are supplied separately, while the learnable depthwise
        # spatial kernel retains per-channel signed filtering capacity.
        self.deform_weight = nn.Parameter(
            torch.ones(channels, 1, self.kernel_size, self.kernel_size))
        self.raw_detail_gain = nn.Parameter(
            torch.tensor(math.log(detail_gain / (1.0 - detail_gain))))
        self.bn = nn.BatchNorm2d(channels)

        nn.init.dirac_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.dirac_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.kaiming_normal_(self.offset_context[0].weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    @property
    def detail_gain(self):
        return self.raw_detail_gain.sigmoid()

    def _predict_offset_mask(self, guidance):
        prediction = self.offset_mask(self.offset_context(guidance))
        offset_channels = self.groups * self.kernel_points * 2
        raw_offset, mask_logits = prediction.split(
            [offset_channels, self.groups * self.kernel_points], dim=1)

        # Compute bounded offsets and normalized weights in FP32 to avoid AMP
        # overflow/underflow, then return to the feature dtype for sampling.
        offset = (raw_offset.float().tanh() * self.max_offset).to(guidance.dtype)
        n, _, h, w = mask_logits.shape
        mask = F.softmax(
            mask_logits.float().reshape(n, self.groups, self.kernel_points, h, w)
            / self.temperature,
            dim=2,
        ).reshape(n, self.groups * self.kernel_points, h, w).to(guidance.dtype)
        return offset.contiguous(), mask.contiguous()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'UAV-DCNv4 expects NCHW with C={self.channels}, '
                             f'got {tuple(x.shape)}')
        deform_conv2d = load_backend()
        value = self.value_proj(x)
        local_mean = F.avg_pool2d(value, 3, stride=1, padding=1,
                                  count_include_pad=True)
        high_frequency = value - local_mean
        detail_gain = self.detail_gain.to(value.dtype)
        guidance = value + detail_gain * high_frequency
        offset, mask = self._predict_offset_mask(guidance)
        # torchvision infers 128 depthwise weight groups from [128,1,3,3]
        # and the independent offset group count from offset/mask channels.
        dynamic = deform_conv2d(
            value,
            offset,
            self.deform_weight.to(dtype=value.dtype),
            bias=None,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            mask=mask,
        )
        evidence = dynamic - local_mean + detail_gain * high_frequency
        return self.bn(self.output_proj(evidence))


# Preserve the previous construction symbol for YAML/checkpoint tooling.
DCNv4Adapter = UAVDCNv4
