"""Context-Unexplained Residual Enhancement (CURE).

CURE is a light-weight feature residual plug-in.  It predicts a feature from
the surrounding context while permanently excluding the center of the
convolutional receptive field, then selectively enhances the part which the
context cannot explain.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormLayer, get_activation


__all__ = [
    "MaskedRingConv2d",
    "ContextUnexplainedResidualEnhancement",
]


def _validate_ring_sizes(kernel_size, exclude_center_size):
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("CURE context.kernel_size must be a positive odd integer")
    if (
        not isinstance(exclude_center_size, int)
        or exclude_center_size <= 0
        or exclude_center_size % 2 == 0
    ):
        raise ValueError(
            "CURE context.exclude_center_size must be a positive odd integer"
        )
    if exclude_center_size >= kernel_size:
        raise ValueError(
            "CURE context.exclude_center_size must be smaller than kernel_size"
        )


class MaskedRingConv2d(nn.Module):
    """A same-shape convolution whose center region can never contribute."""

    def __init__(
        self,
        channels,
        kernel_size=5,
        exclude_center_size=3,
        depthwise=True,
        bias=False,
    ):
        super().__init__()
        _validate_ring_sizes(kernel_size, exclude_center_size)
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("CURE channels must be a positive integer")

        self.in_channels = channels
        self.out_channels = channels
        self.kernel_size = kernel_size
        self.exclude_center_size = exclude_center_size
        self.padding = kernel_size // 2
        self.groups = channels if depthwise else 1
        self.depthwise = bool(depthwise)

        self.weight = nn.Parameter(
            torch.empty(channels, channels // self.groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(channels)) if bias else None

        mask = torch.ones(1, 1, kernel_size, kernel_size)
        start = (kernel_size - exclude_center_size) // 2
        end = start + exclude_center_size
        mask[:, :, start:end, start:end] = 0.0
        self.register_buffer("kernel_mask", mask)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Keep even the stored center weights at zero.  The forward-time mask is
        # still the hard guarantee if weights are later loaded or manipulated.
        with torch.no_grad():
            self.weight.mul_(self.kernel_mask)
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def masked_weight(self):
        return self.weight * self.kernel_mask

    def forward(self, x):
        return F.conv2d(
            x,
            self.masked_weight,
            self.bias,
            stride=1,
            padding=self.padding,
            groups=self.groups,
        )


class CenterExcludedContextPredictor(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=5,
        exclude_center_size=3,
        depthwise=True,
        act="silu",
    ):
        super().__init__()
        self.ring_conv = MaskedRingConv2d(
            channels,
            kernel_size=kernel_size,
            exclude_center_size=exclude_center_size,
            depthwise=depthwise,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = get_activation(act)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.ring_conv(x)
        x = self.act(self.norm(x))
        return self.pointwise(x)


class SpatialReliabilityGate(nn.Module):
    def __init__(self, channels, hidden_ratio=0.25, act="silu"):
        super().__init__()
        if hidden_ratio <= 0:
            raise ValueError("CURE gate.hidden_ratio must be positive")
        self.hidden_channels = max(int(channels * hidden_ratio), 16)
        self.reduce = ConvNormLayer(
            2 * channels + 1,
            self.hidden_channels,
            kernel_size=1,
            stride=1,
            act=act,
        )
        self.predict = nn.Conv2d(
            self.hidden_channels, 1, kernel_size=1, stride=1, bias=True
        )

    def forward(self, x, embedding, discrepancy):
        gate_feature = self.reduce(torch.cat([x, embedding, discrepancy], dim=1))
        return torch.sigmoid(self.predict(gate_feature))


class ContextUnexplainedResidualEnhancement(nn.Module):
    """Enhance context-unexplained semantic evidence without changing shape."""

    def __init__(
        self,
        channels,
        kernel_size=5,
        exclude_center_size=3,
        depthwise=True,
        use_residual_projection=True,
        gate_enabled=True,
        gate_hidden_ratio=0.25,
        spatial_gate=True,
        alpha_init=0.0,
        act="silu",
        eps=1.0e-6,
        debug=False,
    ):
        super().__init__()
        _validate_ring_sizes(kernel_size, exclude_center_size)
        if not spatial_gate:
            raise ValueError("CURE v1 supports only gate.spatial_gate=true")
        if eps <= 0:
            raise ValueError("CURE eps must be positive")

        self.channels = channels
        self.gate_enabled = bool(gate_enabled)
        self.use_residual_projection = bool(use_residual_projection)
        self.eps = float(eps)
        self.debug = bool(debug)

        self.context_predictor = CenterExcludedContextPredictor(
            channels,
            kernel_size=kernel_size,
            exclude_center_size=exclude_center_size,
            depthwise=depthwise,
            act=act,
        )
        self.residual_proj = ConvNormLayer(
            channels, channels, kernel_size=1, stride=1, act=act
        ) if self.use_residual_projection else nn.Identity()
        self.gate = SpatialReliabilityGate(
            channels, hidden_ratio=gate_hidden_ratio, act=act
        ) if self.gate_enabled else None
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.last_debug_info = {}
        self.last_debug_maps = {}

    @staticmethod
    def _statistics(tensor):
        detached = tensor.detach().float()
        return {
            "mean": float(detached.mean()),
            "std": float(detached.std(unbiased=False)),
            "min": float(detached.min()),
            "max": float(detached.max()),
        }

    def _record_debug(
        self, x, context, residual, discrepancy, gate, enhancement, output
    ):
        context_abs = context.detach().float().abs()
        residual_abs = residual.detach().float().abs()
        enhancement_abs = enhancement.detach().float().abs()
        feature_abs_mean = x.detach().float().abs().mean()
        ratio = enhancement_abs.mean() / (feature_abs_mean + self.eps)
        discrepancy_stats = self._statistics(discrepancy)
        gate_stats = self._statistics(gate)

        self.last_debug_info = {
            "input_shape": tuple(x.shape),
            "context_pred_shape": tuple(context.shape),
            "residual_shape": tuple(residual.shape),
            "discrepancy_shape": tuple(discrepancy.shape),
            "gate_shape": tuple(gate.shape),
            "output_shape": tuple(output.shape),
            "alpha": float(self.alpha.detach()),
            "context_abs_mean": float(context_abs.mean()),
            "residual_abs_mean": float(residual_abs.mean()),
            "residual_abs_std": float(residual_abs.std(unbiased=False)),
            "discrepancy_mean": discrepancy_stats["mean"],
            "discrepancy_std": discrepancy_stats["std"],
            "discrepancy_min": discrepancy_stats["min"],
            "discrepancy_max": discrepancy_stats["max"],
            "gate_mean": gate_stats["mean"],
            "gate_std": gate_stats["std"],
            "gate_min": gate_stats["min"],
            "gate_max": gate_stats["max"],
            "enhancement_abs_mean": float(enhancement_abs.mean()),
            "enhancement_to_feature_ratio": float(ratio),
        }
        self.last_debug_maps = {
            "discrepancy": discrepancy.detach().float().cpu(),
            "gate": gate.detach().float().cpu(),
            "residual_magnitude": residual.detach().float().abs().mean(
                dim=1, keepdim=True
            ).cpu(),
        }

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "CURE expects [B, {}, H, W], got {}".format(
                    self.channels, tuple(x.shape)
                )
            )

        context = self.context_predictor(x)
        residual = x - context
        embedding = self.residual_proj(residual)

        # Keep cosine normalization numerically stable under AMP, then return
        # the scalar map in the feature dtype for the gate convolutions.
        cosine = F.cosine_similarity(
            x.float(), context.float(), dim=1, eps=self.eps
        ).clamp(min=-1.0, max=1.0)
        discrepancy = (1.0 - cosine).unsqueeze(1).to(dtype=x.dtype)

        if self.gate is None:
            gate = torch.ones_like(discrepancy)
        else:
            gate = self.gate(x, embedding, discrepancy)

        enhancement = self.alpha.to(dtype=x.dtype) * gate * embedding
        output = x + enhancement
        if self.debug:
            self._record_debug(
                x, context, residual, discrepancy, gate, enhancement, output
            )
        return output
