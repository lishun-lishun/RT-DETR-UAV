"""Agreement-Calibrated Residual Routing (ACR) for RT-DETR CCFF.

ACR is intentionally limited to cross-scale feature routing after the
HybridEncoder input projections.  It neither adds a feature level nor changes
the existing CSPRepLayer, AIFI, decoder, matcher, or loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'CrossScaleEnergyCalibration',
    'SemanticAgreementRouter',
    'ScaleExclusiveResidualRouter',
    'ACRFusion',
]


def _validate_positive(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive number')
    return float(value)


def _tensor_stats(value):
    value = value.detach().float()
    return {
        'mean': value.mean().item(),
        'std': value.std(unbiased=False).item(),
        'min': value.min().item(),
        'max': value.max().item(),
    }


class CrossScaleEnergyCalibration(nn.Module):
    """Match the per-channel RMS of a deep feature to a shallow feature."""

    def __init__(self, scale_min=0.5, scale_max=2.0,
                 detach_scale=True, eps=1e-6):
        super().__init__()
        self.scale_min = _validate_positive('ACR.scale_min', scale_min)
        self.scale_max = _validate_positive('ACR.scale_max', scale_max)
        if self.scale_min > self.scale_max:
            raise ValueError('ACR.scale_min must not exceed ACR.scale_max')
        if not isinstance(detach_scale, bool):
            raise ValueError('ACR.detach_scale must be boolean')
        self.detach_scale = detach_scale
        self.eps = _validate_positive('ACR.eps', eps)

    def forward(self, shallow, upsampled):
        # Statistics stay FP32 under autocast; the routed feature returns to
        # the original activation dtype so the rest of the encoder keeps AMP.
        shallow_fp32 = shallow.float()
        upsampled_fp32 = upsampled.float()
        rms_shallow = torch.sqrt(
            shallow_fp32.square().mean(dim=(-2, -1), keepdim=True) + self.eps)
        rms_upsampled = torch.sqrt(
            upsampled_fp32.square().mean(dim=(-2, -1), keepdim=True) + self.eps)
        scale = (rms_shallow / (rms_upsampled + self.eps)).clamp(
            self.scale_min, self.scale_max)
        if self.detach_scale:
            scale = scale.detach()
        calibrated = upsampled * scale.to(dtype=upsampled.dtype)
        return calibrated, scale


class SemanticAgreementRouter(nn.Module):
    """Gate only deep semantic injection using per-pixel cosine agreement."""

    def __init__(self, rho=0.2, theta=0.0, tau=0.2, eps=1e-6):
        super().__init__()
        if not isinstance(rho, (int, float)) or isinstance(rho, bool) \
                or not 0 <= rho <= 1:
            raise ValueError('ACR.semantic.rho must be in [0, 1]')
        if not isinstance(theta, (int, float)) or isinstance(theta, bool):
            raise ValueError('ACR.semantic.theta must be numeric')
        self.rho = float(rho)
        self.theta = float(theta)
        self.tau = _validate_positive('ACR.semantic.tau', tau)
        self.eps = _validate_positive('ACR.eps', eps)

    def forward(self, shallow, calibrated):
        agreement = F.cosine_similarity(
            shallow.float(), calibrated.float(), dim=1, eps=self.eps).unsqueeze(1)
        gate = self.rho + (1.0 - self.rho) * torch.sigmoid(
            (agreement - self.theta) / self.tau)
        routed = calibrated * gate.to(dtype=calibrated.dtype)
        return routed, agreement, gate


class _DetailDownsample(nn.Module):
    """Independent Conv+Norm route matching the original PAN stride style."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2,
                              padding=1, bias=False)
        # Keep the attribute name ``norm`` so it follows the original
        # HybridEncoder norm/bias zero-weight-decay optimizer rule.
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, value):
        return self.norm(self.conv(value))


class ScaleExclusiveResidualRouter(nn.Module):
    """Extract spatially salient shallow-minus-calibrated residual detail."""

    def __init__(self, channels, route_name, theta=1.0, tau=0.5,
                 beta_max=0.5, beta_init=0.1, eps=1e-6):
        super().__init__()
        if route_name not in ('34', '45'):
            raise ValueError('ACR detail route_name must be 34 or 45')
        if not isinstance(theta, (int, float)) or isinstance(theta, bool):
            raise ValueError('ACR.detail.theta must be numeric')
        self.theta = float(theta)
        self.tau = _validate_positive('ACR.detail.tau', tau)
        self.beta_max = _validate_positive('ACR.detail.beta_max', beta_max)
        beta_init = _validate_positive('ACR.detail.beta_init', beta_init)
        if beta_init >= self.beta_max:
            raise ValueError('ACR.detail.beta_init must be smaller than beta_max')
        self.eps = _validate_positive('ACR.eps', eps)
        self.route_name = route_name
        probability = beta_init / self.beta_max
        raw_init = math.log(probability / (1.0 - probability))
        self.register_parameter(
            f'raw_beta{route_name}', nn.Parameter(torch.tensor(raw_init)))
        self.detail_downsample = _DetailDownsample(channels)

    @property
    def raw_beta(self):
        return getattr(self, f'raw_beta{self.route_name}')

    def effective_beta(self):
        return self.beta_max * torch.sigmoid(self.raw_beta)

    def forward(self, shallow, calibrated):
        residual = shallow - calibrated
        residual_fp32 = residual.float()
        saliency = torch.sqrt(
            residual_fp32.square().mean(dim=1, keepdim=True) + self.eps)
        mean = saliency.mean(dim=(-2, -1), keepdim=True)
        variance = (saliency - mean).square().mean(dim=(-2, -1), keepdim=True)
        std = torch.sqrt(variance + self.eps)
        z = (saliency - mean) / (std + self.eps)
        gate = torch.sigmoid((z - self.theta) / self.tau)
        routed_residual = residual * gate.to(dtype=residual.dtype)
        return routed_residual, z, gate

    def inject(self, base, routed_residual):
        detail = self.detail_downsample(routed_residual)
        if detail.shape != base.shape:
            raise RuntimeError(
                f'ACR detail route {self.route_name} shape mismatch: '
                f'base={tuple(base.shape)}, detail={tuple(detail.shape)}')
        beta = self.effective_beta()
        output = base + beta.to(dtype=detail.dtype) * detail
        return output, detail, beta


class ACRFusion(nn.Module):
    """One deep-to-shallow semantic route and its reverse detail route."""

    def __init__(self, channels, route_name, *, energy_calibration=True,
                 semantic_routing=True, detail_routing=True,
                 scale_min=0.5, scale_max=2.0, detach_scale=True,
                 semantic=None, detail=None, eps=1e-6,
                 debug=False, debug_interval=100):
        super().__init__()
        for name, value in (
                ('energy_calibration', energy_calibration),
                ('semantic_routing', semantic_routing),
                ('detail_routing', detail_routing),
                ('debug', debug)):
            if not isinstance(value, bool):
                raise ValueError(f'ACR.{name} must be boolean')
        if not isinstance(debug_interval, int) or isinstance(debug_interval, bool) \
                or debug_interval < 1:
            raise ValueError('ACR.debug_interval must be a positive integer')
        if semantic is not None and not isinstance(semantic, dict):
            raise ValueError('ACR.semantic must be a mapping')
        if detail is not None and not isinstance(detail, dict):
            raise ValueError('ACR.detail must be a mapping')
        semantic = {} if semantic is None else dict(semantic)
        detail = {} if detail is None else dict(detail)
        unknown_semantic = set(semantic) - {'rho', 'theta', 'tau'}
        unknown_detail = set(detail) - {'theta', 'tau', 'beta_max', 'beta_init'}
        if unknown_semantic:
            raise ValueError(f'Unknown ACR.semantic options: {sorted(unknown_semantic)}')
        if unknown_detail:
            raise ValueError(f'Unknown ACR.detail options: {sorted(unknown_detail)}')

        self.route_name = route_name
        self.energy_enabled = energy_calibration
        self.semantic_enabled = semantic_routing
        self.detail_enabled = detail_routing
        self.debug = debug
        self.debug_interval = debug_interval
        self._debug_step = 0
        self._pending_stats = None
        self.energy = CrossScaleEnergyCalibration(
            scale_min, scale_max, detach_scale, eps)
        self.semantic_router = SemanticAgreementRouter(eps=eps, **semantic)
        self.detail_router = (ScaleExclusiveResidualRouter(
            channels, route_name, eps=eps, **detail)
                              if detail_routing else None)

    def forward(self, shallow, upsampled):
        if shallow.shape != upsampled.shape:
            raise RuntimeError(
                f'ACR{self.route_name} requires aligned features, got '
                f'{tuple(shallow.shape)} and {tuple(upsampled.shape)}')
        if self.energy_enabled:
            calibrated, scale = self.energy(shallow, upsampled)
        else:
            calibrated = upsampled
            scale = torch.ones_like(upsampled[:, :, :1, :1], dtype=torch.float32)
        if self.semantic_enabled:
            routed, agreement, semantic_gate = self.semantic_router(shallow, calibrated)
        else:
            routed = calibrated
            agreement = F.cosine_similarity(
                shallow.float(), calibrated.float(), dim=1, eps=self.energy.eps).unsqueeze(1)
            semantic_gate = torch.ones_like(agreement)
        if self.detail_router is not None:
            detail_residual, detail_z, detail_gate = self.detail_router(shallow, calibrated)
        else:
            detail_residual = None
            detail_z = detail_gate = None

        if self.debug:
            self._pending_stats = {
                'scale': _tensor_stats(scale),
                'semantic_gate': _tensor_stats(semantic_gate),
                'agreement': _tensor_stats(agreement),
                'detail_z': _tensor_stats(detail_z) if detail_z is not None else None,
                'detail_gate': (_tensor_stats(detail_gate)
                                if detail_gate is not None else None),
                'shallow_feature_norm': shallow.detach().float().square().mean().sqrt().item(),
            }
        return routed, detail_residual

    def inject_detail(self, base, detail_residual):
        if self.detail_router is None:
            return base
        if detail_residual is None:
            raise RuntimeError(f'ACR{self.route_name} detail residual is missing')
        output, detail, beta = self.detail_router.inject(base, detail_residual)
        if self.debug:
            self._debug_step += 1
            if (self._debug_step - 1) % self.debug_interval == 0:
                stats = dict(self._pending_stats or {})
                stats.update({
                    'beta_eff': beta.detach().float().item(),
                    'detail_route_norm': detail.detach().float().square().mean().sqrt().item(),
                    'base_feature_norm': base.detach().float().square().mean().sqrt().item(),
                })
                print(f'ACR{self.route_name} stats: {stats}')
            self._pending_stats = None
        return output
