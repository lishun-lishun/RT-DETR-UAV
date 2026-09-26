"""Persistent Detail Relay (PDR) branch for the PResNet backbone."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..common import ConvNormLayer, get_activation


__all__ = [
    'DetailMemory', 'DetailRelay', 'SemanticAgreementGate',
    'DetailInjection', 'PersistentDetailRelay',
]


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class DetailMemory(nn.Module):
    """Persistent Detail Memory: project the real stride-4 C2 feature."""

    def __init__(self, in_channels, detail_channels, act='relu'):
        super().__init__()
        _positive_int(in_channels, 'DetailMemory in_channels')
        _positive_int(detail_channels, 'DetailMemory detail_channels')
        self.projection = ConvNormLayer(
            in_channels, detail_channels, kernel_size=1, stride=1, act=act)

    def forward(self, c2):
        return self.projection(c2)


class _DepthwiseConvNormAct(nn.Module):
    """PResNet-style Conv/BN/activation with channel-wise 3x3 convolution."""

    def __init__(self, channels, act='relu'):
        super().__init__()
        _positive_int(channels, 'depthwise channels')
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DetailRelay(nn.Module):
    """Lossless-style Spatial Relay using space-to-depth before projection."""

    def __init__(self, in_channels, out_channels, act='relu'):
        super().__init__()
        _positive_int(in_channels, 'DetailRelay in_channels')
        _positive_int(out_channels, 'DetailRelay out_channels')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unshuffle = nn.PixelUnshuffle(2)
        self.reduce = ConvNormLayer(
            4 * in_channels, out_channels, kernel_size=1, stride=1, act=act)
        self.dwconv = _DepthwiseConvNormAct(out_channels, act=act)

    def forward(self, detail):
        if detail.ndim != 4 or detail.shape[1] != self.in_channels:
            raise ValueError('DetailRelay expects NCHW input with configured channels')
        height, width = detail.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                'PDR PixelUnshuffle requires even feature height and width; '
                f'got {height}x{width}. No silent crop is performed.')
        # PixelUnshuffle preserves every 2x2 sample as four channel phases.
        return self.dwconv(self.reduce(self.unshuffle(detail)))


class SemanticAgreementGate(nn.Module):
    """Cosine semantic agreement with a non-zero detail-preservation floor."""

    def __init__(self, rho=0.25, tau=0.2, learnable_theta=True,
                 theta_init=0.0, eps=1e-6):
        super().__init__()
        if not 0.0 <= float(rho) < 1.0:
            raise ValueError('PDR gate rho must be in [0, 1)')
        if not math.isfinite(float(tau)) or float(tau) <= 0:
            raise ValueError('PDR gate tau must be finite and positive')
        if not math.isfinite(float(theta_init)):
            raise ValueError('PDR theta_init must be finite')
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError('PDR gate eps must be finite and positive')
        if not isinstance(learnable_theta, bool):
            raise ValueError('PDR learnable_theta must be boolean')
        self.rho = float(rho)
        self.tau = float(tau)
        self.eps = float(eps)
        theta = torch.tensor(float(theta_init), dtype=torch.float32)
        if learnable_theta:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)

    def forward(self, main, detail):
        if main.shape != detail.shape:
            raise ValueError(
                f'PDR semantic gate shape mismatch: {tuple(main.shape)} vs '
                f'{tuple(detail.shape)}')
        if main.device != detail.device or main.dtype != detail.dtype:
            raise ValueError('PDR semantic gate requires matching dtype and device')
        # Only cosine arithmetic is promoted to FP32 under AMP. The surrounding
        # backbone remains autocast-enabled.
        agreement = F.cosine_similarity(
            main.float(), detail.float(), dim=1, eps=self.eps).unsqueeze(1)
        gate32 = self.rho + (1.0 - self.rho) * torch.sigmoid(
            (agreement - self.theta.float()) / self.tau)
        gate = gate32.to(dtype=detail.dtype)
        return gate, agreement


class DetailInjection(nn.Module):
    """Semantic-Verified Detail Injection into one original main feature."""

    def __init__(self, detail_channels, main_channels, use_semantic_gate=True,
                 gate=None, fusion=None, debug=False):
        super().__init__()
        _positive_int(detail_channels, 'DetailInjection detail_channels')
        _positive_int(main_channels, 'DetailInjection main_channels')
        if not isinstance(use_semantic_gate, bool) or not isinstance(debug, bool):
            raise ValueError('PDR use_semantic_gate/debug must be boolean')
        gate = {} if gate is None else dict(gate)
        fusion = {} if fusion is None else dict(fusion)
        allowed_gate = {'rho', 'tau', 'learnable_theta', 'theta_init', 'eps'}
        allowed_fusion = {'alpha_max', 'alpha_init'}
        if set(gate) - allowed_gate:
            raise ValueError(f'Unknown PDR gate options: {sorted(set(gate) - allowed_gate)}')
        if set(fusion) - allowed_fusion:
            raise ValueError(
                f'Unknown PDR fusion options: {sorted(set(fusion) - allowed_fusion)}')

        alpha_max = float(fusion.get('alpha_max', 0.5))
        alpha_init = float(fusion.get('alpha_init', 0.1))
        if not math.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError('PDR alpha_max must be finite and positive')
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError('PDR alpha_init must be in (0, alpha_max)')
        ratio = alpha_init / alpha_max
        raw_init = math.log(ratio / (1.0 - ratio))

        self.projection = ConvNormLayer(
            detail_channels, main_channels, kernel_size=1, stride=1, act=None)
        self.semantic_gate = (SemanticAgreementGate(**gate)
                              if use_semantic_gate else None)
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.alpha_max = alpha_max
        self.debug = debug
        self.last_debug_stats = None
        self.last_debug_tensors = None

    @property
    def alpha(self):
        return self.alpha_max * self.raw_alpha.float().sigmoid()

    @property
    def theta(self):
        return None if self.semantic_gate is None else self.semantic_gate.theta

    def forward(self, main, detail):
        projected = self.projection(detail)
        if projected.shape != main.shape:
            raise ValueError(
                f'PDR injection shape mismatch: main={tuple(main.shape)}, '
                f'detail={tuple(projected.shape)}')
        if projected.device != main.device or projected.dtype != main.dtype:
            raise ValueError('PDR injection requires matching dtype and device')

        if self.semantic_gate is None:
            gate = None
            agreement = None
            verified = projected
        else:
            gate, agreement = self.semantic_gate(main, projected)
            verified = gate * projected
        residual = self.alpha.to(dtype=verified.dtype) * verified
        output = main + residual

        if self.debug:
            with torch.no_grad():
                main32 = main.detach().float()
                detail32 = projected.detach().float()
                residual32 = residual.detach().float()
                stats = {
                    'raw_alpha': self.raw_alpha.detach(),
                    'alpha_eff': self.alpha.detach(),
                    'detail_norm': torch.linalg.vector_norm(detail32),
                    'main_norm': torch.linalg.vector_norm(main32),
                    'injection_norm': torch.linalg.vector_norm(residual32),
                }
                tensors = {
                    'detail_energy': detail32.square().mean(1).sqrt(),
                    'main_energy': main32.square().mean(1).sqrt(),
                    'injection_energy': residual32.square().mean(1).sqrt(),
                }
                if gate is not None:
                    gate32 = gate.detach().float()
                    stats.update({
                        'theta': self.theta.detach(),
                        'gate_mean': gate32.mean(),
                        'gate_std': gate32.std(unbiased=False),
                        'gate_min': gate32.amin(),
                        'gate_max': gate32.amax(),
                    })
                    tensors['gate'] = gate32.squeeze(1)
                    tensors['agreement'] = agreement.detach().float().squeeze(1)
                self.last_debug_stats = stats
                self.last_debug_tensors = tensors
        return output


class PersistentDetailRelay(nn.Module):
    """C2 -> D2 -> D3 -> D4 persistent detail path and C3/C4 injections."""

    def __init__(self, backbone_channels, detail_channels=None,
                 use_relay3=True, use_relay4=True, use_semantic_gate=True,
                 gate=None, fusion=None, act='relu', debug=False):
        super().__init__()
        if not isinstance(backbone_channels, (list, tuple)) or len(backbone_channels) < 3:
            raise ValueError('PDR backbone_channels must provide C2/C3/C4 channels')
        c2_main, c3_main, c4_main = [
            _positive_int(value, f'PDR backbone channel {index}')
            for index, value in enumerate(backbone_channels[:3], 2)]
        detail_channels = ({'c2': 32, 'c3': 64, 'c4': 96}
                           if detail_channels is None else dict(detail_channels))
        if set(detail_channels) != {'c2', 'c3', 'c4'}:
            raise ValueError('PDR detail_channels must contain exactly c2/c3/c4')
        d2, d3, d4 = [
            _positive_int(detail_channels[name], f'PDR detail_channels.{name}')
            for name in ('c2', 'c3', 'c4')]
        for name, value in (('use_relay3', use_relay3),
                            ('use_relay4', use_relay4),
                            ('use_semantic_gate', use_semantic_gate),
                            ('debug', debug)):
            if not isinstance(value, bool):
                raise ValueError(f'PDR {name} must be boolean')
        if not use_relay3:
            raise ValueError('PDR requires use_relay3=true in the first implementation')
        if use_relay4 and not use_relay3:
            raise ValueError('PDR relay4 requires relay3')

        self.use_relay3 = use_relay3
        self.use_relay4 = use_relay4
        self.use_semantic_gate = use_semantic_gate
        self.debug = debug

        self.detail_memory = DetailMemory(c2_main, d2, act=act)
        self.relay3 = DetailRelay(d2, d3, act=act)
        self.injection3 = DetailInjection(
            d3, c3_main, use_semantic_gate=use_semantic_gate,
            gate=gate, fusion=fusion, debug=debug)
        self.relay4 = DetailRelay(d3, d4, act=act) if use_relay4 else None
        self.injection4 = (DetailInjection(
            d4, c4_main, use_semantic_gate=use_semantic_gate,
            gate=gate, fusion=fusion, debug=debug) if use_relay4 else None)
        self.last_debug_stats = None
        self.last_debug_tensors = None

    def make_details(self, c2):
        """Create the persistent chain; D4 is always derived from D3."""
        d2 = self.detail_memory(c2)
        d3 = self.relay3(d2)
        d4 = self.relay4(d3) if self.relay4 is not None else None
        return d2, d3, d4

    def inject3(self, main3, detail3):
        output = self.injection3(main3, detail3)
        self._collect_debug()
        return output

    def inject4(self, main4, detail4):
        if self.injection4 is None or detail4 is None:
            raise RuntimeError('PDR relay4/injection4 is disabled')
        output = self.injection4(main4, detail4)
        self._collect_debug()
        return output

    def _collect_debug(self):
        if not self.debug:
            return
        stats = {}
        tensors = {}
        for level, injection in ((3, self.injection3), (4, self.injection4)):
            if injection is None or injection.last_debug_stats is None:
                continue
            # Stable public names for logging/visualization.  Keep all values
            # as device tensors: enabling debug must not add an implicit CUDA
            # synchronization on every training iteration.
            stat_names = {
                'raw_alpha': f'raw_alpha{level}',
                'alpha_eff': f'alpha{level}_eff',
                'theta': f'theta{level}',
                'gate_mean': f'gate{level}_mean',
                'gate_std': f'gate{level}_std',
                'gate_min': f'gate{level}_min',
                'gate_max': f'gate{level}_max',
                'detail_norm': f'detail{level}_norm',
                'main_norm': f'main{level}_norm',
                'injection_norm': f'injection{level}_norm',
            }
            tensor_names = {
                'gate': f'gate{level}',
                'agreement': f'agreement{level}',
                'detail_energy': f'detail{level}_energy',
                'main_energy': f'main{level}_energy',
                'injection_energy': f'injection{level}_energy',
            }
            stats.update({stat_names[name]: value
                          for name, value in injection.last_debug_stats.items()})
            tensors.update({tensor_names[name]: value
                            for name, value in injection.last_debug_tensors.items()})
        self.last_debug_stats = stats
        self.last_debug_tensors = tensors
