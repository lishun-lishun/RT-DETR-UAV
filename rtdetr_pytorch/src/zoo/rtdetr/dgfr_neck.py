"""Direct Global Fusion Residual (DGFR) neck for RT-DETR.

DGFR is an optional post-CCFF residual refinement.  It uses the three
projected/AIFI-updated input features (X3/X4/X5) to build an independent,
direct all-scale fusion feature for each original CCFF output (O3/O4/O5),
then injects it through a bounded channel-wise LayerScale.  The module does
not add a detection level and never changes the public P3/P4/P5 interface.

``ConvNormLayer`` and ``CSPRepLayer`` are accepted as factories so that the
caller can reuse the exact blocks defined by :mod:`hybrid_encoder` without
creating a module import cycle.  A lazy fallback import keeps this class
convenient to instantiate directly in isolated tests and tools.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DirectScaleAdapter', 'DirectGlobalFusion', 'DGFRNeck']


_LEVELS = (3, 4, 5)


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _positive_float(name, value):
    value = _finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f'{name} must be positive')
    return value


def _validate_level(name, value):
    if (not isinstance(value, int) or isinstance(value, bool)
            or value not in _LEVELS):
        raise ValueError(f'{name} must be one of {_LEVELS}')
    return value


def _resolve_factories(conv_norm_factory, fusion_factory=None):
    """Resolve the native HybridEncoder blocks without a top-level import."""
    if conv_norm_factory is None or fusion_factory is None:
        # Delayed until construction: hybrid_encoder can safely import this
        # module first and pass its already-defined classes explicitly.
        from .hybrid_encoder import ConvNormLayer, CSPRepLayer
        if conv_norm_factory is None:
            conv_norm_factory = ConvNormLayer
        if fusion_factory is None:
            fusion_factory = CSPRepLayer
    if not callable(conv_norm_factory):
        raise TypeError('conv_norm_factory must be callable')
    if fusion_factory is not None and not callable(fusion_factory):
        raise TypeError('fusion_factory must be callable')
    return conv_norm_factory, fusion_factory


def _validate_bchw(name, value, channels):
    if not torch.is_tensor(value) or value.ndim != 4:
        raise RuntimeError(f'DGFR {name} must be a BCHW tensor')
    if value.shape[1] != channels:
        raise RuntimeError(
            f'DGFR {name} has {value.shape[1]} channels, expected {channels}')
    if not value.is_floating_point() or value.is_complex():
        raise RuntimeError(f'DGFR {name} must use a real floating-point dtype')


class DirectScaleAdapter(nn.Module):
    """Align one projected source level directly to one target level.

    Upsampling always uses nearest-neighbour interpolation with the *actual*
    target tensor size.  Downsampling uses one 3x3/stride-2/padding-1 native
    ``ConvNormLayer`` per level transition; X3 -> target 5 therefore contains
    two consecutive stride-2 layers and never a stride-4 convolution/pool.
    Every source-target pair owns its own adapter parameters.
    """

    def __init__(self, source_level, target_level, hidden_dim=256,
                 fusion_channels=64, conv_norm_factory=None, act='silu'):
        super().__init__()
        self.source_level = _validate_level(
            'DGFR source_level', source_level)
        self.target_level = _validate_level(
            'DGFR target_level', target_level)
        self.hidden_dim = _positive_int('DGFR.hidden_dim', hidden_dim)
        self.fusion_channels = _positive_int(
            'DGFR.fusion_channels', fusion_channels)
        conv_norm_factory, _ = _resolve_factories(
            conv_norm_factory, fusion_factory=lambda **_: None)

        if self.source_level <= self.target_level:
            down_steps = self.target_level - self.source_level
            if down_steps == 0:
                self.layers = conv_norm_factory(
                    self.hidden_dim, self.fusion_channels, 1, 1,
                    padding=0, act=act)
            else:
                layers = []
                in_channels = self.hidden_dim
                for _ in range(down_steps):
                    layers.append(conv_norm_factory(
                        in_channels, self.fusion_channels, 3, 2,
                        padding=1, act=act))
                    in_channels = self.fusion_channels
                self.layers = nn.Sequential(*layers)
        else:
            # The resize deliberately precedes this 1x1 projection, matching
            # the prescribed direct-alignment order.
            self.layers = conv_norm_factory(
                self.hidden_dim, self.fusion_channels, 1, 1,
                padding=0, act=act)

    @property
    def direction(self):
        if self.source_level < self.target_level:
            return 'down'
        if self.source_level > self.target_level:
            return 'up'
        return 'same'

    def forward(self, source, target_size):
        _validate_bchw('adapter source', source, self.hidden_dim)
        if (not isinstance(target_size, (tuple, list))
                or len(target_size) != 2
                or any(not isinstance(value, int) or value < 1
                       for value in target_size)):
            raise RuntimeError('DGFR target_size must contain two positive ints')
        target_size = tuple(target_size)

        if self.direction == 'up':
            source = F.interpolate(
                source, size=target_size, mode='nearest')
            aligned = self.layers(source)
        else:
            aligned = self.layers(source)

        if tuple(aligned.shape[-2:]) != target_size:
            raise RuntimeError(
                'DGFR downsample geometry does not match the requested target '
                f'size: got {tuple(aligned.shape[-2:])}, expected {target_size}')
        return aligned


class DirectGlobalFusion(nn.Module):
    """Build one target scale from three independent direct adapters."""

    def __init__(self, target_level, hidden_dim=256, fusion_channels=64,
                 conv_norm_factory=None, fusion_factory=None, act='silu'):
        super().__init__()
        self.target_level = _validate_level(
            'DGFR target_level', target_level)
        self.hidden_dim = _positive_int('DGFR.hidden_dim', hidden_dim)
        self.fusion_channels = _positive_int(
            'DGFR.fusion_channels', fusion_channels)
        conv_norm_factory, fusion_factory = _resolve_factories(
            conv_norm_factory, fusion_factory)

        self.adapters = nn.ModuleDict({
            f'x{source_level}': DirectScaleAdapter(
                source_level=source_level,
                target_level=self.target_level,
                hidden_dim=self.hidden_dim,
                fusion_channels=self.fusion_channels,
                conv_norm_factory=conv_norm_factory,
                act=act)
            for source_level in _LEVELS
        })
        self.fusion = fusion_factory(
            in_channels=3 * self.fusion_channels,
            out_channels=self.hidden_dim,
            num_blocks=1,
            expansion=0.5,
            act=act)

    def forward(self, proj_feats, return_aux=False):
        if not isinstance(proj_feats, (tuple, list)) or len(proj_feats) != 3:
            raise RuntimeError('DGFR proj_feats must contain X3, X4 and X5')
        for index, feature in enumerate(proj_feats):
            _validate_bchw(f'X{index + 3}', feature, self.hidden_dim)
        batch_sizes = {feature.shape[0] for feature in proj_feats}
        if len(batch_sizes) != 1:
            raise RuntimeError('DGFR projected features must share a batch size')

        target_size = tuple(proj_feats[self.target_level - 3].shape[-2:])
        aligned = [
            self.adapters[f'x{source_level}'](
                proj_feats[source_level - 3], target_size)
            for source_level in _LEVELS
        ]
        concatenated = torch.cat(aligned, dim=1)
        expected_channels = 3 * self.fusion_channels
        if concatenated.shape[1] != expected_channels:
            raise RuntimeError(
                f'DGFR concat has {concatenated.shape[1]} channels, '
                f'expected {expected_channels}')
        evidence = self.fusion(concatenated)
        if (evidence.shape[1] != self.hidden_dim
                or tuple(evidence.shape[-2:]) != target_size):
            raise RuntimeError('DGFR fusion block changed its output contract')
        if not return_aux:
            return evidence
        return evidence, {
            'aligned': tuple(aligned),
            'concatenated': concatenated,
        }


class DGFRNeck(nn.Module):
    """Direct all-scale fusion with bounded channel-wise residual injection."""

    def __init__(self, hidden_dim=256, fusion_channels=64,
                 gamma_max=0.25, gamma_init=0.05, debug=False,
                 debug_interval=100, conv_norm_factory=None,
                 fusion_factory=None, act='silu'):
        super().__init__()
        self.hidden_dim = _positive_int('DGFR.hidden_dim', hidden_dim)
        self.fusion_channels = _positive_int(
            'DGFR.fusion_channels', fusion_channels)
        self.gamma_max = _positive_float('DGFR.gamma_max', gamma_max)
        gamma_init = _finite_float('DGFR.gamma_init', gamma_init)
        if abs(gamma_init) >= self.gamma_max:
            raise ValueError(
                'abs(DGFR.gamma_init) must be smaller than DGFR.gamma_max')
        if not isinstance(debug, bool):
            raise ValueError('DGFR.debug must be a boolean')
        self.debug_interval = _positive_int(
            'DGFR.debug_interval', debug_interval)
        self.debug = debug

        conv_norm_factory, fusion_factory = _resolve_factories(
            conv_norm_factory, fusion_factory)
        fusion_kwargs = dict(
            hidden_dim=self.hidden_dim,
            fusion_channels=self.fusion_channels,
            conv_norm_factory=conv_norm_factory,
            fusion_factory=fusion_factory,
            act=act)
        self.fusion3 = DirectGlobalFusion(target_level=3, **fusion_kwargs)
        self.fusion4 = DirectGlobalFusion(target_level=4, **fusion_kwargs)
        self.fusion5 = DirectGlobalFusion(target_level=5, **fusion_kwargs)

        raw_init = math.atanh(gamma_init / self.gamma_max)
        raw_shape = (1, self.hidden_dim, 1, 1)
        self.raw_gamma3 = nn.Parameter(torch.full(raw_shape, raw_init))
        self.raw_gamma4 = nn.Parameter(torch.full(raw_shape, raw_init))
        self.raw_gamma5 = nn.Parameter(torch.full(raw_shape, raw_init))

        self.last_debug_stats = None
        self._debug_call_count = 0

    def effective_gamma(self, level=None):
        """Return one, or all three, bounded channel-wise LayerScales."""
        if level is None:
            return tuple(self.effective_gamma(value) for value in _LEVELS)
        level = _validate_level('DGFR gamma level', level)
        raw_gamma = getattr(self, f'raw_gamma{level}')
        return self.gamma_max * torch.tanh(raw_gamma)

    @staticmethod
    def _norm(value):
        return torch.linalg.vector_norm(value.float()).detach()

    def _record_debug_stats(self, evidences, outs, outputs, gammas):
        stats = {}
        for level, evidence, original, output, gamma in zip(
                _LEVELS, evidences, outs, outputs, gammas):
            gamma_fp32 = gamma.float()
            stats[f'gamma{level}_mean'] = gamma_fp32.mean().detach()
            stats[f'gamma{level}_min'] = gamma_fp32.min().detach()
            stats[f'gamma{level}_max'] = gamma_fp32.max().detach()
            stats[f'e{level}_to_o{level}_norm_ratio'] = (
                self._norm(evidence)
                / self._norm(original).clamp_min(1e-12))
            stats[f'y{level}_norm'] = self._norm(output)
        self.last_debug_stats = stats

    def forward(self, proj_feats, outs, return_aux=False):
        if not isinstance(proj_feats, (tuple, list)) or len(proj_feats) != 3:
            raise RuntimeError('DGFR proj_feats must contain X3, X4 and X5')
        if not isinstance(outs, (tuple, list)) or len(outs) != 3:
            raise RuntimeError('DGFR outs must contain O3, O4 and O5')
        for index, (projected, original) in enumerate(zip(proj_feats, outs)):
            level = index + 3
            _validate_bchw(f'X{level}', projected, self.hidden_dim)
            _validate_bchw(f'O{level}', original, self.hidden_dim)
            if (projected.shape[0] != original.shape[0]
                    or projected.shape[-2:] != original.shape[-2:]):
                raise RuntimeError(
                    f'DGFR X{level} and O{level} must share batch/spatial shape')

        evidences = (
            self.fusion3(proj_feats),
            self.fusion4(proj_feats),
            self.fusion5(proj_feats),
        )
        gammas = self.effective_gamma()
        outputs = tuple(
            original + gamma.to(dtype=original.dtype) * evidence
            for original, evidence, gamma in zip(outs, evidences, gammas)
        )

        self._debug_call_count += 1
        if (self.debug
                and (self._debug_call_count - 1) % self.debug_interval == 0):
            self._record_debug_stats(
                evidences, outs, outputs, gammas)
        elif not self.debug:
            self.last_debug_stats = None

        result = list(outputs)
        if not return_aux:
            return result
        return result, {
            'evidences': evidences,
            'gammas': gammas,
        }
