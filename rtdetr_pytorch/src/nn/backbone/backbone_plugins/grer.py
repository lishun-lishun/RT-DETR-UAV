"""Global-Rarity Evidence Relay for the S3 -> S4 bypass."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..common import ConvNormLayer
from .cced import resolve_groups


def _validate_options(projection, fusion, rarity):
    projection = {} if projection is None else dict(projection)
    fusion = {} if fusion is None else dict(fusion)
    rarity = {} if rarity is None else dict(rarity)
    unknown_projection = set(projection) - {'norm', 'activation'}
    unknown_fusion = set(fusion) - {'alpha_init', 'alpha_max'}
    unknown_rarity = set(rarity) - {'threshold', 'temperature'}
    if unknown_projection or unknown_fusion or unknown_rarity:
        raise ValueError('GRER unsupported options: '
                         f'projection={sorted(unknown_projection)}, '
                         f'fusion={sorted(unknown_fusion)}, '
                         f'rarity={sorted(unknown_rarity)}')
    if projection.get('norm', True) is not True:
        raise ValueError('GRER projection.norm must be true in the first implementation')
    if projection.get('activation', False) not in (False, None):
        raise ValueError('GRER projection.activation must be false')
    return (float(fusion.get('alpha_init', 0.0)),
            float(fusion.get('alpha_max', 0.20)),
            float(rarity.get('threshold', 2.0)),
            float(rarity.get('temperature', 1.0)))


class GRERRelay(nn.Module):
    """Relay spatially rare stride-8 features into the stride-16 stage."""

    def __init__(self, ch_in, ch_out, transition='3to4', groups=8,
                 local_kernel=3, eps=1e-6, rarity=None, projection=None,
                 fusion=None, debug=False, debug_interval=100):
        super().__init__()
        if transition != '3to4':
            raise ValueError('GRERRelay supports only transition=3to4')
        if local_kernel != 3:
            raise ValueError('GRER first implementation requires local_kernel=3')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('GRER eps must be finite and positive')
        if not isinstance(debug, bool):
            raise ValueError('GRER debug must be a YAML boolean')
        if not isinstance(debug_interval, int) or debug_interval <= 0:
            raise ValueError('GRER debug_interval must be a positive integer')
        alpha_init, alpha_max, threshold, temperature = _validate_options(
            projection, fusion, rarity)
        if not all(math.isfinite(v) for v in (alpha_init, alpha_max, threshold, temperature)):
            raise ValueError('GRER scalar options must be finite')
        if alpha_max <= 0 or temperature <= 0:
            raise ValueError('GRER alpha_max and rarity.temperature must be positive')

        self.ch_in = int(ch_in)
        self.ch_out = int(ch_out)
        self.requested_groups = groups
        self.groups = resolve_groups(ch_in, groups)
        self.local_kernel = local_kernel
        self.eps = float(eps)
        self.threshold = threshold
        self.temperature = temperature
        self.alpha_max = alpha_max
        self.debug = debug
        self.debug_interval = debug_interval
        self._debug_iteration = 0
        self.raw_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.projection = ConvNormLayer(ch_in, ch_out, 1, 1, act=None)
        if self.debug:
            print(f'[GRER] requested_groups={groups}, actual_groups={self.groups}')

    @property
    def alpha_eff(self):
        return self.alpha_max * self.raw_alpha.tanh()

    @staticmethod
    def _even_pad(x):
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            x = F.pad(x, (0, width % 2, 0, height % 2), mode='replicate')
        return x

    def rarity_features(self, x, return_stats=False):
        """Return AvgPool2d(g * X) and optional differentiable statistics."""
        if x.ndim != 4 or x.shape[1] != self.ch_in or not x.is_floating_point():
            raise ValueError(f'GRER expects floating NCHW with C={self.ch_in}')
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        local_mean = F.avg_pool2d(work, self.local_kernel, stride=1,
                                  padding=self.local_kernel // 2,
                                  count_include_pad=False)
        residual = work - local_mean
        batch, channels, height, width = residual.shape
        grouped = residual.reshape(batch, self.groups, channels // self.groups,
                                   height, width)
        group_deviation = (grouped.square().mean(dim=2) + self.eps).sqrt()
        local_score = group_deviation.mean(dim=1, keepdim=True)

        flattened = local_score.flatten(2)
        median = flattened.median(dim=2, keepdim=True).values.reshape(batch, 1, 1, 1)
        absolute = (local_score - median).abs()
        mad = absolute.flatten(2).median(dim=2, keepdim=True).values.reshape(batch, 1, 1, 1)
        mad_safe = mad.clamp_min(self.eps)
        z = (local_score - median) / mad_safe
        gate = torch.sigmoid((z - self.threshold) / self.temperature)

        # The required first implementation relays g * X, never g * residual.
        rare = gate * work
        rare = F.avg_pool2d(self._even_pad(rare), kernel_size=2, stride=2)
        rare = rare.to(dtype=x.dtype)
        if return_stats:
            return rare, {'local_mean': local_mean, 'residual': residual,
                          'group_deviation': group_deviation,
                          'local_score': local_score, 'median': median,
                          'mad': mad, 'mad_safe': mad_safe, 'z': z, 'gate': gate}
        return rare

    def _print_debug(self, stats, evidence, base):
        score, z, gate = stats['local_score'], stats['z'], stats['gate']
        evidence_norm = evidence.float().norm()
        base_norm = (base.float().norm().clamp_min(self.eps)
                     if base is not None else evidence_norm.new_tensor(float('nan')))
        values = {
            'local_score_mean': score.mean(), 'median': stats['median'].mean(),
            'MAD': stats['mad'].mean(), 'z_mean': z.mean(),
            'z_p90': torch.quantile(z, .90), 'z_p99': torch.quantile(z, .99),
            'gate_mean': gate.mean(), 'gate_std': gate.std(unbiased=False),
            'gate_p50': torch.quantile(gate, .50),
            'gate_p90': torch.quantile(gate, .90),
            'gate_p99': torch.quantile(gate, .99), 'gate_max': gate.max(),
            'evidence_norm': evidence_norm, 'alpha_effective': self.alpha_eff,
            'evidence/base': evidence_norm / base_norm,
        }
        rendered = ' '.join(f'{key}={float(value.detach()):.6g}'
                            for key, value in values.items())
        print(f'[GRER iter={self._debug_iteration}] {rendered}')

    def forward(self, x, base=None):
        debug_now = self.debug and self._debug_iteration % self.debug_interval == 0
        if debug_now:
            rare, stats = self.rarity_features(x, return_stats=True)
        else:
            rare, stats = self.rarity_features(x), None
        evidence = self.projection(rare)
        if debug_now:
            self._print_debug(stats, evidence, base)
        self._debug_iteration += 1
        return self.alpha_eff.to(dtype=evidence.dtype) * evidence
