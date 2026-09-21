"""Cross-Channel Consensus Evidence Downsampling for the S3 -> S4 bypass."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..common import ConvNormLayer


def resolve_groups(channels, requested):
    """Select the largest preferred group count <= requested that divides C."""
    if not isinstance(requested, int) or requested <= 0:
        raise ValueError('CCED groups must be a positive integer')
    for groups in (8, 4, 2, 1):
        if groups <= requested and channels % groups == 0:
            return groups
    raise ValueError(f'CCED cannot group {channels} channels with request {requested}')


def _validate_options(projection, fusion):
    projection = {} if projection is None else dict(projection)
    fusion = {} if fusion is None else dict(fusion)
    unknown_projection = set(projection) - {'norm', 'activation'}
    unknown_fusion = set(fusion) - {'alpha_init', 'alpha_max'}
    if unknown_projection or unknown_fusion:
        raise ValueError('CCED unsupported options: '
                         f'projection={sorted(unknown_projection)}, '
                         f'fusion={sorted(unknown_fusion)}')
    if projection.get('norm', True) is not True:
        raise ValueError('CCED projection.norm must be true in the first implementation')
    if projection.get('activation', False) not in (False, None):
        raise ValueError('CCED projection.activation must be false')
    return float(fusion.get('alpha_init', 0.0)), float(fusion.get('alpha_max', 0.20))


class CCEDTransition(nn.Module):
    """Parallel stride-8 -> stride-16 consensus-evidence residual branch."""

    def __init__(self, ch_in, ch_out, transition='3to4', groups=8, eps=1e-6,
                 projection=None, fusion=None, debug=False, debug_interval=100):
        super().__init__()
        if transition != '3to4':
            raise ValueError('CCEDTransition supports only transition=3to4')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('CCED eps must be finite and positive')
        if not isinstance(debug, bool):
            raise ValueError('CCED debug must be a YAML boolean')
        if not isinstance(debug_interval, int) or debug_interval <= 0:
            raise ValueError('CCED debug_interval must be a positive integer')
        alpha_init, alpha_max = _validate_options(projection, fusion)
        if not math.isfinite(alpha_init) or not math.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError('CCED alpha_init must be finite and alpha_max positive')

        self.ch_in = int(ch_in)
        self.ch_out = int(ch_out)
        self.requested_groups = groups
        self.groups = resolve_groups(ch_in, groups)
        self.eps = float(eps)
        self.alpha_max = alpha_max
        self.debug = debug
        self.debug_interval = debug_interval
        self._debug_iteration = 0
        self.raw_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.projection = ConvNormLayer(ch_in, ch_out, 1, 1, act=None)
        if self.debug:
            print(f'[CCED] requested_groups={groups}, actual_groups={self.groups}')

    @property
    def alpha_eff(self):
        return self.alpha_max * self.raw_alpha.tanh()

    @staticmethod
    def _even_pad(x):
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            x = F.pad(x, (0, width % 2, 0, height % 2), mode='replicate')
        return x

    def consensus_evidence(self, x, return_stats=False):
        """Return unprojected D and optional FP32 diagnostic tensors.

        Shapes are phases [B,C,4,h,w], grouped residual
        [B,G,C/G,4,h,w], q [B,G,4,h,w], and consensus [B,4,h,w].
        No softmax is used.
        """
        if x.ndim != 4 or x.shape[1] != self.ch_in or not x.is_floating_point():
            raise ValueError(f'CCED expects floating NCHW with C={self.ch_in}')
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        work = self._even_pad(work)
        phases = torch.stack((work[:, :, 0::2, 0::2],
                              work[:, :, 0::2, 1::2],
                              work[:, :, 1::2, 0::2],
                              work[:, :, 1::2, 1::2]), dim=2)
        residual = phases - phases.mean(dim=2, keepdim=True)
        batch, channels, _, height, width = residual.shape
        grouped = residual.reshape(batch, self.groups, channels // self.groups,
                                   4, height, width)
        energy = (grouped.square().mean(dim=2) + self.eps).sqrt()

        # Ordinary proportional normalization (explicitly not Softmax).
        q = energy / (energy.sum(dim=2, keepdim=True) + self.eps)
        q = q / q.sum(dim=2, keepdim=True).clamp_min(self.eps)
        consensus = (q + self.eps).log().mean(dim=1).exp()
        consensus = consensus / (consensus.sum(dim=1, keepdim=True) + self.eps)
        consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(self.eps)
        dense = (consensus.unsqueeze(1) * residual).sum(dim=2)
        dense = dense.to(dtype=x.dtype)
        if return_stats:
            return dense, {'phases': phases, 'grouped_residual': grouped,
                           'energy': energy, 'q': q, 'consensus': consensus}
        return dense

    def _print_debug(self, stats, evidence, base):
        q, consensus = stats['q'], stats['consensus']
        entropy = -(consensus * (consensus + self.eps).log()).sum(dim=1).mean()
        evidence_norm = evidence.float().norm()
        base_norm = (base.float().norm().clamp_min(self.eps)
                     if base is not None else evidence_norm.new_tensor(float('nan')))
        values = {
            'q_mean': q.mean(), 'q_max': q.max(),
            'consensus_mean': consensus.mean(), 'consensus_max': consensus.max(),
            'consensus_entropy': entropy, 'evidence_norm': evidence_norm,
            'alpha_effective': self.alpha_eff,
            'evidence/base': evidence_norm / base_norm,
        }
        rendered = ' '.join(f'{key}={float(value.detach()):.6g}'
                            for key, value in values.items())
        print(f'[CCED iter={self._debug_iteration}] {rendered}')

    def forward(self, x, base=None):
        debug_now = self.debug and self._debug_iteration % self.debug_interval == 0
        if debug_now:
            dense, stats = self.consensus_evidence(x, return_stats=True)
        else:
            dense, stats = self.consensus_evidence(x), None
        evidence = self.projection(dense)
        if debug_now:
            self._print_debug(stats, evidence, base)
        self._debug_iteration += 1
        return self.alpha_eff.to(dtype=evidence.dtype) * evidence
