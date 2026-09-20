"""Residual side branches, NOT replacements for original PResNet stages."""

import math
import warnings

import torch
from torch import nn

from ..secd import SECDTransition


TYPES = {'P0': ('srfd',), 'P1': ('deconv',),
         'P2': ('uav_dcnv4', 'dcnv4'),
         'P3': ('secd',), 'P4': ('fadc', 'wtconv')}


class ResidualPlugin(nn.Module):
    """Return a scaled branch; PResNet adds it to the original feature.

    The branch is deliberately executed even at raw_alpha=0: skipping it
    would prevent the gate from receiving its first gradient.
    """

    def __init__(self, branch, alpha_init=0.0, alpha_max=0.20):
        super().__init__()
        if (not math.isfinite(alpha_init) or not math.isfinite(alpha_max)
                or alpha_max <= 0):
            raise ValueError('Plugin alpha_init must be finite and alpha_max positive')
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = float(alpha_max)
        self.branch = branch

    @property
    def alpha_eff(self):
        return self.alpha_max * self.raw_alpha.tanh()

    def forward(self, x):
        evidence = self.branch(x)
        # Keep the surrounding feature dtype under AMP. Scalar gates stay FP32
        # parameters, but never promote a whole activation to FP32.
        return self.alpha_eff.to(dtype=evidence.dtype) * evidence


def build_plugins(config, depth, variant, num_stages):
    if not isinstance(config, dict):
        raise ValueError('BackbonePlugins must be a mapping of P0..P4')
    unknown = set(config) - set(TYPES)
    if unknown:
        raise ValueError(f'Unknown BackbonePlugins points: {sorted(unknown)}')
    active = {}
    for point, options in config.items():
        if not isinstance(options, dict):
            raise ValueError(f'{point} must be a mapping; P4 accepts exactly one type')
        if not isinstance(options.get('enabled', False), bool):
            raise ValueError(f'{point}.enabled must be a YAML boolean')
        kind = options.get('type', TYPES[point][0])
        if kind not in TYPES[point]:
            raise ValueError(f'{point}.type must be one of {TYPES[point]}, got {kind!r}')
        if options.get('enabled', False):
            active[point] = dict(options)
    if not active:
        return None
    if depth != 18 or variant != 'd' or num_stages != 4:
        raise ValueError('First-round BackbonePlugins require PResNet18 variant=d, four stages')
    if len(active) > 1:
        warnings.warn('Backbone plugin combination enabled. Complete single-point '
                      'screening first; prioritize complementary functions and at '
                      'most three points in the first combination round.', UserWarning)
    plugins = nn.ModuleDict()
    # Fork RNG keeps every existing encoder/decoder initialization identical to
    # Baseline for a fixed seed; only side branches acquire additional weights.
    with torch.random.fork_rng(devices=[]):
        for point in TYPES:
            if point not in active:
                continue
            options = active[point]
            kind = options.pop('type', TYPES[point][0])
            options.pop('enabled')
            alpha = {name: options.pop(name) for name in ('alpha_init', 'alpha_max')
                     if name in options}
            if point == 'P3':
                plugins['p3'] = SECDTransition(128, 256, **alpha, **options)
                continue
            if kind == 'srfd':
                from ..backbone_plugins.srfd import SRFDPlugin
                if options:
                    raise ValueError(f'P0-SRFD unsupported options: {sorted(options)}')
                branch = SRFDPlugin(3, 64)
            elif kind == 'deconv':
                from ..backbone_plugins.deconv import DEConvPlugin
                if options:
                    raise ValueError(f'P1-DEConv unsupported options: {sorted(options)}')
                # Restore source DEConv's own terminal BN/SiLU. Original stage
                # BN/ReLU objects are untouched, unlike the old replacement.
                branch = nn.Sequential(DEConvPlugin(64, 64), nn.BatchNorm2d(64), nn.SiLU())
            elif kind in ('uav_dcnv4', 'dcnv4'):
                from .dcnv4 import UAVDCNv4
                branch = UAVDCNv4(128, **options)
            elif kind == 'fadc':
                from .fadc import FADC
                branch = FADC(256, **options)
            else:
                from .wtconv import WTConv2d
                branch = WTConv2d(256, **options)
            plugins[point.lower()] = ResidualPlugin(branch, **alpha)
    return plugins
