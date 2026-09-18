"""Opt-in PResNet18 adapter, leaving the original PResNet/BasicBlock intact.

The existing __share__ mechanism supplies the inherited PResNet mapping;
__share__ does NOT instantiate it. Ordinary YAMLs still select PResNet and
cannot retain a plugin setting from a preceding experiment in the same process.
"""

import inspect
import torch

from src.core import register
from ..presnet import PResNet as OriginalPResNet
from .registry import BACKBONE_PLUGIN_REGISTRY, resolve_plugin


@register
class PResNetWithPlugin(OriginalPResNet):
    __share__ = ['PResNet', 'BackbonePlugin', 'SECD', 'MERT']

    def __init__(self, PResNet=None, BackbonePlugin=None, SECD=None, MERT=None):
        # Filter the stock registry's private schema metadata, not its settings.
        names = inspect.signature(OriginalPResNet.__init__).parameters
        options = {key: value for key, value in (PResNet or {}).items()
                   if key in names and key not in ('self', 'SECD')}
        if options.get('depth') is None:
            raise ValueError('PResNetWithPlugin requires an inherited PResNet.depth')
        plugin = resolve_plugin(BackbonePlugin, options.get('num_stages', 4))
        if plugin['enabled']:
            if options['depth'] != 18:
                raise ValueError('First-round BackbonePlugin experiments support PResNet18 only')
            if (SECD or {}).get('enabled', False) or (MERT or {}).get('enabled', False):
                raise ValueError('Pure BackbonePlugin screening requires MERT=false and SECD=false')
        # Stock pretrained loading is still STRICT, before replacing any conv.
        # Thus every unaffected stage/norm/shortcut uses exactly its old weights.
        super().__init__(**options, SECD=SECD)
        self.plugin_config = plugin
        self.plugin_pretrained_report = {
            'original_pretrained_loaded': bool(options.get('pretrained', False)),
            'original_strict_missing_keys': 0, 'original_strict_unexpected_keys': 0,
            'replacements': [],
        }
        if not plugin['enabled']:
            return
        # Plugin creation must not shift initialization of the existing encoder
        # and decoder. It remains deterministic under the same CLI seed.
        with torch.random.fork_rng(devices=[]):
            if plugin['name'] == 'srfd':
                self.conv1 = BACKBONE_PLUGIN_REGISTRY['srfd'](3, 64)
                self.plugin_pretrained_report['replacements'].append({
                    'path': 'conv1 + functional max_pool2d',
                    'initialization': 'new SRFD; original stem discarded; all S2-S5 weights retained'})
            else:
                branches = ('branch2a', 'branch2b') if plugin['conv'] == 'both' else (
                    'branch2a' if plugin['conv'] == 'conv1' else 'branch2b',)
                for stage_name in plugin['stages']:
                    stage_idx = int(stage_name[1:]) - 2
                    for block_idx, block in enumerate(self.res_layers[stage_idx].blocks):
                        for branch_name in branches:
                            layer = getattr(block, branch_name)
                            old_conv = layer.conv
                            operator = BACKBONE_PLUGIN_REGISTRY[plugin['name']](
                                old_conv.in_channels, old_conv.out_channels,
                                stride=old_conv.stride[0], **plugin['params'])
                            init = (operator.init_from_conv(old_conv)
                                    if options.get('pretrained', False)
                                    else 'new source-style initialization; pretrained disabled')
                            # Keep the original ConvNormLayer, norm, act and
                            # both shortcut types; replace ONLY its spatial conv.
                            layer.conv = operator
                            self.plugin_pretrained_report['replacements'].append({
                                'path': f'res_layers.{stage_idx}.blocks.{block_idx}.{branch_name}.conv',
                                'initialization': init})
        if options.get('freeze_norm', True):
            self._freeze_norm(self)
        freeze_at = options.get('freeze_at', -1)
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for index in range(min(freeze_at, len(self.res_layers))):
                self._freeze_parameters(self.res_layers[index])

    def forward(self, x):
        if not self.plugin_config['enabled'] or self.plugin_config['name'] != 'srfd':
            return super().forward(x)
        # SRFD already performs 4x downsampling; a second max pool would break
        # every feature stride. This special path touches the Backbone only.
        x = self.conv1(x)
        outs = []
        for index, stage in enumerate(self.res_layers):
            x = stage(x)
            if index in self.return_idx:
                outs.append(x)
        return outs
