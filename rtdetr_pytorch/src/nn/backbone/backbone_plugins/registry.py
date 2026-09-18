"""Single-plugin validation and factory. Uses the existing RT-DETR registry."""

from .srfd import SRFDPlugin
from .deconv import DEConvPlugin
from .drb import DRBPlugin
from .akconv import AKConvPlugin
from .rfaconv import RFAConvPlugin


BACKBONE_PLUGIN_REGISTRY = {
    'srfd': SRFDPlugin, 'deconv': DEConvPlugin, 'drb': DRBPlugin,
    'akconv': AKConvPlugin, 'rfaconv': RFAConvPlugin,
}

UAV_STAGES = {'deconv': ['S3', 'S4'], 'drb': ['S4'],
              'akconv': ['S3', 'S4'], 'rfaconv': ['S3', 'S4']}


def resolve_plugin(config, num_stages=4):
    if config is None:
        return {'enabled': False, 'name': 'none'}
    if not isinstance(config, dict):
        raise ValueError('BackbonePlugin must be ONE mapping, not a list of plugins')
    unknown = set(config) - {'enabled', 'name', 'placement', 'stages', 'conv', 'params'}
    if unknown:
        raise ValueError(f'Unknown BackbonePlugin fields: {sorted(unknown)}')
    if not isinstance(config.get('enabled', False), bool):
        raise ValueError('BackbonePlugin.enabled must be boolean')
    if not config.get('enabled', False):
        return {'enabled': False, 'name': 'none'}
    name = config.get('name')
    if not isinstance(name, str) or name not in BACKBONE_PLUGIN_REGISTRY:
        raise ValueError('Enable exactly ONE plugin: srfd/deconv/drb/akconv/rfaconv')
    placement = config.get('placement', 'uav_placement')
    if placement not in ('source_placement', 'uav_placement'):
        raise ValueError('placement must be source_placement or uav_placement')
    defaults = (['Stem'] if name == 'srfd' else
                ['S2', 'S3', 'S4', 'S5'] if placement == 'source_placement' else UAV_STAGES[name])
    stages = config.get('stages', defaults)
    if (not isinstance(stages, (list, tuple)) or not stages
            or not all(isinstance(stage, str) for stage in stages)
            or len(set(stages)) != len(stages)):
        raise ValueError('stages must be a nonempty list of distinct stage names')
    allowed = ['Stem'] if name == 'srfd' else [f'S{i + 2}' for i in range(num_stages)]
    if any(stage not in allowed for stage in stages):
        raise ValueError(f'{name} stages must be drawn from {allowed}')
    conv = config.get('conv', 'both' if placement == 'source_placement'
                      and name in ('akconv', 'rfaconv') else 'conv2')
    if conv not in ('conv1', 'conv2', 'both'):
        raise ValueError('conv must be conv1/conv2/both')
    if name == 'drb' and conv != 'conv2':
        raise ValueError('Source DRB is stride-1 depthwise: use conv2, not downsampling conv1')
    if name == 'srfd' and 'conv' in config:
        raise ValueError('SRFD replaces Stem, not block conv1/conv2')
    params = config.get('params', {})
    if not isinstance(params, dict):
        raise ValueError('plugin params must be a mapping')
    allowed_params = {'srfd': set(), 'deconv': set(), 'drb': {'kernel_size'},
                      'akconv': {'num_param'}, 'rfaconv': {'kernel_size'}}[name]
    if set(params) - allowed_params:
        raise ValueError(f'{name} params may only contain {sorted(allowed_params)}')
    return {'enabled': True, 'name': name, 'placement': placement,
            'stages': list(stages), 'conv': conv, 'params': dict(params)}
