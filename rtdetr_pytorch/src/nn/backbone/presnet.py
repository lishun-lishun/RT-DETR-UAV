'''by lyuwenyu
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d
from .secd import SECDTransition

from src.core import register


__all__ = ['PResNet']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register
class PResNet(nn.Module):
    __share__ = ['SECD', 'BackbonePlugins', 'CCED', 'GRER',
                 'BackboneVariant', 'PHSB', 'BackboneEnhancement', 'BAFR', 'HCBR',
                 'BackboneModification', 'BDPD', 'MSDConv']

    def __init__(
        self, 
        depth, 
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False,
        SECD=None,
        BackbonePlugins=None,
        CCED=None,
        GRER=None,
        BackboneVariant=None,
        PHSB=None,
        BackboneEnhancement=None,
        BAFR=None,
        HCBR=None,
        BackboneModification=None,
        BDPD=None,
        MSDConv=None):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        modification = ({} if BackboneModification is None
                        else dict(BackboneModification))
        unknown_modification = set(modification) - {'bpdp', 'msdconv'}
        if unknown_modification:
            raise ValueError('Unknown BackboneModification options: '
                             f'{sorted(unknown_modification)}')
        self.bpdp_enabled = modification.get('bpdp', False)
        self.msdconv_enabled = modification.get('msdconv', False)
        if (not isinstance(self.bpdp_enabled, bool)
                or not isinstance(self.msdconv_enabled, bool)):
            raise ValueError('BackboneModification.bpdp/msdconv must be YAML booleans')
        variant_cfg = {} if BackboneVariant is None else dict(BackboneVariant)
        self.backbone_variant_type = variant_cfg.get('type', 'baseline')
        if self.backbone_variant_type not in ('baseline', 'hsdr', 'phsb'):
            raise ValueError('BackboneVariant.type must be baseline, hsdr, or phsb')
        self.backbone_variant_debug = variant_cfg.get('debug', False)
        self.backbone_variant_debug_interval = variant_cfg.get('debug_interval', 100)
        if (not isinstance(self.backbone_variant_debug, bool)
                or not isinstance(self.backbone_variant_debug_interval, int)
                or self.backbone_variant_debug_interval < 1):
            raise ValueError('BackboneVariant debug must be boolean and interval positive')
        self._variant_debug_iteration = 0
        enhancement = {} if BackboneEnhancement is None else dict(BackboneEnhancement)
        unknown = set(enhancement) - {'bafr', 'hcbr'}
        if unknown:
            raise ValueError(f'Unknown BackboneEnhancement options: {sorted(unknown)}')
        self.bafr_enabled = enhancement.get('bafr', False)
        self.hcbr_enabled = enhancement.get('hcbr', False)
        if not isinstance(self.bafr_enabled, bool) or not isinstance(self.hcbr_enabled, bool):
            raise ValueError('BackboneEnhancement.bafr/hcbr must be YAML booleans')
        if self.bafr_enabled or self.hcbr_enabled:
            if depth != 18 or variant != 'd' or num_stages != 4:
                raise ValueError('BAFR/HCBR first round requires PResNet18-d with four stages')
            if self.backbone_variant_type != 'baseline':
                raise ValueError('BAFR/HCBR cannot mix HSDR/PHSB in the first round')
            old_methods = (
                (SECD or {}).get('enabled', False),
                (CCED or {}).get('enabled', False),
                (GRER or {}).get('enabled', False),
                any(isinstance(options, dict) and options.get('enabled', False)
                    for options in (BackbonePlugins or {}).values()),
            )
            if any(old_methods):
                raise ValueError('BAFR/HCBR cannot mix legacy backbone plugins in the first round')
        if self.backbone_variant_type != 'baseline':
            if depth != 18 or variant != 'd' or num_stages != 4:
                raise ValueError('HSDR/PHSB first round requires PResNet18-d with four stages')
            old_methods = (
                (SECD or {}).get('enabled', False),
                (CCED or {}).get('enabled', False),
                (GRER or {}).get('enabled', False),
                any(isinstance(options, dict) and options.get('enabled', False)
                    for options in (BackbonePlugins or {}).values()),
            )
            if any(old_methods):
                raise ValueError('HSDR/PHSB first round cannot mix existing backbone plugins')
        self.stage_blocks = list(block_nums[:num_stages])
        if self.backbone_variant_type == 'hsdr':
            requested = variant_cfg.get('stage_blocks')
            if requested not in ([2, 3, 2, 1], [2, 4, 3, 1]):
                raise ValueError('First-round HSDR stage_blocks must be [2,3,2,1] or [2,4,3,1]')
            self.stage_blocks = list(requested)
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        # BDPD v1 replaces exactly the main-path stride-2 ConvNormLayer in the
        # first S3 BasicBlock (effective stride 4 -> 8). The original VD
        # AvgPool+1x1 shortcut and the rest of the BasicBlock remain untouched.
        # MSDConv v1 refines the complete S3 output before it is returned as P3
        # and before S4 consumes it. Forked RNG keeps later model initialization
        # identical to Baseline for a fixed seed.
        self.msdconv_p3 = None
        if self.bpdp_enabled or self.msdconv_enabled:
            if depth != 18 or variant != 'd' or num_stages != 4:
                raise ValueError('BDPD/MSDConv v1 requires PResNet18-d with four stages')
            enabled_legacy = (
                self.backbone_variant_type != 'baseline',
                self.bafr_enabled,
                self.hcbr_enabled,
                (SECD or {}).get('enabled', False),
                (CCED or {}).get('enabled', False),
                (GRER or {}).get('enabled', False),
                any(isinstance(options, dict) and options.get('enabled', False)
                    for options in (BackbonePlugins or {}).values()),
            )
            if any(enabled_legacy):
                raise ValueError('BDPD/MSDConv first round cannot mix existing backbone methods')
            with torch.random.fork_rng(devices=[]):
                if self.bpdp_enabled:
                    from .backbone_modules.bdpd import BDPDDownsample
                    transition = self.res_layers[1].blocks[0]
                    old = transition.branch2a
                    if (not isinstance(transition, BasicBlock)
                            or old.conv.in_channels != 64
                            or old.conv.out_channels != 128
                            or old.conv.kernel_size != (3, 3)
                            or old.conv.stride != (2, 2)):
                        raise RuntimeError('BDPD expected the audited S3 block-0 64->128 '
                                           '3x3 stride-2 main-path convolution')
                    options = {} if BDPD is None else dict(BDPD)
                    position = options.pop('position', 'stride4_to_8')
                    if position != 'stride4_to_8':
                        raise ValueError('BDPD v1 position must be stride4_to_8')
                    transition.branch2a = BDPDDownsample(64, 128, act=act, **options)
                if self.msdconv_enabled:
                    from .backbone_modules.msdconv import MSDConv as MSDConvModule
                    options = {} if MSDConv is None else dict(MSDConv)
                    position = options.pop('position', 'p3')
                    if position != 'p3':
                        raise ValueError('MSDConv v1 position must be p3')
                    self.msdconv_p3 = MSDConvModule(_out_channels[1], **options)

        if self.backbone_variant_type == 'hsdr':
            # Build the original eight blocks first, then alter only counts in
            # a forked RNG stream. Existing weights/key names and the later
            # Encoder/Decoder initialization remain aligned to Baseline.
            with torch.random.fork_rng(devices=[]):
                for idx, target_count in enumerate(self.stage_blocks):
                    blocks = self.res_layers[idx].blocks
                    for _ in range(target_count - len(blocks)):
                        blocks.append(BasicBlock(_out_channels[idx], _out_channels[idx],
                                                 stride=1, shortcut=True,
                                                 variant=variant, act=act))
                    while len(blocks) > target_count:
                        del blocks[-1]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        # S3 has two original BasicBlocks. Replace only its last spatial
        # modelling block; the shortcut, later convolution and key hierarchy
        # remain the original block's. New weights use an isolated RNG stream.
        if self.bafr_enabled:
            from .backbone_modules.bafr import BAFRBlock
            with torch.random.fork_rng(devices=[]):
                blocks = self.res_layers[1].blocks
                if len(blocks) != 2 or not isinstance(blocks[-1], BasicBlock):
                    raise RuntimeError('BAFR requires original two-block S3 BasicBlock stage')
                blocks[-1] = BAFRBlock(blocks[-1], **({} if BAFR is None else dict(BAFR)))

        # Separate bypass attributes preserve every original res_layers.* key.
        # Disabled SECD creates no module/parameter and consumes no RNG state.
        self.secd_34 = None
        self.secd_45 = None
        secd_cfg = {} if SECD is None else dict(SECD)
        if secd_cfg.get('enabled', False):
            transitions = secd_cfg.get('transitions', ['3to4'])
            if not isinstance(transitions, (list, tuple)) or not transitions:
                raise ValueError('SECD transitions must be a nonempty list')
            if len(set(transitions)) != len(transitions):
                raise ValueError('SECD transitions must not contain duplicates')
            options = {k: v for k, v in secd_cfg.items()
                       if k not in ('enabled', 'transitions')}
            # Keep subsequent encoder/decoder initialization on the original
            # seeded RNG stream while deterministically initializing bypasses.
            with torch.random.fork_rng(devices=[]):
                for transition in transitions:
                    if transition not in ('3to4', '4to5'):
                        raise ValueError(f'Unknown SECD transition: {transition}')
                    target_idx = 2 if transition == '3to4' else 3
                    if target_idx >= num_stages:
                        raise ValueError(f'SECD {transition} requires stage S{target_idx + 2}')
                    bypass = SECDTransition(_out_channels[target_idx - 1],
                                            _out_channels[target_idx], **options)
                    setattr(self, 'secd_34' if target_idx == 2 else 'secd_45', bypass)

        self.plugins = None
        if BackbonePlugins is not None:
            from .plugin_points import build_plugins
            self.plugins = build_plugins(BackbonePlugins, depth, variant, num_stages)
            if self.plugins is not None and (self.secd_34 is not None or self.secd_45 is not None):
                raise ValueError('Use P3 for new plugin-point SECD; do not mix legacy SECD and BackbonePlugins')

        # CCED/GRER are parallel S3 -> S4 evidence branches. They never wrap or
        # replace res_layers, so every original stage key remains unchanged.
        self.cced_34 = None
        self.grer_34 = None
        cced_cfg = {} if CCED is None else dict(CCED)
        grer_cfg = {} if GRER is None else dict(GRER)
        for name, config in (('CCED', cced_cfg), ('GRER', grer_cfg)):
            if not isinstance(config.get('enabled', False), bool):
                raise ValueError(f'{name}.enabled must be a YAML boolean')
        cced_enabled = cced_cfg.pop('enabled', False)
        grer_enabled = grer_cfg.pop('enabled', False)
        if cced_enabled or grer_enabled:
            if depth != 18 or variant != 'd' or num_stages != 4:
                raise ValueError('CCED/GRER first implementation requires PResNet18-d with four stages')
            if self.secd_34 is not None or self.secd_45 is not None or self.plugins is not None:
                raise ValueError('CCED/GRER screening cannot mix legacy SECD or BackbonePlugins')
            # Do not shift initialization of HybridEncoder/decoder for a fixed
            # seed; only the new bypasses consume this forked RNG stream.
            with torch.random.fork_rng(devices=[]):
                if cced_enabled:
                    from .backbone_plugins.cced import CCEDTransition
                    self.cced_34 = CCEDTransition(_out_channels[1], _out_channels[2],
                                                  **cced_cfg)
                if grer_enabled:
                    from .backbone_plugins.grer import GRERRelay
                    self.grer_34 = GRERRelay(_out_channels[1], _out_channels[2],
                                             **grer_cfg)

        self.phsb = None
        if self.backbone_variant_type == 'phsb':
            from .phsb import PHSBBranch
            phsb_cfg = {} if PHSB is None else dict(PHSB)
            with torch.random.fork_rng(devices=[]):
                self.phsb = PHSBBranch(_out_channels[1], _out_channels[2],
                                       BasicBlock, **phsb_cfg)

        self.hcbr_p3 = None
        self.hcbr_p4 = None
        if self.hcbr_enabled:
            from .backbone_modules.hcbr import HCBR as HCBRModule
            hcbr_cfg = {} if HCBR is None else dict(HCBR)
            use_p3 = hcbr_cfg.pop('use_p3', True)
            use_p4 = hcbr_cfg.pop('use_p4', True)
            if not isinstance(use_p3, bool) or not isinstance(use_p4, bool):
                raise ValueError('HCBR.use_p3/use_p4 must be YAML booleans')
            if not (use_p3 or use_p4):
                raise ValueError('HCBR requires at least one of use_p3/use_p4')
            with torch.random.fork_rng(devices=[]):
                if use_p3:
                    self.hcbr_p3 = HCBRModule(_out_channels[1], **hcbr_cfg)
                    self.hcbr_p3.debug_name = 'P3'
                if use_p4:
                    self.hcbr_p4 = HCBRModule(_out_channels[2], **hcbr_cfg)
                    self.hcbr_p4.debug_name = 'P4'

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])
            if self.secd_34 is not None and freeze_at > 2:
                self._freeze_parameters(self.secd_34)
            if self.secd_45 is not None and freeze_at > 3:
                self._freeze_parameters(self.secd_45)
            if self.cced_34 is not None and freeze_at > 2:
                self._freeze_parameters(self.cced_34)
            if self.grer_34 is not None and freeze_at > 2:
                self._freeze_parameters(self.grer_34)
            if self.phsb is not None and freeze_at > 1:
                self._freeze_parameters(self.phsb)
            if self.hcbr_p3 is not None and freeze_at > 1:
                self._freeze_parameters(self.hcbr_p3)
            if self.hcbr_p4 is not None and freeze_at > 2:
                self._freeze_parameters(self.hcbr_p4)
            if self.msdconv_p3 is not None and freeze_at > 1:
                self._freeze_parameters(self.msdconv_p3)
            if self.plugins is not None:
                for point, plugin in self.plugins.items():
                    stage_idx = {'p0': -1, 'p1': 0, 'p2': 1, 'p3': 2, 'p4': 2}[point]
                    if stage_idx < freeze_at:
                        self._freeze_parameters(plugin)

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            if self.backbone_variant_type == 'hsdr':
                incompatible = self.load_state_dict(state, strict=False)
                extra = {(idx, block_idx)
                         for idx, target_count in enumerate(self.stage_blocks)
                         for block_idx in range(block_nums[idx], target_count)}
                removed = {(idx, block_idx)
                           for idx, target_count in enumerate(self.stage_blocks)
                           for block_idx in range(target_count, block_nums[idx])}
                def belongs_to(key, stage_blocks):
                    return any(key.startswith(f'res_layers.{idx}.blocks.{block_idx}.')
                               for idx, block_idx in stage_blocks)
                invalid_missing = [key for key in incompatible.missing_keys
                                   if not belongs_to(key, extra)]
                invalid_unexpected = [key for key in incompatible.unexpected_keys
                                      if not belongs_to(key, removed)]
                if invalid_missing or invalid_unexpected:
                    raise RuntimeError('HSDR pretrained backbone keys mismatch: '
                                       f'missing={invalid_missing}, unexpected={invalid_unexpected}')
            elif self.phsb is not None:
                incompatible = self.load_state_dict(state, strict=False)
                invalid_missing = [key for key in incompatible.missing_keys
                                   if not key.startswith('phsb.')]
                if invalid_missing or incompatible.unexpected_keys:
                    raise RuntimeError('PHSB pretrained backbone keys mismatch: '
                                       f'missing={invalid_missing}, '
                                       f'unexpected={incompatible.unexpected_keys}')
            elif self.bpdp_enabled or self.msdconv_enabled:
                incompatible = self.load_state_dict(state, strict=False)
                replaced_prefix = 'res_layers.1.blocks.0.branch2a.'
                allowed_missing = []
                if self.bpdp_enabled:
                    allowed_missing.append(replaced_prefix)
                if self.msdconv_enabled:
                    allowed_missing.append('msdconv_p3.')
                invalid_missing = [key for key in incompatible.missing_keys
                                   if not key.startswith(tuple(allowed_missing))]
                invalid_unexpected = [key for key in incompatible.unexpected_keys
                                      if not (self.bpdp_enabled
                                              and key.startswith(replaced_prefix))]
                if invalid_missing or invalid_unexpected:
                    raise RuntimeError('BDPD/MSDConv pretrained backbone keys mismatch: '
                                       f'missing={invalid_missing}, '
                                       f'unexpected={invalid_unexpected}')
                self.replaced_pretrained_keys = sorted(
                    key for key in state if self.bpdp_enabled
                    and key.startswith(replaced_prefix))
            elif self.bafr_enabled or self.hcbr_enabled:
                incompatible = self.load_state_dict(state, strict=False)
                bafr_prefix = 'res_layers.1.blocks.1.branch2a.conv.'
                allowed_missing = ('hcbr_p3.', 'hcbr_p4.')
                invalid_missing = [key for key in incompatible.missing_keys
                                   if not key.startswith(allowed_missing)
                                   and not (self.bafr_enabled and key.startswith(bafr_prefix))]
                allowed_unexpected = {bafr_prefix + 'weight'} if self.bafr_enabled else set()
                invalid_unexpected = [key for key in incompatible.unexpected_keys
                                      if key not in allowed_unexpected]
                if invalid_missing or invalid_unexpected:
                    raise RuntimeError('BAFR/HCBR pretrained backbone keys mismatch: '
                                       f'missing={invalid_missing}, '
                                       f'unexpected={invalid_unexpected}')
            elif (self.secd_34 is None and self.secd_45 is None and self.plugins is None
                    and self.cced_34 is None and self.grer_34 is None):
                self.load_state_dict(state)
                incompatible = None
            else:
                incompatible = self.load_state_dict(state, strict=False)
                missing = [k for k in incompatible.missing_keys
                           if not k.startswith(('secd_34.', 'secd_45.', 'plugins.',
                                                'cced_34.', 'grer_34.'))]
                if missing or incompatible.unexpected_keys:
                    raise RuntimeError('PResNet pretrained backbone keys mismatch: '
                                       f'missing={missing}, '
                                       f'unexpected={incompatible.unexpected_keys}')
            missing_keys = [] if incompatible is None else list(incompatible.missing_keys)
            unexpected_keys = [] if incompatible is None else list(incompatible.unexpected_keys)
            loaded_keys = sorted(set(self.state_dict()).intersection(state))
            self.pretrained_load_report = {
                'loaded_keys': loaded_keys,
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
            }
            if (self.backbone_variant_type != 'baseline' or self.bafr_enabled
                    or self.hcbr_enabled or self.bpdp_enabled or self.msdconv_enabled):
                enhancement_name = '+'.join(name for name, enabled in
                                             (('BAFR', self.bafr_enabled),
                                              ('HCBR', self.hcbr_enabled),
                                              ('BDPD', self.bpdp_enabled),
                                              ('MSDConv', self.msdconv_enabled)) if enabled)
                method_name = enhancement_name or self.backbone_variant_type.upper()
                print(f'{method_name} pretrained: '
                      f'loaded={len(loaded_keys)}, missing={missing_keys}, '
                      f'unexpected={unexpected_keys}')
            print(f'Load PResNet{depth} state_dict')
            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        image = x
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        if self.plugins is not None:
            return self._forward_plugins(x, image)
        if self.cced_34 is not None or self.grer_34 is not None:
            return self._forward_cced_grer(x)
        if self.phsb is not None:
            return self._forward_phsb(x)
        if self.secd_34 is not None or self.secd_45 is not None:
            return self._forward_secd(x)
        if self.hcbr_p3 is not None or self.hcbr_p4 is not None:
            return self._forward_hcbr(x)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == 1 and self.msdconv_p3 is not None:
                x = self.msdconv_p3(x)
            if idx in self.return_idx:
                outs.append(x)
        if self.backbone_variant_debug:
            self._debug_variant_features(outs)
        return outs

    def _forward_hcbr(self, x):
        outs = []
        for idx, stage in enumerate(self.res_layers):
            if idx == 2 and self.hcbr_p4 is not None:
                # The stride-16 first block runs before HCBR-P4; remaining
                # S4 blocks consume its output. This is not an output-only hook.
                x = stage.blocks[0](x)
                x = self.hcbr_p4(x)
                for block in stage.blocks[1:]:
                    x = block(x)
            else:
                x = stage(x)
            if idx == 1 and self.hcbr_p3 is not None:
                x = self.hcbr_p3(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

    def _forward_cced_grer(self, x):
        outs = []
        for idx, stage in enumerate(self.res_layers):
            stage_input = x
            stage_base = stage(stage_input)
            if idx == 2:  # complete original S4: stride8 -> stride16
                x = stage_base
                # Both branches independently read the exact same F3 and are
                # added directly to the complete original F4_base.
                if self.cced_34 is not None:
                    x = x + self.cced_34(stage_input, base=stage_base)
                if self.grer_34 is not None:
                    x = x + self.grer_34(stage_input, base=stage_base)
            else:
                x = stage_base
            # At idx=3, original S5 has consumed the already enhanced F4.
            if idx in self.return_idx:
                outs.append(x)
        return outs

    def _forward_phsb(self, x):
        f2 = self.res_layers[0](x)
        f3_base = self.res_layers[1](f2)
        # The main downsampling stage consumes the untouched F3_base, never
        # the enhanced P3, so the branch only intervenes through two explicit
        # and independently gated residuals.
        f4_base = self.res_layers[2](f3_base)
        p3_residual, p4_residual = self.phsb(f3_base, f4_base)
        f3 = f3_base + p3_residual
        f4 = f4_base + p4_residual
        f5 = self.res_layers[3](f4)
        levels = (f2, f3, f4, f5)
        return [feature for idx, feature in enumerate(levels) if idx in self.return_idx]

    def _debug_variant_features(self, features):
        if self._variant_debug_iteration % self.backbone_variant_debug_interval == 0:
            returned_stages = [idx for idx in range(len(self.res_layers))
                               if idx in self.return_idx]
            for stage_idx, feature in zip(returned_stages, features):
                work = feature.float()
                values = {
                    'mean': work.mean(), 'std': work.std(unbiased=False),
                    'L2': work.norm(),
                    'spatial_var': work.var(dim=(-2, -1), unbiased=False).mean(),
                    'channel_var': work.mean(dim=(-2, -1)).var(dim=1, unbiased=False).mean(),
                }
                message = ' '.join(f'{name}={float(value.detach()):.6g}'
                                   for name, value in values.items())
                print(f'[BackboneVariant {self.backbone_variant_type} '
                      f'P{stage_idx + 2} iter={self._variant_debug_iteration}] {message}')
        self._variant_debug_iteration += 1

    def _forward_plugins(self, x, image):
        # Original conv1, Blocks, BasicBlocks and res_layers.* keys stay intact.
        if 'p0' in self.plugins:
            x = x + self.plugins['p0'](image)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            stage_input = x
            x = stage(stage_input)
            if idx == 0 and 'p1' in self.plugins:       # complete S2, stride4
                x = x + self.plugins['p1'](x)
            elif idx == 1 and 'p2' in self.plugins:     # complete S3, stride8
                x = x + self.plugins['p2'](x)
            elif idx == 2:                              # complete S4, stride16
                if 'p3' in self.plugins:                # S3 -> S4 side branch
                    x = x + self.plugins['p3'](stage_input)
                if 'p4' in self.plugins:
                    x = x + self.plugins['p4'](x)
            # S5 executes exactly its original stage, with no plugin hook.
            if idx in self.return_idx:
                outs.append(x)
        return outs

    def _forward_secd(self, x):
        outs = []
        for idx, stage in enumerate(self.res_layers):
            stage_input = x
            x = stage(stage_input)
            bypass = self.secd_34 if idx == 2 else self.secd_45 if idx == 3 else None
            if bypass is not None:
                x = x + bypass(stage_input)
            if idx in self.return_idx:
                outs.append(x)
        return outs


