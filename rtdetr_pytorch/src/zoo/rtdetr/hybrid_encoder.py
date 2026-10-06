'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation
from .acr_neck import ACRFusion
from .slr_neck import SLRNeck
from .paf_neck import PhaseAdaptiveFusion
from .bor_neck import BackgroundOrthogonalResidual
from .dgfr_neck import DGFRNeck
from .resample_neck import (
    LearnablePixelReassemblyUpsample,
    SubpixelPreservingDownsample,
)

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    __share__ = ['ACR', 'SLR', 'PAF', 'BOR', 'DGFR', 'LPRU', 'SPDR']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 ACR=None,
                 SLR=None,
                 PAF=None,
                 BOR=None,
                 DGFR=None,
                 LPRU=None,
                 SPDR=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        acr_cfg = {} if ACR is None else copy.deepcopy(ACR)
        if not isinstance(acr_cfg, dict):
            raise ValueError('ACR must be a mapping')
        allowed_acr = {
            'enabled', 'energy_calibration', 'semantic_routing',
            'detail_routing', 'scale_min', 'scale_max', 'detach_scale',
            'semantic', 'detail', 'eps', 'debug', 'debug_interval',
        }
        unknown_acr = set(acr_cfg) - allowed_acr
        if unknown_acr:
            raise ValueError(f'Unknown ACR options: {sorted(unknown_acr)}')
        self.acr_enabled = acr_cfg.pop('enabled', False)
        if not isinstance(self.acr_enabled, bool):
            raise ValueError('ACR.enabled must be boolean')

        slr_cfg = {} if SLR is None else copy.deepcopy(SLR)
        if not isinstance(slr_cfg, dict):
            raise ValueError('SLR must be a mapping')
        allowed_slr = {
            'enabled', 'detail_source_channels', 'query_dim',
            'position_dim', 'alpha_max', 'alpha_init',
        }
        unknown_slr = set(slr_cfg) - allowed_slr
        if unknown_slr:
            raise ValueError(f'Unknown SLR options: {sorted(unknown_slr)}')
        self.slr_enabled = slr_cfg.pop('enabled', False)
        if not isinstance(self.slr_enabled, bool):
            raise ValueError('SLR.enabled must be boolean')
        if self.slr_enabled and self.acr_enabled:
            raise ValueError('SLR and ACR are independent experiments and cannot mix')

        paf_cfg = {} if PAF is None else copy.deepcopy(PAF)
        if not isinstance(paf_cfg, dict):
            raise ValueError('PAF must be a mapping')
        allowed_paf = {
            'enabled', 'align_54', 'align_43', 'query_dim',
            'debug', 'debug_interval',
        }
        unknown_paf = set(paf_cfg) - allowed_paf
        if unknown_paf:
            raise ValueError(f'Unknown PAF options: {sorted(unknown_paf)}')
        self.paf_enabled = paf_cfg.pop('enabled', False)
        if not isinstance(self.paf_enabled, bool):
            raise ValueError('PAF.enabled must be boolean')
        self.paf_align_54 = paf_cfg.pop('align_54', True)
        self.paf_align_43 = paf_cfg.pop('align_43', True)
        if not isinstance(self.paf_align_54, bool):
            raise ValueError('PAF.align_54 must be boolean')
        if not isinstance(self.paf_align_43, bool):
            raise ValueError('PAF.align_43 must be boolean')

        bor_cfg = {} if BOR is None else copy.deepcopy(BOR)
        if not isinstance(bor_cfg, dict):
            raise ValueError('BOR must be a mapping')
        allowed_bor = {
            'enabled', 'outer_kernel', 'inner_kernel', 'theta', 'tau',
            'alpha_max', 'alpha_init', 'eps', 'debug',
        }
        unknown_bor = set(bor_cfg) - allowed_bor
        if unknown_bor:
            raise ValueError(f'Unknown BOR options: {sorted(unknown_bor)}')
        self.bor_enabled = bor_cfg.pop('enabled', False)
        if not isinstance(self.bor_enabled, bool):
            raise ValueError('BOR.enabled must be boolean')
        if self.paf_enabled and self.bor_enabled:
            raise ValueError(
                'PAF and BOR cannot be enabled simultaneously in current experiments.')
        if ((self.paf_enabled or self.bor_enabled)
                and (self.acr_enabled or self.slr_enabled)):
            raise ValueError(
                'PAF/BOR first-round experiments cannot mix with ACR or SLR')

        dgfr_cfg = {} if DGFR is None else copy.deepcopy(DGFR)
        if not isinstance(dgfr_cfg, dict):
            raise ValueError('DGFR must be a mapping')
        allowed_dgfr = {
            'enabled', 'fusion_channels', 'gamma_max', 'gamma_init',
            'debug', 'debug_interval',
        }
        unknown_dgfr = set(dgfr_cfg) - allowed_dgfr
        if unknown_dgfr:
            raise ValueError(f'Unknown DGFR options: {sorted(unknown_dgfr)}')
        self.dgfr_enabled = dgfr_cfg.pop('enabled', False)
        if not isinstance(self.dgfr_enabled, bool):
            raise ValueError('DGFR.enabled must be boolean')
        if (self.dgfr_enabled and any((
                self.acr_enabled, self.slr_enabled,
                self.paf_enabled, self.bor_enabled))):
            raise ValueError(
                'DGFR is an independent experiment and cannot mix with '
                'ACR, SLR, PAF or BOR')

        lpru_cfg = {} if LPRU is None else copy.deepcopy(LPRU)
        if not isinstance(lpru_cfg, dict):
            raise ValueError('LPRU must be a mapping')
        allowed_lpru = {'enabled', 'alpha_max', 'alpha_init', 'debug'}
        unknown_lpru = set(lpru_cfg) - allowed_lpru
        if unknown_lpru:
            raise ValueError(f'Unknown LPRU options: {sorted(unknown_lpru)}')
        self.lpru_enabled = lpru_cfg.pop('enabled', False)
        if not isinstance(self.lpru_enabled, bool):
            raise ValueError('LPRU.enabled must be boolean')

        spdr_cfg = {} if SPDR is None else copy.deepcopy(SPDR)
        if not isinstance(spdr_cfg, dict):
            raise ValueError('SPDR must be a mapping')
        allowed_spdr = {'enabled', 'beta_max', 'beta_init', 'debug'}
        unknown_spdr = set(spdr_cfg) - allowed_spdr
        if unknown_spdr:
            raise ValueError(f'Unknown SPDR options: {sorted(unknown_spdr)}')
        self.spdr_enabled = spdr_cfg.pop('enabled', False)
        if not isinstance(self.spdr_enabled, bool):
            raise ValueError('SPDR.enabled must be boolean')

        if self.lpru_enabled and self.spdr_enabled:
            raise ValueError(
                'LPRU and SPDR must be evaluated independently in the current experiment.')
        old_neck_enabled = any((
            self.acr_enabled, self.slr_enabled, self.paf_enabled,
            self.bor_enabled, self.dgfr_enabled,
        ))
        if (self.lpru_enabled or self.spdr_enabled) and old_neck_enabled:
            raise ValueError(
                'LPRU/SPDR experiments cannot mix with ACR, SLR, PAF, BOR or DGFR')

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # ACR is a pure optional route around the existing CCFF fusion.  When
        # disabled, no ACR module or parameter is constructed and the forward
        # below executes the original statements unchanged.
        if self.acr_enabled:
            if len(in_channels) != 3:
                raise ValueError('ACR first implementation requires exactly P3/P4/P5')
            defaults = {
                'energy_calibration': True,
                'semantic_routing': True,
                'detail_routing': True,
                'scale_min': 0.5,
                'scale_max': 2.0,
                'detach_scale': True,
                'semantic': {'rho': 0.20, 'theta': 0.0, 'tau': 0.20},
                'detail': {'theta': 1.0, 'tau': 0.50,
                           'beta_max': 0.5, 'beta_init': 0.1},
                'eps': 1e-6,
                'debug': False,
                'debug_interval': 100,
            }
            for key, value in acr_cfg.items():
                if (key in ('semantic', 'detail') and isinstance(value, dict)):
                    defaults[key].update(value)
                else:
                    defaults[key] = value
            self.acr_54 = ACRFusion(hidden_dim, '45', **copy.deepcopy(defaults))
            self.acr_43 = ACRFusion(hidden_dim, '34', **copy.deepcopy(defaults))

        # SLR is constructed after every original HybridEncoder block and is
        # called only after CCFF has completed N3/N4/N5. Isolate its random
        # initialization so a fixed seed preserves the original decoder and
        # all pre-existing parameter initializations.
        if self.slr_enabled:
            detail_channels = slr_cfg.pop('detail_source_channels', None)
            if detail_channels is None:
                raise ValueError(
                    'SLR.detail_source_channels is required when SLR is enabled')
            with torch.random.fork_rng(devices=[]):
                self.slr = SLRNeck(
                    hidden_dim=hidden_dim,
                    detail_source_channels=detail_channels,
                    **slr_cfg)

        # PAF keeps the original nearest-neighbour upsampling and CSPRepLayer;
        # it only aligns the already-upsampled feature immediately before each
        # top-down concatenation. The two routes intentionally do not share
        # projection weights. Isolating initialization preserves every common
        # seeded parameter in the original detector.
        if self.paf_enabled:
            if len(in_channels) != 3:
                raise ValueError(
                    'PAF first implementation requires exactly P3/P4/P5')
            with torch.random.fork_rng(devices=[]):
                if self.paf_align_54:
                    self.paf54 = PhaseAdaptiveFusion(
                        hidden_dim=hidden_dim, **copy.deepcopy(paf_cfg))
                if self.paf_align_43:
                    self.paf43 = PhaseAdaptiveFusion(
                        hidden_dim=hidden_dim, **copy.deepcopy(paf_cfg))

        # BOR is a post-CCFF residual on N3 only. N4 and N5 never enter this
        # module, and the complete original top-down/bottom-up paths run first.
        if self.bor_enabled:
            if len(in_channels) != 3:
                raise ValueError(
                    'BOR first implementation requires exactly P3/P4/P5')
            with torch.random.fork_rng(devices=[]):
                self.bor = BackgroundOrthogonalResidual(
                    hidden_dim=hidden_dim, **bor_cfg)

        # DGFR is a parallel post-CCFF branch. It consumes only the unified
        # input-projection/AIFI features and injects its outputs after the
        # untouched top-down and bottom-up CCFF have produced O3/O4/O5.
        # Isolated initialization keeps all common original weights bit-exact
        # under the same seed.
        if self.dgfr_enabled:
            if len(in_channels) != 3:
                raise ValueError(
                    'DGFR first implementation requires exactly P3/P4/P5')
            with torch.random.fork_rng(devices=[]):
                self.dgfr = DGFRNeck(
                    hidden_dim=hidden_dim,
                    conv_norm_factory=ConvNormLayer,
                    fusion_factory=CSPRepLayer,
                    act=act,
                    **dgfr_cfg)

        # Each resampling location owns an independent block. Construction is
        # isolated from the global RNG so enabling a candidate leaves every
        # common detector parameter identically initialized under the same
        # seed. Disabled candidates construct no module and execute no call.
        if self.lpru_enabled:
            if len(in_channels) != 3:
                raise ValueError('LPRU requires exactly P3/P4/P5')
            with torch.random.fork_rng(devices=[]):
                self.lpru54 = LearnablePixelReassemblyUpsample(
                    channels=hidden_dim, **copy.deepcopy(lpru_cfg))
                self.lpru43 = LearnablePixelReassemblyUpsample(
                    channels=hidden_dim, **copy.deepcopy(lpru_cfg))

        if self.spdr_enabled:
            if len(in_channels) != 3:
                raise ValueError('SPDR requires exactly P3/P4/P5')
            with torch.random.fork_rng(devices=[]):
                self.spdr34 = SubpixelPreservingDownsample(
                    channels=hidden_dim, **copy.deepcopy(spdr_cfg))
                self.spdr45 = SubpixelPreservingDownsample(
                    channels=hidden_dim, **copy.deepcopy(spdr_cfg))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                # persistent=False keeps it out of state_dict, so released
                # checkpoints still load with strict=True
                self.register_buffer(f'pos_embed{idx}', pos_embed, persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        detail_source = None
        if isinstance(feats, dict):
            if set(feats) != {'features', 'detail'}:
                raise ValueError(
                    'SLR backbone output must contain only features and detail')
            if not self.slr_enabled:
                raise RuntimeError(
                    'Backbone returned an SLR detail source while SLR is disabled')
            detail_source = feats['detail']
            feats = feats['features']
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        detail_residuals = {}
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            if self.lpru_enabled:
                lpru = self.lpru54 if idx == 2 else self.lpru43
                upsample_feat = lpru(feat_high, upsample_feat)
            if self.acr_enabled:
                acr_fusion = self.acr_54 if idx == 2 else self.acr_43
                upsample_feat, detail_residuals[idx - 1] = acr_fusion(
                    feat_low, upsample_feat)
            if self.paf_enabled:
                paf_fusion = (getattr(self, 'paf54', None) if idx == 2
                              else getattr(self, 'paf43', None))
                if paf_fusion is not None:
                    upsample_feat = paf_fusion(feat_low, upsample_feat)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if self.spdr_enabled:
                spdr = self.spdr34 if idx == 0 else self.spdr45
                downsample_feat = spdr(feat_low, downsample_feat)
            if self.acr_enabled:
                acr_fusion = self.acr_43 if idx == 0 else self.acr_54
                downsample_feat = acr_fusion.inject_detail(
                    downsample_feat, detail_residuals.get(idx))
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        if self.dgfr_enabled:
            outs = self.dgfr(proj_feats, outs)
        if self.slr_enabled:
            if detail_source is None:
                raise RuntimeError('SLR is enabled but the backbone supplied no detail source')
            outs[0] = self.slr(outs[0], detail_source)
        if self.bor_enabled:
            outs[0] = self.bor(outs[0])
        return outs
