"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None, return_backbone_features=False):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        if return_backbone_features:
            input_size = tuple(x.shape[-2:])
            backbone_features = self.backbone(x)
            x = self.encoder(backbone_features)
            x = self.decoder(x, targets)
            return x, backbone_features, input_size

        # Keep the original default path, including feature lifetimes: do not
        # retain pre-encoder maps throughout decoder inference when CTER is off.
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
