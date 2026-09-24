
import math

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR',
           'OneCycleLR', 'LambdaLR', 'WarmupCosineLR']



SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register(lr_scheduler.OneCycleLR)
LambdaLR = register(lr_scheduler.LambdaLR)


@register
class WarmupCosineLR(lr_scheduler._LRScheduler):
    """Epoch-wise linear warmup followed by proportional cosine decay.

    ``min_lr_ratio`` is applied independently to every parameter group's base
    LR, so the detector/backbone LR ratio is preserved throughout training.
    The repository calls ``step()`` once after each completed epoch.
    """

    def __init__(self, optimizer, total_epochs, warmup_epochs=5,
                 warmup_start_factor=0.1, min_lr_ratio=0.01, last_epoch=-1):
        if not isinstance(total_epochs, int) or total_epochs < 2:
            raise ValueError('total_epochs must be an integer >= 2')
        if (not isinstance(warmup_epochs, int) or warmup_epochs < 0
                or warmup_epochs >= total_epochs):
            raise ValueError('warmup_epochs must be in [0, total_epochs)')
        if not 0.0 < float(warmup_start_factor) <= 1.0:
            raise ValueError('warmup_start_factor must be in (0, 1]')
        if not 0.0 <= float(min_lr_ratio) <= 1.0:
            raise ValueError('min_lr_ratio must be in [0, 1]')
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = float(warmup_start_factor)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _factor(self, epoch):
        epoch = min(max(int(epoch), 0), self.total_epochs - 1)
        if self.warmup_epochs and epoch < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return 1.0
            progress = epoch / (self.warmup_epochs - 1)
            return (self.warmup_start_factor
                    + (1.0 - self.warmup_start_factor) * progress)
        cosine_count = self.total_epochs - self.warmup_epochs
        progress = ((epoch - self.warmup_epochs) / max(1, cosine_count - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def get_lr(self):
        factor = self._factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]
