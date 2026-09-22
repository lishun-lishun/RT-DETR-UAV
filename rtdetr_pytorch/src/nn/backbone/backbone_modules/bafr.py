"""Blur-Adaptive Field Routing (BAFR) for a PResNet BasicBlock.

The routing field replaces only ``branch2a.conv``.  The original block's
normalisation, second (channel-mixing) convolution, shortcut and activation
keep their names and can therefore reuse their pretrained parameters.
"""

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['BAFRBlock', 'BAFRSpatialField']


def _fixed_integer(value, expected, name):
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError('%s must be %d in BAFR v1' % (name, expected))


class BAFRSpatialField(nn.Module):
    """Use local fine/support evidence to mix two depthwise spatial fields."""

    def __init__(self, channels, fine_kernel=3, support_kernel=7, eps=1e-6,
                 fine_branch=None, support_branch=None, debug=False):
        super().__init__()
        _fixed_integer(fine_kernel, 3, 'fine_kernel')
        _fixed_integer(support_kernel, 7, 'support_kernel')
        if isinstance(eps, bool) or not isinstance(eps, (float, int)) or eps <= 0:
            raise ValueError('eps must be a positive number')
        if not isinstance(debug, bool):
            raise ValueError('debug must be boolean')
        fine_branch = {'kernel': 3, 'dilation': 1} if fine_branch is None else dict(fine_branch)
        support_branch = ({'kernel': 3, 'first_dilation': 1, 'second_dilation': 2}
                          if support_branch is None else dict(support_branch))
        if set(fine_branch) != {'kernel', 'dilation'}:
            raise ValueError('fine_branch requires kernel and dilation only')
        if set(support_branch) != {'kernel', 'first_dilation', 'second_dilation'}:
            raise ValueError('support_branch requires kernel and first/second_dilation only')
        _fixed_integer(fine_branch['kernel'], 3, 'fine_branch.kernel')
        _fixed_integer(fine_branch['dilation'], 1, 'fine_branch.dilation')
        _fixed_integer(support_branch['kernel'], 3, 'support_branch.kernel')
        _fixed_integer(support_branch['first_dilation'], 1, 'support_branch.first_dilation')
        _fixed_integer(support_branch['second_dilation'], 2, 'support_branch.second_dilation')
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError('channels must be a positive integer')

        self.eps = float(eps)
        self.debug = debug
        self.last_debug_stats = None
        self.fine_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.support_dw1 = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=False)
        self.support_dw2 = nn.Conv2d(channels, channels, 3, padding=2,
                                     dilation=2, groups=channels, bias=False)

    def route_evidence(self, x):
        """Return (fine evidence, support evidence, support route) in FP32.

        This method is intentionally public for offline diagnostics.  Calling
        it under ``torch.no_grad()`` avoids retaining an autograd graph.
        Pooling excludes padded cells so a spatially constant input has zero
        evidence even at the image boundary.
        """
        x32 = x.float()
        a3 = F.avg_pool2d(x32, kernel_size=3, stride=1, padding=1,
                          count_include_pad=False)
        a7 = F.avg_pool2d(x32, kernel_size=7, stride=1, padding=3,
                          count_include_pad=False)
        ef = (x32 - a3).abs().mean(dim=1, keepdim=True)
        es = (a3 - a7).abs().mean(dim=1, keepdim=True)
        route = es / (ef + es + self.eps)
        return ef, es, route

    def forward(self, x):
        ef, es, route = self.route_evidence(x)
        fine = self.fine_dw(x)
        support = self.support_dw2(self.support_dw1(x))
        # Route/evidence arithmetic remains FP32 even under CUDA autocast.
        mixed = ((1.0 - route) * fine.float() + route * support.float())
        if self.debug:
            ef_detached = ef.detach().flatten()
            es_detached = es.detach().flatten()
            route_detached = route.detach().flatten()
            self.last_debug_stats = {
                'fine_evidence_mean': ef_detached.mean(),
                'fine_evidence_p90': torch.quantile(ef_detached, 0.90),
                'support_evidence_mean': es_detached.mean(),
                'support_evidence_p90': torch.quantile(es_detached, 0.90),
                'route_b_mean': route_detached.mean(),
                'route_b_p10': torch.quantile(route_detached, 0.10),
                'route_b_p50': torch.quantile(route_detached, 0.50),
                'route_b_p90': torch.quantile(route_detached, 0.90),
                'fine_branch_norm': torch.linalg.vector_norm(fine.detach().float()),
                'support_branch_norm': torch.linalg.vector_norm(support.detach().float()),
                'output_norm': torch.linalg.vector_norm(mixed.detach()),
            }
        return mixed.to(dtype=fine.dtype)


class BAFRBlock(nn.Module):
    """Replace BasicBlock spatial modelling, preserving its residual topology.

    Args:
        original_block: Existing stride-one, identity-shortcut BasicBlock.
        **kwargs: Parameters accepted by :class:`BAFRSpatialField`.
    """

    expansion = 1

    def __init__(self, original_block, **kwargs):
        super().__init__()
        if not hasattr(original_block, 'branch2a') or not hasattr(original_block, 'branch2b'):
            raise TypeError('BAFRBlock requires an existing PResNet BasicBlock')
        old_conv = original_block.branch2a.conv
        if (not isinstance(old_conv, nn.Conv2d) or old_conv.stride != (1, 1)
                or old_conv.in_channels != old_conv.out_channels
                or old_conv.kernel_size != (3, 3) or not original_block.shortcut):
            raise ValueError('BAFRBlock v1 requires a stride-one identity-shortcut BasicBlock')
        self.shortcut = original_block.shortcut
        self.branch2a = original_block.branch2a
        self.branch2b = original_block.branch2b
        self.act = original_block.act
        if not self.shortcut:
            self.short = original_block.short
        self.branch2a.conv = BAFRSpatialField(old_conv.in_channels, **kwargs)

    def route_evidence(self, x):
        return self.branch2a.conv.route_evidence(x)

    @property
    def last_debug_stats(self):
        return self.branch2a.conv.last_debug_stats

    def forward(self, x):
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)
