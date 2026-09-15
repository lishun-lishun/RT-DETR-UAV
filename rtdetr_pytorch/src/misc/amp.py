"""One autocast policy shared by training, evaluation and benchmarking."""

import torch


def autocast_context(device, enabled=False):
    device = torch.device(device)
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16,
        enabled=bool(enabled and device.type == 'cuda'),
        cache_enabled=True,
    )
