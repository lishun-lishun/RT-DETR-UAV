"""Profile CRCG/CGQS/QCLR candidates without training or dataset access.

Conv/Linear GFLOPs are a reproducible lower bound (2 FLOPs per MAC). CUDA
timing and peak allocated memory are reported when a CUDA device is selected.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time
import warnings

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_crcg_cgqs_qclr import (  # noqa: E402
    FAMILY, _allow_import_only_stubs)


CANDIDATES = ('', '_crcg', '_cgqs', '_qclr', '_crcg_cgqs_qclr')


def build(suffix, device):
    from src.core import YAMLConfig

    path = ROOT / 'configs' / 'rtdetr' / f'{FAMILY}{suffix}.yml'
    config = YAMLConfig(str(path))
    config.yaml_cfg['HRNetV2W18']['pretrained'] = False
    model = config.model.eval().to(device)
    model.multi_scale = None
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    return model


def conv_linear_macs(model, image, autocast_context):
    macs = 0

    def hook(layer, inputs, output):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += (output.numel() * (layer.in_channels // layer.groups)
                     * layer.kernel_size[0] * layer.kernel_size[1])
        else:
            macs += output.numel() * layer.in_features

    handles = [module.register_forward_hook(hook)
               for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode(), autocast_context():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return macs


def profile_one(suffix, device, batch_size, image_size, warmup, repeats, amp):
    torch.manual_seed(0)
    model = build(suffix, device)
    image = torch.randn(batch_size, 3, image_size, image_size, device=device)

    def autocast_context():
        if device.type == 'cuda':
            return torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=amp)
        return nullcontext()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        macs = conv_linear_macs(model, image, autocast_context)
        with torch.inference_mode():
            for _ in range(warmup):
                with autocast_context():
                    model(image)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                for _ in range(repeats):
                    with autocast_context():
                        model(image)
            end.record()
            torch.cuda.synchronize(device)
            milliseconds = start.elapsed_time(end) / repeats
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 2 ** 20
        else:
            start_time = time.perf_counter()
            with torch.inference_mode():
                for _ in range(repeats):
                    with autocast_context():
                        model(image)
            milliseconds = (time.perf_counter() - start_time) * 1000 / repeats
            peak_memory_mb = None

    result = {
        'name': FAMILY + suffix,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameters': sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad),
        'conv_linear_gflops_per_image': 2 * macs / batch_size / 1e9,
        'average_inference_ms_per_batch': milliseconds,
        'images_per_second': batch_size * 1000 / milliseconds,
        'peak_cuda_allocated_mb': peak_memory_mb,
        'batch_size': batch_size,
        'image_size': image_size,
        'amp': bool(amp and device.type == 'cuda'),
        'device': str(device),
    }
    del model, image
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_size < 32 \
            or args.warmup < 0 or args.repeats < 1:
        parser.error('invalid batch/image/warmup/repeats value')
    device = torch.device(
        'cuda:0' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')

    _allow_import_only_stubs()
    results = [profile_one(
        suffix, device, args.batch_size, args.image_size,
        args.warmup, args.repeats, args.amp)
        for suffix in CANDIDATES]
    report = json.dumps(results, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
