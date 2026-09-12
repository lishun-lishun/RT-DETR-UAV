"""Measure RT-DETR/CURE parameters, FLOPs, latency, FPS, and CUDA memory."""

import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.nn as nn


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig  # noqa: E402
from src.nn.backbone.cure import MaskedRingConv2d  # noqa: E402


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    return checkpoint["model"]


def count_conv_linear_flops(model, image):
    """Count dense Conv2d/Linear FLOPs, using two FLOPs per multiply-add."""
    total = [0]
    handles = []

    def convolution_hook(module, _inputs, output):
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        total[0] += output.numel() * kernel_ops * 2

    def masked_convolution_hook(module, _inputs, output):
        # The implementation uses a dense convolution with masked-zero weights,
        # so report the actual k*k compute rather than only nonzero coefficients.
        kernel_ops = (
            module.kernel_size
            * module.kernel_size
            * module.in_channels
            // module.groups
        )
        total[0] += output.numel() * kernel_ops * 2

    def linear_hook(module, _inputs, output):
        total[0] += output.numel() * module.in_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, MaskedRingConv2d):
            handles.append(module.register_forward_hook(masked_convolution_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(image)
    for handle in handles:
        handle.remove()
    return total[0]


def main(args):
    config = YAMLConfig(args.config)
    # Random initialization is sufficient for structural benchmarking, and a
    # supplied detector checkpoint replaces every model weight anyway.
    config.yaml_cfg["PResNet"]["pretrained"] = False
    model = config.model
    if args.checkpoint:
        model.load_state_dict(checkpoint_state(args.checkpoint), strict=True)

    device = torch.device(args.device)
    model.to(device).eval()
    image = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    use_cuda = device.type == "cuda"
    amp_enabled = bool(args.amp and use_cuda)

    def amp_context():
        if amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    with amp_context():
        flops = count_conv_linear_flops(model, image)

    def forward():
        with torch.no_grad(), amp_context():
            model(image)

    for _ in range(args.warmup):
        forward()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.iterations):
        forward()
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / args.iterations
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "amp": amp_enabled,
        "batch_size": args.batch_size,
        "input_size": [args.height, args.width],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_ms_per_batch": latency_ms,
        "images_per_second": args.batch_size * 1000.0 / latency_ms,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "conv_linear_flops": flops,
        "conv_linear_gflops": flops / 1.0e9,
    }
    if use_cuda:
        result["peak_cuda_memory_mib"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    main(parser.parse_args())
