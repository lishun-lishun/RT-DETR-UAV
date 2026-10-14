"""Benchmark one RT-DETR model at a fixed 1x3x640x640 input.

The script reports model parameters, two explicitly labelled GFLOPs lower
bounds, batch-1 FPS, and single-image latency.  Timing excludes image loading,
preprocessing, and host-to-device transfer.  Postprocessing is excluded unless
``--include-postprocess`` is specified.

Example:
    python tools/benchmark_performance.py \
        -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml \
        -r output/rtdetr_r18vd_dut_anti_uav/best.pth \
        --device cuda:0 --amp --output performance.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, Tuple

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INPUT_SIZE = 640
INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, type=Path,
                        help='model YAML configuration')
    parser.add_argument('-r', '--resume', type=Path,
                        help='checkpoint; auto mode prefers ema.module')
    parser.add_argument('--weights', choices=('auto', 'ema', 'model'),
                        default='auto',
                        help='checkpoint state to benchmark (default: auto)')
    parser.add_argument('--device', default='cuda:0',
                        help='cuda device or cpu (default: cuda:0)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='untimed warmup forwards (default: 50)')
    parser.add_argument('--iterations', type=int, default=200,
                        help='timed forwards (default: 200)')
    parser.add_argument('--amp', action='store_true',
                        help='use CUDA FP16 autocast for GFLOPs and timing')
    parser.add_argument('--tf32', action='store_true',
                        help='allow TF32 for CUDA FP32 matrix operations')
    parser.add_argument('--deploy', action='store_true',
                        help='convert supported layers to their deploy form')
    parser.add_argument('--include-postprocess', action='store_true',
                        help='include RT-DETR postprocessing in FPS/latency')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4,
                        help='PyTorch CPU threads (default: 4)')
    parser.add_argument('--output', type=Path,
                        help='optional JSON result path')
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup must be >= 0')
    if args.iterations < 1:
        parser.error('--iterations must be >= 1')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be >= 1')
    if not args.config.is_file():
        parser.error(f'configuration does not exist: {args.config}')
    if args.resume is not None and not args.resume.is_file():
        parser.error(f'checkpoint does not exist: {args.resume}')
    if args.weights != 'auto' and args.resume is None:
        parser.error('--weights ema/model requires --resume')
    if args.amp and not str(args.device).startswith('cuda'):
        parser.error('--amp is supported only for CUDA benchmarking')
    return args


def disable_pretrained_download(cfg: Any) -> None:
    """Disable only in-memory backbone downloads before model construction."""
    model_cfg = cfg.yaml_cfg.get('RTDETR', {})
    backbone_name = model_cfg.get('backbone') if isinstance(model_cfg, dict) else None
    backbone_cfg = cfg.yaml_cfg.get(backbone_name) if backbone_name else None
    if isinstance(backbone_cfg, dict) and 'pretrained' in backbone_cfg:
        backbone_cfg['pretrained'] = False


def force_640_eval_shape(cfg: Any) -> None:
    """Build any cached encoder/decoder anchors for this benchmark's shape."""
    for component_name in ('HybridEncoder', 'RTDETRTransformer'):
        component = cfg.yaml_cfg.get(component_name)
        if isinstance(component, dict) and 'eval_spatial_size' in component:
            component['eval_spatial_size'] = [INPUT_SIZE, INPUT_SIZE]


def select_checkpoint_state(checkpoint: Any, requested: str) -> Tuple[Dict, str]:
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint must be a dictionary')
    ema = checkpoint.get('ema')
    ema_state = ema.get('module') if isinstance(ema, dict) else None
    model_state = checkpoint.get('model')
    if requested == 'ema':
        if not isinstance(ema_state, dict) or not ema_state:
            raise KeyError('checkpoint has no usable ema.module state')
        return ema_state, 'ema.module'
    if requested == 'model':
        if not isinstance(model_state, dict) or not model_state:
            raise KeyError('checkpoint has no usable model state')
        return model_state, 'model'
    if isinstance(ema_state, dict) and ema_state:
        return ema_state, 'ema.module'
    if isinstance(model_state, dict) and model_state:
        return model_state, 'model'
    raise KeyError('checkpoint contains neither ema.module nor model state')


def build_model(args: argparse.Namespace, device: torch.device):
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config), resume='', tuning='')
    disable_pretrained_download(cfg)
    force_640_eval_shape(cfg)
    model = cfg.model
    checkpoint_source = 'random initialization (no checkpoint supplied)'
    checkpoint_metadata: Dict[str, Any] = {}
    if args.resume is not None:
        checkpoint = torch.load(str(args.resume), map_location='cpu')
        state, checkpoint_source = select_checkpoint_state(checkpoint, args.weights)
        model.load_state_dict(state, strict=True)
        for key in ('last_epoch', 'best_stat', 'date', 'validation_stats'):
            if key in checkpoint:
                checkpoint_metadata[key] = checkpoint[key]
        del checkpoint, state

    model.eval()
    if hasattr(model, 'multi_scale'):
        model.multi_scale = None
    if args.deploy:
        model = model.deploy()
    model = model.to(device).eval()

    postprocessor = cfg.postprocessor
    if args.deploy and hasattr(postprocessor, 'deploy'):
        postprocessor = postprocessor.deploy()
    postprocessor = postprocessor.to(device).eval()
    return model, postprocessor, checkpoint_source, checkpoint_metadata


def autocast_context(device: torch.device, amp: bool):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16,
                              enabled=amp)
    return nullcontext()


def detector_forward(model: nn.Module, postprocessor: nn.Module,
                     image: torch.Tensor, original_size: torch.Tensor,
                     include_postprocess: bool):
    output = model(image)
    if not isinstance(output, dict) or not {'pred_logits', 'pred_boxes'} <= set(output):
        raise RuntimeError(
            'unexpected detector output; expected pred_logits and pred_boxes')
    if include_postprocess:
        return postprocessor(output, original_size)
    return output


def conv_linear_gflops(model: nn.Module, image: torch.Tensor,
                       device: torch.device, amp: bool) -> float:
    """Return 2*MAC for executed nn.Conv2d/nn.Linear layers only."""
    macs = 0

    def hook(layer: nn.Module, _inputs: Tuple, output: torch.Tensor) -> None:
        nonlocal macs
        if not isinstance(output, torch.Tensor):
            raise TypeError(f'non-tensor output from {type(layer).__name__}')
        if isinstance(layer, nn.Conv2d):
            macs += (output.numel() * (layer.in_channels // layer.groups)
                     * layer.kernel_size[0] * layer.kernel_size[1])
        else:
            macs += output.numel() * layer.in_features

    handles = [module.register_forward_hook(hook) for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode(), autocast_context(device, amp):
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return 2.0 * macs / 1e9


def profiler_gflops(model: nn.Module, image: torch.Tensor,
                     device: torch.device, amp: bool) -> Tuple[float, Dict, Dict]:
    """Return the operators counted by torch.profiler(with_flops=True)."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode(), autocast_context(device, amp), \
            torch.profiler.profile(activities=activities, with_flops=True,
                                   record_shapes=False) as profile:
        model(image)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    counted: Dict[str, Dict[str, int]] = {}
    uncounted: Dict[str, int] = {}
    relevant_uncounted = (
        'attention', 'grid_sampler', 'softmax', 'layer_norm', 'batch_norm',
        'interpolate', 'upsample', 'topk', 'sigmoid', 'pool')
    for event in profile.key_averages():
        flops = int(event.flops or 0)
        if flops:
            counted[event.key] = {'flops': flops, 'calls': int(event.count)}
        elif any(name in event.key.lower() for name in relevant_uncounted):
            uncounted[event.key] = int(event.count)
    total = sum(value['flops'] for value in counted.values())
    return total / 1e9, dict(sorted(counted.items())), dict(sorted(uncounted.items()))


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timed_forward(model: nn.Module, postprocessor: nn.Module,
                  image: torch.Tensor, original_size: torch.Tensor,
                  device: torch.device, amp: bool, include_postprocess: bool,
                  warmup: int, iterations: int) -> Dict[str, float]:
    def run_once():
        return detector_forward(model, postprocessor, image, original_size,
                                include_postprocess)

    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context(device, amp):
                run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        with torch.inference_mode():
            for start, end in zip(starts, ends):
                start.record()
                with autocast_context(device, amp):
                    run_once()
                end.record()
        torch.cuda.synchronize(device)
        milliseconds = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        timer = 'torch.cuda.Event (GPU execution time)'
    else:
        milliseconds = []
        with torch.inference_mode():
            for _ in range(iterations):
                start = time.perf_counter()
                with autocast_context(device, amp):
                    run_once()
                milliseconds.append((time.perf_counter() - start) * 1000.0)
        timer = 'time.perf_counter (CPU wall time)'

    mean_ms = statistics.fmean(milliseconds)
    return {
        'mean_ms': mean_ms,
        'median_ms': statistics.median(milliseconds),
        'p95_ms': percentile(milliseconds, 0.95),
        'min_ms': min(milliseconds),
        'max_ms': max(milliseconds),
        'fps': 1000.0 / mean_ms,
        'timer': timer,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def print_summary(report: Dict[str, Any]) -> None:
    metrics = report['metrics']
    timing = metrics['timing']
    print('\nRT-DETR 640x640 performance benchmark')
    print(f"  Config:              {report['protocol']['config']}")
    print(f"  Checkpoint state:    {report['protocol']['checkpoint_state']}")
    print(f"  Device:              {report['protocol']['device_name']}")
    print(f"  Precision:           {report['protocol']['precision']}")
    print(f"  Timing scope:        {report['protocol']['timing_scope']}")
    print(f"  Params:              {metrics['params_total']:,}")
    print(f"  Trainable Params:    {metrics['params_trainable']:,}")
    print(f"  GFLOPs (profiler):   {metrics['torch_profiler_counted_gflops']:.3f}  "
          '(counted-operator lower bound)')
    print(f"  GFLOPs (Conv/Linear):{metrics['conv_linear_2mac_gflops']:.3f}  "
          '(2*MAC lower bound)')
    print(f"  FPS (batch=1):       {timing['fps']:.2f}")
    print(f"  Latency mean:        {timing['mean_ms']:.3f} ms/image")
    print(f"  Latency median/p95:  {timing['median_ms']:.3f} / "
          f"{timing['p95_ms']:.3f} ms")
    print('  Note: GFLOPs values are lower bounds because torch.profiler and the')
    print('        Conv/Linear hook do not count every attention/custom operator.')


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False')
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    elif device.type != 'cpu':
        raise ValueError('only CUDA and CPU devices are supported')

    model, postprocessor, checkpoint_source, metadata = build_model(args, device)
    image = torch.randn(INPUT_SHAPE, device=device)
    original_size = torch.tensor([[INPUT_SIZE, INPUT_SIZE]], device=device)

    profiler_flops, flop_breakdown, uncounted = profiler_gflops(
        model, image, device, args.amp)
    conv_flops = conv_linear_gflops(model, image, device, args.amp)
    timing = timed_forward(
        model, postprocessor, image, original_size, device, args.amp,
        args.include_postprocess, args.warmup, args.iterations)

    precision = 'FP16 autocast' if args.amp else ('FP32/TF32' if args.tf32 else 'FP32')
    device_name = (torch.cuda.get_device_name(device)
                   if device.type == 'cuda' else 'CPU')
    report = {
        'protocol': {
            'config': str(args.config.resolve()),
            'checkpoint': str(args.resume.resolve()) if args.resume else None,
            'checkpoint_state': checkpoint_source,
            'checkpoint_metadata': json_safe(metadata),
            'input_shape': list(INPUT_SHAPE),
            'batch_size': 1,
            'device': str(device),
            'device_name': device_name,
            'precision': precision,
            'deploy': args.deploy,
            'warmup': args.warmup,
            'iterations': args.iterations,
            'timing_scope': ('model forward + RT-DETR postprocessor'
                             if args.include_postprocess else 'model forward only'),
            'timing_excludes': [
                'image loading', 'decode', 'resize/normalization',
                'host-to-device transfer'],
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cudnn_benchmark': bool(device.type == 'cuda'),
            'tf32': bool(args.tf32 and device.type == 'cuda'),
            'seed': args.seed,
        },
        'metrics': {
            'params_total': sum(parameter.numel() for parameter in model.parameters()),
            'params_trainable': sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
            'torch_profiler_counted_gflops': profiler_flops,
            'conv_linear_2mac_gflops': conv_flops,
            'flops_scope': 'detector forward only; batch=1; postprocess excluded',
            'flops_warning': (
                'Both GFLOPs values are lower bounds. torch.profiler(with_flops=True) '
                'does not assign FLOPs to every attention, grid-sampling, normalization '
                'or custom operator. Do not compare them directly with another tool.'),
            'timing': timing,
        },
        'diagnostics': {
            'torch_profiler_counted_operators': flop_breakdown,
            'torch_profiler_uncounted_operator_examples': uncounted,
        },
    }
    print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + os.linesep,
            encoding='utf-8')
        print(f'  JSON saved to:       {args.output.resolve()}')


if __name__ == '__main__':
    main()
