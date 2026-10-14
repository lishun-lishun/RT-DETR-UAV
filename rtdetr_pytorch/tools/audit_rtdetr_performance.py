"""Audit RT-DETR complexity and PyTorch inference performance at 640x640.

This is a read-only diagnostic.  It does not train, save weights, or alter the
configuration/checkpoint on disk.  Results are written to a new JSON file.

The report deliberately keeps these quantities separate:

* executed Conv2d/Linear MACs (one multiply-accumulate = one MAC),
* 2 x MACs (the convention that counts multiply and add separately),
* torch.profiler's operator FLOPs, and
* optional fvcore/THOP estimates and their unsupported operators.

Example (CUDA_VISIBLE_DEVICES remaps physical GPU 1 to cuda:0):

    CUDA_VISIBLE_DEVICES=1 python tools/audit_rtdetr_performance.py \
      -c configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml \
      -r output/rtdetr_r18vd_dut_anti_uav/best.pth \
      --device cuda:0 --amp --batches 1 8 \
      --output performance_audit_results.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_SIZE = 640
HEAD_PREFIXES = (
    'decoder.enc_output',
    'decoder.enc_score_head',
    'decoder.enc_bbox_head',
    'decoder.dec_score_head',
    'decoder.dec_bbox_head',
)
GROUPS = ('Backbone', 'Hybrid Encoder', 'Decoder', 'Detection Head', 'Unassigned')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument('-r', '--resume', type=Path,
                        help='checkpoint; omit only for architecture-only diagnostics')
    parser.add_argument('--weights', choices=('auto', 'ema', 'model'), default='auto',
                        help='auto prefers ema.module, matching normal evaluation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true',
                        help='enable CUDA FP16 autocast')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--profile-iterations', type=int, default=10,
                        help='iterations used only for the bottleneck operator table')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path,
                        default=Path('performance_audit_results.json'))
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f'configuration does not exist: {args.config}')
    if args.resume is not None and not args.resume.is_file():
        parser.error(f'checkpoint does not exist: {args.resume}')
    if args.resume is None and args.weights != 'auto':
        parser.error('--weights ema/model requires --resume')
    if any(batch < 1 for batch in args.batches):
        parser.error('--batches values must be positive')
    if args.warmup < 0 or args.iterations < 1 or args.profile_iterations < 1:
        parser.error('warmup must be >= 0; iteration counts must be >= 1')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be >= 1')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('--amp requires a CUDA device')
    args.batches = list(dict.fromkeys(args.batches))
    return args


def autocast_context(device: torch.device, enabled: bool):
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16, enabled=enabled)
    return nullcontext()


def disable_pretrained_download(cfg: Any) -> None:
    model_cfg = cfg.yaml_cfg.get('RTDETR', {})
    backbone_name = model_cfg.get('backbone') if isinstance(model_cfg, dict) else None
    backbone_cfg = cfg.yaml_cfg.get(backbone_name) if backbone_name else None
    if isinstance(backbone_cfg, dict) and 'pretrained' in backbone_cfg:
        backbone_cfg['pretrained'] = False


def force_audit_shape(cfg: Any) -> None:
    for name in ('HybridEncoder', 'RTDETRTransformer'):
        component = cfg.yaml_cfg.get(name)
        if isinstance(component, dict) and 'eval_spatial_size' in component:
            component['eval_spatial_size'] = [INPUT_SIZE, INPUT_SIZE]


def select_state(checkpoint: Any, requested: str) -> Tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint must be a dictionary')
    ema = checkpoint.get('ema')
    ema_state = ema.get('module') if isinstance(ema, dict) else None
    model_state = checkpoint.get('model')
    if requested in ('auto', 'ema') and isinstance(ema_state, dict) and ema_state:
        return ema_state, 'ema.module'
    if requested == 'ema':
        raise KeyError('checkpoint has no usable ema.module')
    if isinstance(model_state, dict) and model_state:
        return model_state, 'model'
    raise KeyError('checkpoint has no usable model state')


def build_model(args: argparse.Namespace, device: torch.device):
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config), resume='', tuning='')
    disable_pretrained_download(cfg)
    force_audit_shape(cfg)
    model = cfg.model
    state_name = 'random initialization (no checkpoint supplied)'
    if args.resume is not None:
        checkpoint = torch.load(str(args.resume), map_location='cpu')
        state, state_name = select_state(checkpoint, args.weights)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'strict checkpoint load was not clean: {incompatible}')
        del state, checkpoint

    model.eval()
    if hasattr(model, 'multi_scale'):
        model.multi_scale = None
    model.to(device).eval()
    return model, state_name


def module_group(name: str) -> str:
    if name == 'backbone' or name.startswith('backbone.'):
        return 'Backbone'
    if name == 'encoder' or name.startswith('encoder.'):
        return 'Hybrid Encoder'
    if any(name == prefix or name.startswith(prefix + '.') for prefix in HEAD_PREFIXES):
        return 'Detection Head'
    if name == 'decoder' or name.startswith('decoder.'):
        return 'Decoder'
    return 'Unassigned'


def parameter_partition(model: nn.Module) -> Dict[str, Dict[str, int]]:
    result = {group: {'params': 0, 'trainable_params': 0} for group in GROUPS}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        group = module_group(name)
        result[group]['params'] += parameter.numel()
        if parameter.requires_grad:
            result[group]['trainable_params'] += parameter.numel()
    result['Total'] = {
        'params': sum(item['params'] for item in result.values()),
        'trainable_params': sum(item['trainable_params'] for item in result.values()),
    }
    return result


def tensor_outputs(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_outputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_outputs(item)


def conv_linear_macs(model: nn.Module, image: torch.Tensor,
                     device: torch.device, amp: bool) -> Dict[str, Any]:
    by_group: MutableMapping[str, int] = defaultdict(int)
    by_type: MutableMapping[str, int] = defaultdict(int)
    calls: Counter[str] = Counter()
    handles = []

    def make_hook(name: str):
        def hook(layer: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            tensors = list(tensor_outputs(output))
            if len(tensors) != 1:
                raise TypeError(f'{name} produced {len(tensors)} tensor outputs')
            out = tensors[0]
            if isinstance(layer, nn.Conv2d):
                mac = (out.numel() * (layer.in_channels // layer.groups)
                       * layer.kernel_size[0] * layer.kernel_size[1])
                kind = 'Conv2d'
            else:
                mac = out.numel() * layer.in_features
                kind = 'Linear'
            group = module_group(name)
            by_group[group] += int(mac)
            by_type[kind] += int(mac)
            calls[f'{group}/{kind}'] += 1
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode(), autocast_context(device, amp):
            model(image)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    total = sum(by_group.values())
    groups = {}
    for group in GROUPS:
        macs = by_group[group]
        groups[group] = {
            'macs': macs,
            'gmacs': macs / 1e9,
            'two_x_macs_gflops': 2.0 * macs / 1e9,
        }
    groups['Total'] = {
        'macs': total,
        'gmacs': total / 1e9,
        'two_x_macs_gflops': 2.0 * total / 1e9,
    }
    return {
        'scope': 'executed nn.Conv2d and nn.Linear modules only',
        'groups': groups,
        'by_type_gmacs': {key: value / 1e9 for key, value in by_type.items()},
        'calls': dict(sorted(calls.items())),
    }


def torch_profiler_flops(model: nn.Module, image: torch.Tensor,
                         device: torch.device, amp: bool) -> Dict[str, Any]:
    def capture(activities: List[Any]) -> Dict[str, Any]:
        with torch.inference_mode(), autocast_context(device, amp), \
                torch.profiler.profile(activities=activities, with_flops=True,
                                       record_shapes=True) as profile:
            model(image)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        breakdown = []
        for event in profile.key_averages(group_by_input_shape=True):
            flops = int(getattr(event, 'flops', 0) or 0)
            if flops:
                breakdown.append({
                    'operator': event.key,
                    'input_shapes': str(getattr(event, 'input_shapes', '')),
                    'calls': int(event.count),
                    'flops': flops,
                    'gflops': flops / 1e9,
                })
        breakdown.sort(key=lambda item: item['flops'], reverse=True)

        nested_pairs = Counter()
        nested_child_flops = 0
        for event in profile.events():
            flops = int(getattr(event, 'flops', 0) or 0)
            parent = getattr(event, 'cpu_parent', None)
            parent_flops = int(getattr(parent, 'flops', 0) or 0) if parent else 0
            if flops and parent_flops:
                nested_pairs[f'{parent.name} -> {event.name}'] += 1
                nested_child_flops += flops
        return {
            'total_gflops': sum(item['flops'] for item in breakdown) / 1e9,
            'counted_operator_breakdown': breakdown,
            'nested_counted_parent_child_pairs': dict(nested_pairs),
            'nested_child_gflops_potentially_double_counted': (
                nested_child_flops / 1e9),
        }

    # ATen CPU-dispatch events are the stable source of profiler FLOP formulae,
    # even when the tensors execute on CUDA.  A second CPU+CUDA capture checks
    # whether this PyTorch/CUPTI combination assigns FLOPs to device events too.
    cpu_dispatch = capture([torch.profiler.ProfilerActivity.CPU])
    combined = cpu_dispatch
    if device.type == 'cuda':
        combined = capture([
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ])
    total = cpu_dispatch['total_gflops']
    return {
        # Backward-compatible alias; the CPU-dispatch figure is preferred.
        'total_gflops': total,
        'cpu_dispatch_activity': cpu_dispatch,
        'cpu_and_cuda_activities': combined,
        'combined_minus_cpu_dispatch_gflops': (
            combined['total_gflops'] - cpu_dispatch['total_gflops']),
        'nested_child_gflops_potentially_double_counted': (
            cpu_dispatch['nested_child_gflops_potentially_double_counted']),
        'interpretation': (
            'Sum of torch.profiler FLOP formulae for registered ATen operators. '
            'The CPU-dispatch-only capture is preferred. If CPU+CUDA is larger, '
            'the extra amount is an activity-accounting artifact, not another '
            'model forward. Neither value is automatically MACs or complete.'),
    }


def fvcore_crosscheck(model: nn.Module, image: torch.Tensor) -> Dict[str, Any]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}
    try:
        analysis = FlopCountAnalysis(model, (image,))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        analysis.tracer_warnings('none')
        total = int(analysis.total())
        by_module = analysis.by_module()
        unsupported = dict(analysis.unsupported_ops())

        def value(name: str) -> int:
            return int(by_module.get(name, 0))

        head = sum(value(prefix) for prefix in HEAD_PREFIXES)
        decoder_all = value('decoder')
        grouped = {
            'Backbone': value('backbone'),
            'Hybrid Encoder': value('encoder'),
            'Decoder': max(0, decoder_all - head),
            'Detection Head': head,
        }
        grouped['Unassigned'] = max(0, total - sum(grouped.values()))
        grouped['Total'] = total
        return {
            'available': True,
            'convention': ('fvcore counts one fused multiply-add as one FLOP; '
                           'values are MAC-like and are not 2x-MAC GFLOPs'),
            'total_gflops_fma_equals_one': total / 1e9,
            'groups_gflops_fma_equals_one': {
                key: number / 1e9 for key, number in grouped.items()},
            'unsupported_operators': unsupported,
            'uncalled_modules': sorted(analysis.uncalled_modules()),
        }
    except Exception as exc:
        return {'available': True, 'failed': True,
                'error': f'{type(exc).__name__}: {exc}'}


def thop_crosscheck(model: nn.Module, image: torch.Tensor) -> Dict[str, Any]:
    try:
        from thop import profile
    except Exception as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}
    try:
        macs, params = profile(model, inputs=(image,), verbose=False)
        return {
            'available': True,
            'total_gmacs': float(macs) / 1e9,
            'total_2x_macs_gflops': 2.0 * float(macs) / 1e9,
            'params': int(params),
            'warning': ('THOP is module-hook based. Functional attention, grid_sample '
                        'and unregistered custom operations can be omitted.'),
        }
    except Exception as exc:
        return {'available': True, 'failed': True,
                'error': f'{type(exc).__name__}: {exc}'}


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def time_forward(model: nn.Module, image: torch.Tensor, device: torch.device,
                 amp: bool, warmup: int, iterations: int) -> Dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context(device, amp):
                model(image)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        with torch.inference_mode():
            for start, end in zip(starts, ends):
                start.record()
                with autocast_context(device, amp):
                    model(image)
                end.record()
        torch.cuda.synchronize(device)
        elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        timer = 'CUDA events; synchronized before and after timed region'
    else:
        elapsed = []
        with torch.inference_mode():
            for _ in range(iterations):
                start = time.perf_counter()
                with autocast_context(device, amp):
                    model(image)
                elapsed.append((time.perf_counter() - start) * 1000.0)
        timer = 'time.perf_counter CPU wall time'
    mean_ms = statistics.fmean(elapsed)
    batch = image.shape[0]
    return {
        'batch_size': batch,
        'batch_latency_mean_ms': mean_ms,
        'batch_latency_median_ms': statistics.median(elapsed),
        'batch_latency_p95_ms': percentile(elapsed, 0.95),
        'throughput_images_per_second': 1000.0 * batch / mean_ms,
        'mean_ms_per_image': mean_ms / batch,
        'warmup_iterations': warmup,
        'timed_iterations': iterations,
        'timer': timer,
        'scope': 'model forward only; excludes preprocessing, H2D and postprocessing',
    }


def time_stages(model: nn.Module, image: torch.Tensor, device: torch.device,
                amp: bool, warmup: int, iterations: int) -> Dict[str, Any]:
    if device.type != 'cuda':
        return {'available': False, 'reason': 'stage timing is CUDA-event only'}

    def pipeline():
        features = model.backbone(image)
        encoded = model.encoder(features)
        return model.decoder(encoded, None)

    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context(device, amp):
                pipeline()
    torch.cuda.synchronize(device)
    records = []
    with torch.inference_mode():
        for _ in range(iterations):
            points = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            points[0].record()
            with autocast_context(device, amp):
                features = model.backbone(image)
                points[1].record()
                encoded = model.encoder(features)
                points[2].record()
                model.decoder(encoded, None)
                points[3].record()
            records.append(points)
    torch.cuda.synchronize(device)
    names = ('Backbone', 'Hybrid Encoder', 'Decoder including Detection Head')
    values = {name: [] for name in names}
    for points in records:
        for index, name in enumerate(names):
            values[name].append(points[index].elapsed_time(points[index + 1]))
    return {
        'available': True,
        'batch_size': image.shape[0],
        'mean_ms': {name: statistics.fmean(times) for name, times in values.items()},
        'note': ('Events are inserted into one normal pipeline. Detection heads are '
                 'nested in decoder and therefore included in its stage latency.'),
    }


def amp_audit(model: nn.Module, image: torch.Tensor, device: torch.device,
              amp: bool) -> Dict[str, Any]:
    dtypes: Counter[str] = Counter()
    handles = []

    def hook(_layer: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        for tensor in tensor_outputs(output):
            if tensor.is_floating_point():
                dtypes[str(tensor.dtype)] += 1

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode(), autocast_context(device, amp):
            autocast_enabled_inside = (torch.is_autocast_enabled()
                                       if device.type == 'cuda' else False)
            output = model(image)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    output_dtypes = Counter(
        str(tensor.dtype) for tensor in tensor_outputs(output)
        if tensor.is_floating_point())
    return {
        'requested': amp,
        'autocast_enabled_inside_context': autocast_enabled_inside,
        'conv_linear_output_dtype_counts': dict(dtypes),
        'model_output_dtype_counts': dict(output_dtypes),
        'verified_fp16_execution': bool(
            amp and autocast_enabled_inside and dtypes.get('torch.float16', 0)),
    }


def bottleneck_profile(model: nn.Module, image: torch.Tensor,
                       device: torch.device, amp: bool,
                       iterations: int) -> Dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    with torch.inference_mode(), torch.profiler.profile(
            activities=activities, record_shapes=False, profile_memory=True) as profile:
        for _ in range(iterations):
            with autocast_context(device, amp):
                model(image)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    rows = []
    for event in profile.key_averages():
        cuda_us = float(getattr(event, 'self_cuda_time_total', 0.0) or 0.0)
        cpu_us = float(getattr(event, 'self_cpu_time_total', 0.0) or 0.0)
        rows.append({
            'operator': event.key,
            'calls': int(event.count),
            'self_cuda_ms_total': cuda_us / 1000.0,
            'self_cpu_ms_total': cpu_us / 1000.0,
        })
    key = ('self_cuda_ms_total' if device.type == 'cuda'
           else 'self_cpu_ms_total')
    rows.sort(key=lambda item: item[key], reverse=True)
    sync_markers = ('_local_scalar_dense', 'item', 'cudaDeviceSynchronize',
                    'cudaStreamSynchronize', 'to', '_to_copy')
    suspicious = [row for row in rows if any(
        marker.lower() in row['operator'].lower() for marker in sync_markers)]
    return {
        'iterations': iterations,
        'top_operators': rows[:30],
        'possible_sync_or_transfer_operators': suspicious,
        'warning': 'Profiler overhead makes this ranking diagnostic, not an FPS result.',
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL).splitlines()[0].strip()
    except Exception:
        return None


def environment(device: torch.device) -> Dict[str, Any]:
    result = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'pytorch': torch.__version__,
        'torchvision': package_version('torchvision'),
        'cuda_runtime_used_by_pytorch': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'nvidia_driver': driver_version(),
        'fvcore': package_version('fvcore'),
        'thop': package_version('thop'),
        'device': str(device),
    }
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result.update({
            'gpu': properties.name,
            'gpu_total_memory_bytes': properties.total_memory,
            'gpu_compute_capability': f'{properties.major}.{properties.minor}',
        })
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def print_summary(report: Mapping[str, Any]) -> None:
    print('\nRT-DETR PERFORMANCE AUDIT (640x640)')
    print(f"  GPU/device: {report['environment'].get('gpu', report['environment']['device'])}")
    print(f"  checkpoint state: {report['protocol']['checkpoint_state']}")
    print(f"  precision: {report['protocol']['precision']}")
    print('\nComponent partition (Conv/Linear executed operations):')
    params = report['complexity']['parameters']
    macs = report['complexity']['conv_linear']
    for group in (*GROUPS[:-1], 'Total'):
        print(f"  {group:16s} params={params[group]['params']:>10,d}  "
              f"GMACs={macs['groups'][group]['gmacs']:>8.3f}  "
              f"2xMAC={macs['groups'][group]['two_x_macs_gflops']:>8.3f} GFLOPs")
    profiler = report['complexity']['torch_profiler']
    print(f"\n  torch.profiler counted: {profiler['total_gflops']:.3f} GFLOPs")
    print(f"  nested counted child events: "
          f"{profiler['nested_child_gflops_potentially_double_counted']:.3f} GFLOPs")
    print('\nTiming:')
    for item in report['timing']:
        print(f"  batch={item['batch_size']}: "
              f"{item['batch_latency_mean_ms']:.3f} ms/batch, "
              f"{item['throughput_images_per_second']:.2f} images/s, "
              f"{item['mean_ms_per_image']:.3f} ms/image")
    print(f"\nJSON: {report['protocol']['output']}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif device.type != 'cpu':
        raise ValueError('only CUDA or CPU devices are supported')

    model, checkpoint_state = build_model(args, device)
    complexity_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.inference_mode(), autocast_context(device, args.amp):
        smoke_output = model(complexity_input)
    if not isinstance(smoke_output, dict) or not {
            'pred_logits', 'pred_boxes'} <= set(smoke_output):
        raise RuntimeError('model did not produce RT-DETR prediction dictionary')

    parameters = parameter_partition(model)
    conv_linear = conv_linear_macs(model, complexity_input, device, args.amp)
    profiler = torch_profiler_flops(model, complexity_input, device, args.amp)
    fvcore = fvcore_crosscheck(model, complexity_input)
    thop = thop_crosscheck(model, complexity_input)
    amp_result = amp_audit(model, complexity_input, device, args.amp)

    timing = []
    for batch in args.batches:
        image = torch.randn(batch, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        timing.append(time_forward(
            model, image, device, args.amp, args.warmup, args.iterations))
        del image
    stage_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    stage_timing = time_stages(
        model, stage_input, device, args.amp, args.warmup, args.iterations)
    bottlenecks = bottleneck_profile(
        model, stage_input, device, args.amp, args.profile_iterations)

    report = {
        'protocol': {
            'config': str(args.config.resolve()),
            'checkpoint': str(args.resume.resolve()) if args.resume else None,
            'checkpoint_state': checkpoint_state,
            'input_shape_for_complexity': [1, 3, INPUT_SIZE, INPUT_SIZE],
            'timed_batches': args.batches,
            'precision': 'CUDA FP16 autocast' if args.amp else 'FP32',
            'model_mode': 'eval',
            'autograd': 'torch.inference_mode',
            'timing_scope': 'PyTorch model forward only',
            'excluded_from_timing': [
                'disk I/O', 'image decode', 'resize/normalization',
                'CPU-to-GPU transfer', 'RT-DETR postprocessor'],
            'fixed_input_size': [INPUT_SIZE, INPUT_SIZE],
            'cudnn_benchmark': bool(device.type == 'cuda'),
            'tf32': False,
            'seed': args.seed,
            'output': str(args.output.resolve()),
        },
        'environment': environment(device),
        'complexity': {
            'parameters': parameters,
            'conv_linear': conv_linear,
            'torch_profiler': profiler,
            'fvcore': fvcore,
            'thop': thop,
        },
        'amp_verification': amp_result,
        'timing': timing,
        'stage_timing_batch_1': stage_timing,
        'bottleneck_profile_batch_1': bottlenecks,
        'partition_definition': {
            'Backbone': 'model.backbone',
            'Hybrid Encoder': 'model.encoder',
            'Detection Head': list(HEAD_PREFIXES),
            'Decoder': ('model.decoder excluding the listed classification/bbox '
                        'heads; includes query projection/selection and decoder layers'),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + os.linesep,
        encoding='utf-8')
    print_summary(report)


if __name__ == '__main__':
    main()
