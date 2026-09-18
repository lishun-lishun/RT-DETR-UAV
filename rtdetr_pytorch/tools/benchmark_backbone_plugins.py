"""Audit pure-Backbone YAMLs and benchmark real detector forward, without data.

Run from rtdetr_pytorch:
  python tools/benchmark_backbone_plugins.py --device cuda --amp --output reports/backbone_plugin_metrics.json
  python tools/benchmark_backbone_plugins.py --resolved-only

Training settings are never edited. Only the profiling model's in-memory
pretrained flag is disabled to avoid downloading weights. GFLOPs are explicitly
PyTorch-profiler counted-operator LOWER BOUNDS, not a claim of total FLOPs.
"""

import argparse
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'rtdetr_r18vd_dut_anti_uav'
METHODS = {'Baseline': '', 'SRFD': '_srfd', 'DEConv': '_deconv',
           'DRB': '_drb', 'AKConv': '_akconv', 'RFAConv': '_rfaconv'}
spec = importlib.util.spec_from_file_location('_backbone_existing_audit',
                                             ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def config_path(method):
    return ROOT / 'configs/rtdetr' / (PREFIX + METHODS[method] + '.yml')


def resolved_audit():
    baseline = audit.fresh_config(config_path('Baseline'))
    failures, methods = [], {}
    for method in METHODS:
        cfg = audit.fresh_config(config_path(method))
        diff = audit.differences(baseline, cfg)
        forbidden = [key for key in diff if key not in
                     ('__include__', 'output_dir', 'RTDETR.backbone', 'BackbonePlugin')
                     and not key.startswith('BackbonePlugin.')]
        failures.extend(f'{method}: forbidden change {key}' for key in forbidden)
        for name in ('MERT', 'SECD'):
            if cfg.get(name, {}).get('enabled', False):
                failures.append(f'{method}: {name} must be disabled')
        if method != 'Baseline' and cfg['RTDETR']['backbone'] != 'PResNetWithPlugin':
            failures.append(f'{method}: opt-in backbone adapter not selected')
        methods[method] = {'config': str(config_path(method).relative_to(ROOT)),
                           'differences': diff, 'plugin': cfg.get('BackbonePlugin')}
    return {'passed': not failures, 'failures': failures, 'methods': methods,
            'unchanged_protocol': {key: baseline.get(key) for key in
                ('PResNet', 'RTDETR', 'HybridEncoder', 'RTDETRTransformer',
                 'SetCriterion', 'optimizer', 'lr_scheduler', 'epoches',
                 'train_dataloader', 'val_dataloader', 'MERT', 'SECD')}}


def build_model(method, seed=0, selective=False, plugin_override=None):
    import torch
    core = audit.import_model_source(selective)
    importlib.import_module('src.nn.backbone.backbone_plugins')
    cfg = copy.deepcopy(audit.fresh_config(config_path(method)))
    cfg['PResNet']['pretrained'] = False  # PROFILING/TEST ONLY, never write YAML
    if plugin_override is not None:
        cfg['BackbonePlugin'] = copy.deepcopy(plugin_override)
        cfg['RTDETR']['backbone'] = 'PResNetWithPlugin'
    torch.manual_seed(seed)
    core.merge_config(cfg)
    return core.create(cfg['model']), cfg


def measure(method, args):
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use --device cpu, or run on your server')
    if args.amp and not args.device.startswith('cuda'):
        raise ValueError('--amp benchmarks CUDA FP16, not a different CPU AMP protocol')
    model, cfg = build_model(method, args.seed, args.model_only_import)
    model = model.to(args.device).eval()
    image = torch.randn(1, 3, 640, 640, device=args.device)
    shapes = []
    handle = model.backbone.register_forward_hook(
        lambda module, inputs, outputs: shapes.extend(list(t.shape) for t in outputs))
    try:
        with torch.inference_mode(), torch.autocast('cuda', enabled=args.amp):
            output = model(image)
    finally:
        handle.remove()
    if shapes != [[1, 128, 80, 80], [1, 256, 40, 40], [1, 512, 20, 20]]:
        raise AssertionError(f'{method}: unexpected backbone outputs {shapes}')
    if any(not torch.isfinite(value).all() for value in output.values()):
        raise AssertionError(f'{method}: nonfinite evaluation output')
    with torch.inference_mode(), torch.autocast('cuda', enabled=args.amp), \
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                   with_flops=True, record_shapes=True) as profile:
        model(image)
    counted = {event.key: int(event.flops) for event in profile.key_averages()
               if event.flops}
    uncounted = sorted(event.key for event in profile.key_averages()
                       if event.key.startswith('aten::') and not event.flops)
    result = {
        'params': sum(p.numel() for p in model.parameters()),
        'backbone_params': sum(p.numel() for p in model.backbone.parameters()),
        'backbone_shapes': shapes,
        'output_shapes': {key: list(value.shape) for key, value in output.items()},
        'counted_operator_gflops_lower_bound': sum(counted.values()) / 1e9,
        'counted_operator_flops': counted,
        'uncounted_operator_names': uncounted,
        'training_pretrained_policy_unchanged': audit.fresh_config(config_path(method))['PResNet']['pretrained'],
    }
    if not args.metrics_only:
        def sync():
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
        times = []
        with torch.inference_mode(), torch.autocast('cuda', enabled=args.amp):
            for _ in range(args.warmup):
                model(image)
            sync()
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                model(image)
                sync()
                times.append((time.perf_counter_ns() - start) / 1e6)
        mean = statistics.mean(times)
        result['latency'] = {'mean_ms': mean, 'median_ms': statistics.median(times),
                             'p95_ms': sorted(times)[math.ceil(len(times) * .95) - 1],
                             'fps_from_mean': 1000 / mean}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    parser.add_argument('--resolved-only', action='store_true')
    parser.add_argument('--metrics-only', action='store_true')
    parser.add_argument('--model-only-import', action='store_true',
                        help='Load real model source only; not a complete training-import check')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        parser.error('warmup>=0, iterations>=1, threads>=1 required')
    report = {'audit': resolved_audit()}
    if not report['audit']['passed']:
        raise AssertionError(report['audit']['failures'])
    if not args.resolved_only:
        import torch
        torch.set_num_threads(args.threads)
        report['measurement_protocol'] = {
            'input': [1, 3, 640, 640], 'seed': args.seed,
            'torch': torch.__version__, 'device': args.device,
            'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'amp': args.amp, 'warmup': args.warmup, 'iterations': args.iterations,
            'model_only_import': args.model_only_import,
            'deploy': False, 'checkpoint': None,
            'latency_scope': 'synchronized wall-clock full model forward, resident input, no DataLoader',
            'flops_scope': 'profiler counted-operator lower bound; excludes some normalization, softmax, sampling arithmetic, etc.',
        }
        results = {}
        for method in args.methods:
            print(f'Measuring {method}...', flush=True)
            results[method] = measure(method, args)
            print(json.dumps({method: results[method]}, ensure_ascii=True), flush=True)
        if 'Baseline' in results:
            baseline = results['Baseline']
            for result in results.values():
                result['added_params'] = result['params'] - baseline['params']
                result['params_change_percent'] = result['added_params'] / baseline['params'] * 100
                result['added_counted_gflops_lower_bound'] = (
                    result['counted_operator_gflops_lower_bound'] - baseline['counted_operator_gflops_lower_bound'])
                result['counted_gflops_change_percent'] = (
                    result['added_counted_gflops_lower_bound'] /
                    baseline['counted_operator_gflops_lower_bound'] * 100)
        report['results'] = results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
        print(f'Saved {args.output}', flush=True)
    elif args.resolved_only:
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
