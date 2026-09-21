"""Fair 640/AMP CUDA benchmark for Baseline, HSDR-B, HSDR-A and PHSB.

GFLOPs are a profiler counted-operator LOWER BOUND, not a claim that PyTorch
counts every elementwise/attention operation. The same implementation and
measurement protocol are applied to all four complete RT-DETR detectors.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_variant_config_audit', ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

CONFIGS = {
    'Baseline': 'rtdetr_r18vd_dut_anti_uav.yml',
    'HSDR-B': 'rtdetr_r18vd_dut_anti_uav_hsdr_b.yml',
    'HSDR-A': 'rtdetr_r18vd_dut_anti_uav_hsdr_a.yml',
    'PHSB': 'rtdetr_r18vd_dut_anti_uav_phsb.yml',
}


def config_path(name):
    return ROOT / 'configs/rtdetr' / CONFIGS[name]


def audit_configs():
    baseline = audit.fresh_config(config_path('Baseline'))
    failures, configs = [], {}
    expected = {'Baseline': ('baseline', None),
                'HSDR-B': ('hsdr', [2, 3, 2, 1]),
                'HSDR-A': ('hsdr', [2, 4, 3, 1]),
                'PHSB': ('phsb', None)}
    for name in CONFIGS:
        config = audit.fresh_config(config_path(name))
        differences = audit.differences(
            {key: value for key, value in baseline.items() if key != '__include__'},
            {key: value for key, value in config.items() if key != '__include__'})
        forbidden = [key for key in differences
                     if key != 'output_dir' and key not in ('BackboneVariant', 'PHSB')
                     and not key.startswith(('BackboneVariant.', 'PHSB.'))]
        failures.extend(f'{name}: forbidden difference {key}' for key in forbidden)
        variant = config.get('BackboneVariant', {})
        if (variant.get('type'), variant.get('stage_blocks')) != expected[name]:
            failures.append(f'{name}: wrong variant/stage_blocks')
        if config['MERT']['enabled'] or config['SECD']['enabled']:
            failures.append(f'{name}: MERT/SECD must be disabled')
        if any(value.get('enabled', False)
               for value in config.get('BackbonePlugins', {}).values()):
            failures.append(f'{name}: existing backbone plugins must be disabled')
        if config['CCED']['enabled'] or config['GRER']['enabled']:
            failures.append(f'{name}: CCED/GRER must be disabled')
        configs[name] = {'path': str(config_path(name).relative_to(ROOT)),
                         'differences_from_baseline': differences}
    return {'passed': not failures, 'failures': failures, 'configs': configs}


def build_model(core, name, seed):
    import torch
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(config_path(name)), PResNet={'pretrained': False})
    model = config.model.eval()
    model.multi_scale = None
    return model


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


def measure(core, name, args):
    import torch
    model = build_model(core, name, args.seed).to(args.device)
    image = torch.randn(1, 3, 640, 640, device=args.device)
    params = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.amp):
        result = model(image)
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError(f'{name} emitted nonfinite outputs')

    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.amp), \
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                   with_flops=True) as profile:
        model(image)
    counted_flops = sum(int(event.flops) for event in profile.key_averages()
                        if event.flops)

    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type='cuda', enabled=args.amp):
                model(image)
    torch.cuda.synchronize(args.device)
    starts, ends = [], []
    with torch.inference_mode():
        for _ in range(args.iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type='cuda', enabled=args.amp):
                model(image)
            end.record()
            starts.append(start)
            ends.append(end)
    torch.cuda.synchronize(args.device)
    latencies = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    mean = statistics.fmean(latencies)
    return {
        'params': params,
        'counted_operator_gflops_lower_bound': counted_flops / 1e9,
        'latency_ms': {'mean': mean, 'median': statistics.median(latencies),
                       'p95': percentile(latencies, .95)},
        'fps': 1000.0 / mean,
        'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(args.device) / 2**20,
        'output_shapes': {key: list(value.shape) for key, value in result.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        parser.error('warmup >= 0, iterations >= 1, threads >= 1 required')
    report = {'audit': audit_configs()}
    if not report['audit']['passed']:
        raise AssertionError(report['audit']['failures'])
    if not args.audit_only:
        import torch
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            parser.error('CUDA device required for torch.cuda.Event benchmark')
        torch.set_num_threads(args.threads)
        core = audit.import_model_source(selective=True)
        report['protocol'] = {
            'input': [1, 3, 640, 640], 'batch': 1, 'amp': args.amp,
            'warmup': args.warmup, 'iterations': args.iterations,
            'timing': 'torch.cuda.Event', 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(args.device),
            'flops': 'PyTorch profiler counted-operator lower bound',
            'pretrained': 'disabled in benchmark memory only; training YAML unchanged',
        }
        report['results'] = {}
        for name in CONFIGS:
            report['results'][name] = measure(core, name, args)
            print(name, json.dumps(report['results'][name]), flush=True)
        base = report['results']['Baseline']
        for result in report['results'].values():
            result['delta_params'] = result['params'] - base['params']
            result['delta_counted_operator_gflops'] = (
                result['counted_operator_gflops_lower_bound']
                - base['counted_operator_gflops_lower_bound'])
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
        print(f'Saved {args.output}')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
