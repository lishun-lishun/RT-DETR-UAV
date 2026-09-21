"""Audit and benchmark Baseline/CCED34/GRER34/combined at batch=1, 640 AMP.

The FLOP number is the PyTorch-profiler counted-operator lower bound. Elementwise
sqrt/log/exp/median/MAD/sigmoid and some fused kernels are not assigned FLOPs by
the profiler, so the report labels this limitation instead of claiming a false
exact total.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_cced_grer_config_audit',
                                              ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

CONFIGS = {
    'Baseline': 'rtdetr_r18vd_dut_anti_uav.yml',
    'CCED34': 'rtdetr_r18vd_dut_anti_uav_cced34.yml',
    'GRER34': 'rtdetr_r18vd_dut_anti_uav_grer34.yml',
    'CCED34+GRER34': 'rtdetr_r18vd_dut_anti_uav_cced34_grer34.yml',
}


def config_path(name):
    return ROOT / 'configs/rtdetr' / CONFIGS[name]


def audit_configs():
    baseline = audit.fresh_config(config_path('Baseline'))
    results, failures = {}, []
    for name in CONFIGS:
        config = audit.fresh_config(config_path(name))
        difference = audit.differences(
            {k: v for k, v in baseline.items() if k != '__include__'},
            {k: v for k, v in config.items() if k != '__include__'})
        forbidden = [key for key in difference if key != 'output_dir'
                     and key not in ('CCED', 'GRER')
                     and not key.startswith(('CCED.', 'GRER.'))]
        failures.extend(f'{name}: forbidden difference {key}' for key in forbidden)
        if config['MERT']['enabled'] or config['SECD']['enabled']:
            failures.append(f'{name}: MERT and legacy SECD must remain disabled')
        results[name] = {'path': str(config_path(name).relative_to(ROOT)),
                         'differences_from_baseline': difference}
    return {'passed': not failures, 'failures': failures, 'configs': results}


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
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure(core, name, args):
    import torch
    model = build_model(core, name, args.seed).to(args.device)
    image = torch.randn(1, 3, 640, 640, device=args.device)
    result = {
        'params': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_params': sum(parameter.numel() for parameter in model.parameters()
                                if parameter.requires_grad),
    }
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.amp):
        output = model(image)
    if any(not torch.isfinite(output[key]).all() for key in ('pred_boxes', 'pred_logits')):
        raise RuntimeError(f'{name} produced nonfinite predictions')

    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.amp), \
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                   with_flops=True, record_shapes=False) as profile:
        model(image)
    counted = {event.key: int(event.flops) for event in profile.key_averages() if event.flops}
    result['counted_operator_gflops_lower_bound'] = sum(counted.values()) / 1e9

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type='cuda', enabled=args.amp):
                model(image)
    torch.cuda.synchronize(args.device)
    starts, ends = [], []
    with torch.inference_mode():
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type='cuda', enabled=args.amp):
                model(image)
            end.record()
            starts.append(start)
            ends.append(end)
    torch.cuda.synchronize(args.device)
    milliseconds = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    mean = statistics.fmean(milliseconds)
    result['latency_ms'] = {
        'mean': mean,
        'median': statistics.median(milliseconds),
        'p95': percentile(milliseconds, .95),
    }
    result['fps'] = 1000.0 / mean
    del model, image, output
    torch.cuda.empty_cache()
    return result


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
    report = {'audit': audit_configs()}
    if not report['audit']['passed']:
        raise AssertionError(report['audit']['failures'])
    if not args.audit_only:
        import torch
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            parser.error('Latency protocol requires a CUDA device')
        torch.set_num_threads(args.threads)
        core = audit.import_model_source(selective=True)
        report['protocol'] = {
            'input': [1, 3, 640, 640], 'batch': 1, 'amp': args.amp,
            'warmup': args.warmup, 'iterations': args.iterations,
            'timing': 'torch.cuda.Event', 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(args.device),
            'flops': 'PyTorch profiler counted-operator lower bound',
        }
        report['results'] = {}
        for name in CONFIGS:
            report['results'][name] = measure(core, name, args)
            print(name, json.dumps(report['results'][name]), flush=True)
        baseline = report['results']['Baseline']
        for result in report['results'].values():
            result['delta_params'] = result['params'] - baseline['params']
            result['delta_counted_operator_gflops'] = (
                result['counted_operator_gflops_lower_bound']
                - baseline['counted_operator_gflops_lower_bound'])
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
        print(f'Saved {args.output}')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
