"""Benchmark four DUT R18 detectors at batch 1, 640 square, AMP inference.

Examples (from rtdetr_pytorch):
    python tools/benchmark_bafr_hcbr.py --device cuda:0 --output reports/bafr_hcbr.json
    python tools/benchmark_bafr_hcbr.py --complexity-only

The reported GFLOPs are *profiler-counted operator FLOPs*, not total model
FLOPs. PyTorch does not assign FLOPs to every operation; see the per-model
counted breakdown and uncounted examples in the JSON report. No checkpoint is
loaded, no dataset is read, and no training configuration file is modified.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'Baseline': 'rtdetr_r18vd_dut_anti_uav.yml',
    'BAFR': 'rtdetr_r18vd_dut_anti_uav_bafr.yml',
    'HCBR': 'rtdetr_r18vd_dut_anti_uav_hcbr.yml',
    'BAFR+HCBR': 'rtdetr_r18vd_dut_anti_uav_bafr_hcbr.yml',
}
EXPECTED_SWITCHES = {
    'Baseline': (False, False),
    'BAFR': (True, False),
    'HCBR': (False, True),
    'BAFR+HCBR': (True, True),
}
ALLOWED_CONFIG_DIFFERENCES = (
    '__include__', 'output_dir', 'BackboneEnhancement', 'BAFR', 'HCBR',
    'PResNet.BackboneEnhancement', 'PResNet.BAFR', 'PResNet.HCBR',
)
UNCOUNTED_EXAMPLES = (
    'pool', 'norm', 'softmax', 'attention', 'grid_sampler', 'upsample',
    'interpolate', 'sigmoid', 'tanh', 'relu', 'gelu', 'pow', 'add', 'mul',
    'div', 'sub', 'sqrt', 'mean', 'sum', 'abs', 'clamp',
)


def config_path(name):
    return ROOT / 'configs' / 'rtdetr' / CONFIGS[name]


def load_audit_helper():
    path = ROOT / 'tools' / 'analyze_dut_models.py'
    spec = importlib.util.spec_from_file_location('_bafr_hcbr_config_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def switches(config):
    selection = config.get('BackboneEnhancement', {})
    if not isinstance(selection, dict):
        raise ValueError('BackboneEnhancement must be a mapping')
    if 'bafr' in selection and 'hcbr' in selection:
        flags = (selection['bafr'], selection['hcbr'])
        if not all(isinstance(flag, bool) for flag in flags):
            raise ValueError('BackboneEnhancement bafr/hcbr must be booleans')
        return flags
    names = {
        'baseline': (False, False), 'bafr': (True, False),
        'hcbr': (False, True), 'bafr_hcbr': (True, True),
        'bafr+hcbr': (True, True),
    }
    if selection.get('type') in names:
        return names[selection['type']]
    raise ValueError('BackboneEnhancement must define bafr/hcbr or type')


def audit_configs(audit):
    resolved = {name: audit.fresh_config(config_path(name)) for name in CONFIGS}
    baseline = resolved['Baseline']
    descriptions = {}
    for name, config in resolved.items():
        actual = switches(config)
        if actual != EXPECTED_SWITCHES[name]:
            raise ValueError(f'{name}: expected switches {EXPECTED_SWITCHES[name]}, got {actual}')
        diff = audit.differences(baseline, config)
        forbidden = [key for key in diff if not any(
            key == prefix or key.startswith(prefix + '.')
            for prefix in ALLOWED_CONFIG_DIFFERENCES)]
        if forbidden:
            raise ValueError(f'{name}: non-backbone YAML differences: {forbidden}')
        descriptions[name] = {
            'config': str(config_path(name).relative_to(ROOT)).replace('\\', '/'),
            'bafr': actual[0], 'hcbr': actual[1],
            'differences_from_baseline': diff,
        }
    return descriptions


def build_model(core, name, seed, device):
    import torch

    # Override only this in-memory copy. The training YAML still uses its
    # normal pretrained policy, while benchmark runs never download weights.
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(config_path(name)), PResNet={'pretrained': False})
    model = config.model.eval()
    model.multi_scale = None
    return model.to(device)


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def complexity_profile(torch, model, image, device, amp):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                 enabled=amp), \
            torch.profiler.profile(activities=activities, with_flops=True,
                                   record_shapes=True) as profile:
        output = model(image)
    if set(output) != {'pred_logits', 'pred_boxes'}:
        raise RuntimeError(f'Unexpected detector output keys: {list(output)}')
    if any(not torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError('Detector produced non-finite output')

    counted, uncounted = {}, {}
    for event in profile.key_averages():
        flops = int(event.flops or 0)
        if flops > 0:
            counted[event.key] = {'flops': flops, 'calls': event.count}
        elif any(word in event.key.lower() for word in UNCOUNTED_EXAMPLES):
            uncounted[event.key] = event.count
    total = sum(item['flops'] for item in counted.values())
    return {
        'params_total': sum(parameter.numel() for parameter in model.parameters()),
        'params_trainable': sum(parameter.numel() for parameter in model.parameters()
                                if parameter.requires_grad),
        'counted_operator_gflops': total / 1e9,
        'counted_operator_breakdown': dict(sorted(counted.items())),
        'uncounted_operator_examples': dict(sorted(uncounted.items())),
        'output_shapes': {key: list(value.shape) for key, value in output.items()},
    }


def cuda_latency(torch, model, image, device, warmup, iterations):
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for _ in range(warmup):
            model(image)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        pairs = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(image)
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize(device)
    elapsed = [start.elapsed_time(end) for start, end in pairs]
    mean = statistics.fmean(elapsed)
    return {
        'latency_ms': {
            'mean': mean, 'median': statistics.median(elapsed),
            'p95': percentile(elapsed, 0.95),
        },
        'fps_batch1': 1000.0 / mean,
        'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
        'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0',
                        help='CUDA device for the full benchmark (default: cuda:0)')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--complexity-only', action='store_true',
                        help='Profile params/counted GFLOPs on CPU; omit CUDA timings')
    parser.add_argument('--output', type=Path, help='Optional JSON output file')
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        parser.error('warmup >= 0, iterations >= 1 and threads >= 1 are required')
    try:
        import torch
    except ImportError as error:
        parser.exit(2, f'PyTorch is required: {error}\n')
    if not args.complexity_only:
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            parser.exit(2, 'CUDA is unavailable. Use --complexity-only for CPU params/GFLOPs.\n')
        try:
            device = torch.device(args.device)
            torch.cuda.get_device_properties(device)
        except (RuntimeError, ValueError, AssertionError) as error:
            parser.exit(2, f'Invalid CUDA device {args.device}: {error}\n')
    else:
        device = torch.device('cpu')

    absent = [str(config_path(name)) for name in CONFIGS if not config_path(name).is_file()]
    if absent:
        parser.exit(2, 'Missing benchmark YAML(s): ' + ', '.join(absent) + '\n')
    torch.set_num_threads(args.threads)
    audit = load_audit_helper()
    try:
        audited = audit_configs(audit)
    except ValueError as error:
        parser.exit(2, f'Configuration audit failed: {error}\n')
    core = audit.import_model_source(selective=True)
    report = {
        'protocol': {
            'input': [1, 3, 640, 640], 'mode': 'eval/inference',
            'device': str(device), 'amp_cuda_fp16': not args.complexity_only,
            'warmup': 0 if args.complexity_only else args.warmup,
            'iterations': 0 if args.complexity_only else args.iterations,
            'timer': None if args.complexity_only else 'torch.cuda.Event',
            'torch': torch.__version__,
            'gpu': None if args.complexity_only else torch.cuda.get_device_name(device),
            'pretrained': 'disabled only in memory for benchmark',
            'flops': ('torch.profiler with_flops=True; counted operator FLOPs '
                      'only, not complete model FLOPs; 1 MAC = 2 FLOPs where '
                      'the profiler uses this convention'),
            'timing_excludes': 'data loading and postprocessing',
            'memory': 'peak allocated/reserved during timed loop, including model and input',
        },
        'configs': audited,
        'results': {},
    }
    for name in CONFIGS:
        model = build_model(core, name, args.seed, device)
        image = torch.randn(1, 3, 640, 640, device=device)
        result = complexity_profile(torch, model, image, device,
                                    amp=not args.complexity_only)
        if not args.complexity_only:
            result.update(cuda_latency(torch, model, image, device,
                                       args.warmup, args.iterations))
        report['results'][name] = result
        print(name, json.dumps({key: value for key, value in result.items()
                                if key not in ('counted_operator_breakdown',
                                               'uncounted_operator_examples')}), flush=True)
        del model, image
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    baseline = report['results']['Baseline']
    for result in report['results'].values():
        result['delta_params'] = result['params_total'] - baseline['params_total']
        result['delta_counted_operator_gflops'] = (
            result['counted_operator_gflops'] - baseline['counted_operator_gflops'])
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
        print(f'Saved {args.output}')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
