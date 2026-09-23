"""Test, profile and benchmark the BDPD/MSDConv PResNet18 variants.

Examples:
  python tools/test_bpdp_msdconv.py --test-bpdp --test-msdconv --test-grad
  python tools/test_bpdp_msdconv.py --test-baseline --test-amp
  python tools/test_bpdp_msdconv.py --test-complexity
  torchrun --nproc_per_node=2 tools/test_bpdp_msdconv.py --test-ddp --model bpdp
  python tools/test_bpdp_msdconv.py --benchmark --model msdconv
"""

import argparse
from contextlib import nullcontext
import json
import statistics
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'bpdp': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml',
    'msdconv': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml',
    'combined': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp_msdconv.yml',
}
_CORE = None


def _iter_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def print_gradient_report(seed=0):
    """Print the core branch gradient norms requested by the acceptance test."""
    import torch

    sys.path.insert(0, str(ROOT))
    from src.nn.backbone.backbone_modules.bdpd import BDPDDownsample
    from src.nn.backbone.backbone_modules.msdconv import MSDConv

    torch.manual_seed(seed)
    bpdp = BDPDDownsample(4, 8).train()
    bpdp_input = torch.randn(2, 4, 17, 19, requires_grad=True)
    (bpdp(bpdp_input) * torch.randn(2, 8, 9, 10)).mean().backward()

    torch.manual_seed(seed)
    msdconv = MSDConv(8).train()
    msdconv_input = torch.randn(2, 8, 17, 19, requires_grad=True)
    (msdconv(msdconv_input) * torch.randn_like(msdconv_input)).mean().backward()

    report = {
        'BDPD': {
            'base_projection_grad_norm': bpdp.base_projection.conv.weight.grad.norm().item(),
            'detail_projection_grad_norm': bpdp.detail_projection.conv.weight.grad.norm().item(),
            'alpha_grad_norm': bpdp.raw_alpha.grad.norm().item(),
            'input_grad_norm': bpdp_input.grad.norm().item(),
        },
        'MSDConv': {
            'router_grad_norm': msdconv.context_router.weight.grad.norm().item(),
            'projection_grad_norm': msdconv.projection.conv.weight.grad.norm().item(),
            'beta_grad_norm': msdconv.raw_beta.grad.norm().item(),
            'input_grad_norm': msdconv_input.grad.norm().item(),
        },
    }
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0
           for values in report.values() for value in values.values()):
        raise RuntimeError('A core BDPD/MSDConv gradient is zero or non-finite')
    print(json.dumps({'gradient_norms': report}, indent=2))


def run_unit_tests(args):
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / 'tests'), pattern='test_bpdp_msdconv.py',
        top_level_dir=str(ROOT))
    flags = any((args.test_baseline, args.test_bpdp, args.test_msdconv,
                 args.test_grad, args.test_amp))
    selected = unittest.TestSuite()
    for case in _iter_cases(suite):
        name = case.id().lower()
        if not flags or any((
                args.test_baseline and 'baseline_equivalence' in name,
                args.test_bpdp and ('bdpdtests' in name or 'structure_and_shapes' in name),
                args.test_msdconv and ('msdconvtests' in name or 'structure_and_shapes' in name),
                args.test_grad and 'backward' in name,
                args.test_amp and 'amp_no_nan_inf' in name)):
            selected.addTest(case)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    if result.wasSuccessful() and args.test_grad:
        print_gradient_report(args.seed)
    return result.wasSuccessful()


def _model_imports():
    global _CORE
    if _CORE is not None:
        return _CORE
    sys.path.insert(0, str(ROOT))
    from tools.analyze_dut_models import import_model_source
    _CORE = import_model_source(selective=True)
    return _CORE


def build_model(name, device='cpu', seed=0):
    import torch
    core = _model_imports()
    torch.manual_seed(seed)
    config = core.YAMLConfig(str(CONFIGS[name]), PResNet={'pretrained': False})
    model = config.model.to(device).eval()
    model.multi_scale = None
    return model


def profile_complexity(seed=0):
    import torch
    import torch.nn as nn

    rows = []
    for name in CONFIGS:
        model = build_model(name, seed=seed)
        sample = torch.randn(1, 3, 640, 640)
        macs = {'value': 0}

        def hook(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                macs['value'] += (output.numel() * (module.in_channels // module.groups)
                                  * module.kernel_size[0] * module.kernel_size[1])
            elif isinstance(module, nn.Linear):
                macs['value'] += output.numel() * module.in_features

        handles = [module.register_forward_hook(hook) for module in model.modules()
                   if isinstance(module, (nn.Conv2d, nn.Linear))]
        try:
            with torch.no_grad(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    with_flops=True) as profile:
                output = model(sample)
        finally:
            for handle in handles:
                handle.remove()
        if any(not torch.isfinite(tensor).all() for tensor in output.values()):
            raise RuntimeError(f'{name} complexity forward produced NaN/Inf')
        profiler_flops = sum(event.flops or 0 for event in profile.key_averages())
        rows.append({
            'model': name,
            'params': sum(parameter.numel() for parameter in model.parameters()),
            'conv_linear_gmacs_lower_bound': macs['value'] / 1e9,
            'profiler_gflops_lower_bound': profiler_flops / 1e9,
        })
        del model, sample, output
    baseline = rows[0]
    for row in rows:
        row['delta_params'] = row['params'] - baseline['params']
        row['delta_profiler_gflops'] = (row['profiler_gflops_lower_bound']
                                        - baseline['profiler_gflops_lower_bound'])
    print(json.dumps({
        'input': [1, 3, 640, 640],
        'warning': ('Profiler FLOPs are lower bounds because not every PyTorch '
                    'operator has a FLOP formula.'),
        'models': rows,
    }, indent=2))


def benchmark(name, warmup=100, iterations=500, seed=0):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('--benchmark requires CUDA')
    model = build_model(name, device='cuda', seed=seed)
    sample = torch.randn(1, 3, 640, 640, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            model(sample)
            end.record()
        torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    ordered = sorted(times)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    mean = statistics.fmean(times)
    result = {
        'model': name, 'batch': 1, 'input': [1, 3, 640, 640], 'AMP': True,
        'warmup': warmup, 'iterations': iterations,
        'mean_latency_ms': mean,
        'median_latency_ms': statistics.median(times),
        'p95_latency_ms': p95,
        'fps_from_mean': 1000.0 / mean,
        'peak_cuda_memory_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
    }
    print(json.dumps(result, indent=2))


def ddp_smoke(name, seed=0):
    import os
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel

    use_cuda = torch.cuda.is_available()
    backend = 'nccl' if use_cuda else 'gloo'
    distributed.init_process_group(backend=backend, init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    model = build_model(name, device=device, seed=seed).backbone.train()
    ddp_options = ({'device_ids': [local_rank], 'output_device': local_rank}
                   if use_cuda else {})
    wrapped = DistributedDataParallel(
        model, find_unused_parameters=False, **ddp_options)
    sample = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
    amp_context = (torch.autocast(device_type='cuda', dtype=torch.float16)
                   if use_cuda else nullcontext())
    with amp_context:
        levels = wrapped(sample)
        loss = sum(level.float().square().mean() for level in levels)
    loss.backward()
    prefixes = []
    if name in ('bpdp', 'combined'):
        prefixes.append('res_layers.1.blocks.0.branch2a.')
    if name in ('msdconv', 'combined'):
        prefixes.append('msdconv_p3.')
    new_grads = [parameter.grad for key, parameter in model.named_parameters()
                 if key.startswith(tuple(prefixes))]
    if not new_grads or any(grad is None or not torch.isfinite(grad).all()
                            for grad in new_grads):
        raise RuntimeError(f'{name} DDP new-module gradient check failed')
    distributed.barrier()
    if distributed.get_rank() == 0:
        print(f'DDP smoke passed: model={name}, backend={backend}, '
              f'world_size={distributed.get_world_size()}, loss={loss.item():.6f}')
    distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-baseline', action='store_true')
    parser.add_argument('--test-bpdp', action='store_true')
    parser.add_argument('--test-msdconv', action='store_true')
    parser.add_argument('--test-grad', action='store_true')
    parser.add_argument('--test-amp', action='store_true')
    parser.add_argument('--test-complexity', action='store_true')
    parser.add_argument('--test-ddp', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--model', choices=tuple(CONFIGS), default='bpdp')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    special = args.test_complexity or args.test_ddp or args.benchmark
    unit_requested = any((args.test_baseline, args.test_bpdp, args.test_msdconv,
                          args.test_grad, args.test_amp))
    if args.test_complexity:
        profile_complexity(args.seed)
    if args.test_ddp:
        ddp_smoke(args.model, args.seed)
    if args.benchmark:
        benchmark(args.model, args.warmup, args.iterations, args.seed)
    if unit_requested or not special:
        return 0 if run_unit_tests(args) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
