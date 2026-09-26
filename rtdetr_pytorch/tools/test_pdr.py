"""Acceptance tests, complexity, DDP smoke and checkpoint audit for PDR.

Examples:
  python tools/test_pdr.py --test-baseline --test-shape --test-grad
  python tools/test_pdr.py --test-amp
  python tools/test_pdr.py --test-complexity
  torchrun --nproc_per_node=2 tools/test_pdr.py --test-ddp --model pdr34
  python tools/test_pdr.py --checkpoint output/.../best.pth
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'baseline': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'pdr3': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr3.yml',
    'pdr34': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr34.yml',
    'pdr34_nogate': (ROOT / 'configs/rtdetr/'
                     'rtdetr_r18vd_dut_anti_uav_pdr34_nogate.yml'),
}
_CORE = None


def _iter_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def run_unit_tests(args):
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / 'tests'), pattern='test_pdr.py',
        top_level_dir=str(ROOT))
    flags = any((args.test_baseline, args.test_shape, args.test_grad,
                 args.test_amp, args.test_pretrained))
    selected = unittest.TestSuite()
    for case in _iter_cases(suite):
        name = case.id().lower()
        if not flags or any((
                args.test_baseline and ('disabled_is_exact' in name or 'configs_fair' in name),
                args.test_shape and ('relay_uses_exact' in name or 'shapes' in name
                                     or 'persistent_chain' in name),
                args.test_grad and 'receive_gradients' in name,
                args.test_amp and 'amp_forward_backward' in name,
                args.test_pretrained and 'pretrained_original' in name)):
            selected.addTest(case)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    if result.wasSuccessful() and args.test_grad:
        print_gradient_report(args.seed)
    return result.wasSuccessful()


def _model_imports():
    global _CORE
    if _CORE is None:
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


def print_gradient_report(seed=0):
    import torch
    sys.path.insert(0, str(ROOT))
    from src.nn.backbone.backbone_modules.pdr import PersistentDetailRelay

    torch.manual_seed(seed)
    module = PersistentDetailRelay([64, 128, 256]).train()
    c2 = torch.randn(2, 64, 32, 32, requires_grad=True)
    _, d3, d4 = module.make_details(c2)
    c3 = module.inject3(torch.randn(2, 128, 16, 16), d3)
    c4 = module.inject4(torch.randn(2, 256, 8, 8), d4)
    ((c3 * torch.randn_like(c3)).mean()
     + (c4 * torch.randn_like(c4)).mean()).backward()
    parameters = {
        'detail_memory': module.detail_memory.projection.conv.weight,
        'relay3_reduce': module.relay3.reduce.conv.weight,
        'relay3_depthwise': module.relay3.dwconv.conv.weight,
        'projection3': module.injection3.projection.conv.weight,
        'raw_alpha3': module.injection3.raw_alpha,
        'theta3': module.injection3.theta,
        'relay4_reduce': module.relay4.reduce.conv.weight,
        'relay4_depthwise': module.relay4.dwconv.conv.weight,
        'projection4': module.injection4.projection.conv.weight,
        'raw_alpha4': module.injection4.raw_alpha,
        'theta4': module.injection4.theta,
    }
    report = {name: parameter.grad.norm().item()
              for name, parameter in parameters.items()}
    if any(value <= 0 or not torch.isfinite(torch.tensor(value))
           for value in report.values()):
        raise RuntimeError('PDR has a zero/non-finite required gradient')
    print(json.dumps({'pdr_gradient_norms': report}, indent=2))


def profile_complexity(seed=0):
    import torch
    import torch.nn as nn

    torch.set_num_threads(2)
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
        if any(not torch.isfinite(value).all() for value in output.values()):
            raise RuntimeError(f'{name} forward contains NaN/Inf')
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
        'warning': 'Profiler/Conv-Linear FLOPs are lower bounds.',
        'models': rows,
    }, indent=2))


def inspect_checkpoint(path, alpha_max=0.5):
    import torch
    payload = torch.load(path, map_location='cpu')
    state = payload
    source = 'root'
    if isinstance(payload, dict) and isinstance(payload.get('ema'), dict):
        state = payload['ema'].get('module', payload['ema'])
        source = 'ema.module'
    elif isinstance(payload, dict) and isinstance(payload.get('model'), dict):
        state = payload['model']
        source = 'model'
    elif isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
        state = payload['state_dict']
        source = 'state_dict'
    found = {}
    for key, value in state.items():
        if key.endswith(('pdr.injection3.raw_alpha', 'pdr.injection4.raw_alpha')):
            level = key.split('injection')[-1].split('.')[0]
            found[f'raw_alpha{level}'] = value.item()
            found[f'alpha{level}_eff'] = (alpha_max * value.float().sigmoid()).item()
        elif key.endswith(('pdr.injection3.semantic_gate.theta',
                           'pdr.injection4.semantic_gate.theta')):
            level = key.split('injection')[-1].split('.')[0]
            found[f'theta{level}'] = value.item()
    if not found:
        raise RuntimeError('No PDR alpha/theta parameters found in checkpoint')
    print(json.dumps({'checkpoint': str(Path(path).resolve()), 'weights': source,
                      'alpha_max': alpha_max, 'pdr': found}, indent=2))


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
    options = ({'device_ids': [local_rank], 'output_device': local_rank}
               if use_cuda else {})
    wrapped = DistributedDataParallel(
        model, find_unused_parameters=False, gradient_as_bucket_view=True,
        **options)
    sample = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
    context = (torch.autocast(device_type='cuda', dtype=torch.float16)
               if use_cuda else nullcontext())
    with context:
        levels = wrapped(sample)
        loss = sum(level.float().square().mean() for level in levels)
    loss.backward()
    parameters = list(model.pdr.named_parameters()) if model.pdr is not None else []
    if not parameters or any(parameter.grad is None
                             or not torch.isfinite(parameter.grad).all()
                             for _, parameter in parameters):
        raise RuntimeError(f'{name} DDP PDR gradient check failed')
    distributed.barrier()
    if distributed.get_rank() == 0:
        print(f'DDP smoke passed: model={name}, backend={backend}, '
              f'world_size={distributed.get_world_size()}, loss={loss.item():.6f}')
    distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-baseline', action='store_true')
    parser.add_argument('--test-shape', action='store_true')
    parser.add_argument('--test-grad', action='store_true')
    parser.add_argument('--test-amp', action='store_true')
    parser.add_argument('--test-pretrained', action='store_true')
    parser.add_argument('--test-complexity', action='store_true')
    parser.add_argument('--test-ddp', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--alpha-max', type=float, default=0.5)
    parser.add_argument('--model', choices=tuple(CONFIGS), default='pdr34')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    special = args.test_complexity or args.test_ddp or args.checkpoint
    unit_requested = any((args.test_baseline, args.test_shape, args.test_grad,
                          args.test_amp, args.test_pretrained))
    if args.test_complexity:
        profile_complexity(args.seed)
    if args.test_ddp:
        if args.model == 'baseline':
            parser.error('--test-ddp requires a PDR model')
        ddp_smoke(args.model, args.seed)
    if args.checkpoint:
        inspect_checkpoint(args.checkpoint, args.alpha_max)
    if unit_requested or not special:
        return 0 if run_unit_tests(args) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
