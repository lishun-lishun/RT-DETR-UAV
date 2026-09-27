"""Validate the optional ACR-Neck without training a dataset.

Examples:
  python tools/validate_acr_neck.py --smoke --complexity
  CUDA_VISIBLE_DEVICES=0 python tools/validate_acr_neck.py --benchmark --amp
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9921 tools/validate_acr_neck.py --ddp-smoke --amp

The MAC counter intentionally follows the project's existing reproducible
Conv/Linear lower-bound convention. Normalization, interpolation, elementwise
ACR statistics/gates, softmax and other functional operations are not counted.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_dut_models import import_model_source  # noqa: E402


core = import_model_source(selective=True)
importlib.import_module('src.nn.backbone.hrnet')
importlib.import_module('src.optim')


CONFIGS = {
    'PResNet18 + Original': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'PResNet18 + ACR': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_acr.yml',
    'HRNetV2-W18 + Original': ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml',
    'HRNetV2-W18 + ACR': ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_acr.yml',
}
ACR_NAMES = ('PResNet18 + ACR', 'HRNetV2-W18 + ACR')


def _is_hrnet(name):
    return name.startswith('HRNetV2-W18')


def build(name, *, training=False):
    overrides = ({'HRNetV2W18': {'pretrained': False,
                                  'pretrained_path': None}}
                 if _is_hrnet(name) else
                 {'PResNet': {'pretrained': False}})
    config = core.YAMLConfig(str(CONFIGS[name]), **overrides)
    model = config.model
    model.multi_scale = None
    model.train(training)
    return config, model


def _conv_linear_macs(model, image):
    total = 0

    def hook(layer, inputs, output):
        nonlocal total
        if isinstance(layer, nn.Conv2d):
            total += (output.numel() * (layer.in_channels // layer.groups)
                      * layer.kernel_size[0] * layer.kernel_size[1])
        elif isinstance(layer, nn.Linear):
            total += output.numel() * layer.in_features

    handles = [layer.register_forward_hook(hook) for layer in model.modules()
               if isinstance(layer, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode():
            output = model(image)
    finally:
        for handle in handles:
            handle.remove()
    return total, output


def complexity():
    results = {}
    for name in CONFIGS:
        torch.manual_seed(0)
        _, model = build(name)
        model = model.cpu()
        image = torch.randn(1, 3, 640, 640)
        macs, output = _conv_linear_macs(model, image)
        results[name] = {
            'whole_params': sum(parameter.numel()
                                for parameter in model.parameters()),
            'whole_conv_linear_macs': macs,
            'whole_conv_linear_flops_2_per_mac': 2 * macs,
            'prediction_shapes': {
                key: list(value.shape) for key, value in output.items()
                if torch.is_tensor(value)
            },
        }
        del model, image, output
    for backbone in ('PResNet18', 'HRNetV2-W18'):
        original = results[f'{backbone} + Original']
        candidate = results[f'{backbone} + ACR']
        candidate['added_params_vs_original'] = (
            candidate['whole_params'] - original['whole_params'])
        candidate['added_conv_linear_macs_vs_original'] = (
            candidate['whole_conv_linear_macs']
            - original['whole_conv_linear_macs'])
    print(json.dumps({'complexity': results}, indent=2))
    return results


def _optimizer_rows(config, model):
    optimizer = config.optimizer
    assignment = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group['params']:
            assignment[id(parameter)] = (
                group_index, group['lr'], group['weight_decay'])
    rows = []
    for name, parameter in model.named_parameters():
        if not name.startswith('encoder.acr_'):
            continue
        group_index, lr, weight_decay = assignment[id(parameter)]
        rows.append({'name': name, 'group': group_index, 'lr': lr,
                     'weight_decay': weight_decay})
    return rows


def smoke():
    result = {'baseline_equivalence': False, 'models': {}, 'optimizer': {}}

    # Configuration construction is deterministic. Disabled ACR creates no
    # module/parameter and must reproduce the exact original CCFF path.
    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
    kwargs = dict(in_channels=[128, 256, 512], hidden_dim=256,
                  expansion=0.5, num_encoder_layers=1,
                  eval_spatial_size=[128, 128])
    torch.manual_seed(7)
    original = HybridEncoder(**kwargs, ACR=None).eval()
    torch.manual_seed(7)
    disabled = HybridEncoder(**kwargs, ACR={'enabled': False}).eval()
    features = [torch.randn(1, 128, 16, 16),
                torch.randn(1, 256, 8, 8),
                torch.randn(1, 512, 4, 4)]
    with torch.inference_mode():
        expected, actual = original(features), disabled(features)
    if original.state_dict().keys() != disabled.state_dict().keys() or not all(
            torch.allclose(left, right, atol=1e-6, rtol=1e-5)
            for left, right in zip(expected, actual)):
        raise RuntimeError('ACR disabled is not equivalent to original CCFF')
    result['baseline_equivalence'] = True
    print('ACR disabled baseline equivalence: PASS')

    image = torch.randn(1, 3, 640, 640)
    for name in ACR_NAMES:
        config, model = build(name)
        captured = {}

        def capture(module, inputs, output):
            captured['encoder_shapes'] = [list(value.shape) for value in output]

        handle = model.encoder.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                output = model(image)
        finally:
            handle.remove()
        if not all(torch.isfinite(value).all() for value in output.values()
                   if torch.is_tensor(value)):
            raise RuntimeError(f'{name} forward contains NaN or Inf')
        expected_features = [[1, 256, 80, 80], [1, 256, 40, 40],
                             [1, 256, 20, 20]]
        if captured['encoder_shapes'] != expected_features:
            raise RuntimeError(f'{name} encoder shapes changed')

        # Encoder-level backward directly proves that every trainable ACR
        # parameter, including both raw betas and downsamplers, is connected.
        model.encoder.train()
        channels = [36, 72, 144] if _is_hrnet(name) else [128, 256, 512]
        inputs = [torch.randn(1, channels[0], 16, 16, requires_grad=True),
                  torch.randn(1, channels[1], 8, 8, requires_grad=True),
                  torch.randn(1, channels[2], 4, 4, requires_grad=True)]
        loss = sum(value.square().mean() for value in model.encoder(inputs))
        loss.backward()
        missing = [parameter_name for parameter_name, parameter
                   in model.encoder.named_parameters()
                   if parameter_name.startswith('acr_')
                   and parameter.requires_grad and parameter.grad is None]
        nonfinite = [parameter_name for parameter_name, parameter
                     in model.encoder.named_parameters()
                     if parameter_name.startswith('acr_')
                     and parameter.grad is not None
                     and not torch.isfinite(parameter.grad).all()]
        if missing or nonfinite:
            raise RuntimeError(
                f'{name} ACR gradient failure: missing={missing}, '
                f'nonfinite={nonfinite}')
        rows = _optimizer_rows(config, model)
        if not rows:
            raise RuntimeError(f'{name} has no ACR optimizer parameters')
        for row in rows:
            expected_decay = 0.0 if '.norm.' in row['name'] else 1e-4
            if row['lr'] != 3e-4 or row['weight_decay'] != expected_decay:
                raise RuntimeError(f'Incorrect ACR optimizer group: {row}')
        result['models'][name] = {
            'forward': 'PASS', 'backward': 'PASS',
            'encoder_shapes': captured['encoder_shapes'],
            'prediction_shapes': {
                key: list(value.shape) for key, value in output.items()
                if torch.is_tensor(value)
            },
        }
        result['optimizer'][name] = rows
        print(f'{name} 640 forward / ACR backward / optimizer: PASS')
        del config, model, output, inputs, loss

    if torch.cuda.is_available():
        _, model = build('PResNet18 + ACR', training=True)
        model.encoder.cuda()
        inputs = [torch.randn(1, 128, 32, 32, device='cuda'),
                  torch.randn(1, 256, 16, 16, device='cuda'),
                  torch.randn(1, 512, 8, 8, device='cuda')]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model.encoder(inputs)
            loss = sum(value.square().mean() for value in output)
        loss.backward()
        if not torch.isfinite(loss) or not all(
                torch.isfinite(value).all() for value in output):
            raise RuntimeError('ACR CUDA AMP produced NaN or Inf')
        result['amp'] = 'PASS'
        print('CUDA AMP forward/backward: PASS')
    else:
        result['amp'] = 'NOT RUN (CUDA unavailable)'
        print('CUDA AMP forward/backward: NOT RUN (CUDA unavailable)')
    return result


def benchmark(warmup, iterations, amp):
    if not torch.cuda.is_available():
        raise RuntimeError('--benchmark requires a CUDA GPU')
    device = torch.device('cuda', 0)
    results = {}
    for name in CONFIGS:
        _, model = build(name)
        model = model.to(device).eval()
        image = torch.randn(1, 3, 640, 640, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(warmup):
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=amp):
                    model(image)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(iterations):
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=amp):
                    model(image)
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0 / iterations
        results[name] = {
            'batch': 1, 'input': [1, 3, 640, 640],
            'amp': amp, 'latency_ms': latency_ms,
            'fps': 1000.0 / latency_ms,
            'peak_memory_mib': torch.cuda.max_memory_allocated(device) / 2**20,
        }
        del model, image
        torch.cuda.empty_cache()
    print(json.dumps({'benchmark': results}, indent=2))
    return results


def ddp_smoke(amp):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 3:
        raise RuntimeError(
            f'--ddp-smoke requires --nproc_per_node=3; got WORLD_SIZE={world_size}')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError('Three visible CUDA GPUs are required for DDP smoke')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    try:
        # ACR lives wholly in the shared HybridEncoder implementation, so one
        # complete ACR detector verifies its DDP reducer/startup semantics.
        _, model = build('PResNet18 + ACR', training=True)
        model = model.cuda(local_rank)
        ddp = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True)
        image = torch.randn(1, 3, 128, 128, device=local_rank)
        targets = [{'labels': torch.tensor([0], device=local_rank),
                    'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]],
                                          device=local_rank)}]
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=amp):
            output = ddp(image, targets)
            loss = (output['pred_logits'].square().mean()
                    + output['pred_boxes'].square().mean())
        loss.backward()
        missing = [name for name, parameter in ddp.module.named_parameters()
                   if name.startswith('encoder.acr_')
                   and parameter.requires_grad and parameter.grad is None]
        if missing:
            raise RuntimeError(f'DDP ACR parameters without gradients: {missing}')
        dist.barrier()
        if local_rank == 0:
            print('Three-GPU NCCL DDP startup/forward/backward: PASS')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--complexity', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--ddp-smoke', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    process_rank = int(os.environ.get('RANK', '0'))
    if not any((args.smoke, args.complexity, args.benchmark, args.ddp_smoke)):
        parser.error('select at least one validation action')
    if args.warmup < 0 or args.iterations < 1:
        parser.error('--warmup must be >= 0 and --iterations must be >= 1')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    report = {}
    if args.smoke:
        report['smoke'] = smoke()
    if args.complexity:
        report['complexity'] = complexity()
    if args.benchmark:
        report['benchmark'] = benchmark(
            args.warmup, args.iterations, args.amp)
    if args.ddp_smoke:
        ddp_smoke(args.amp)
        report['ddp_smoke'] = 'PASS on rank 0; see console'
    if args.output and process_rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
