"""Validate SLR-Neck without running dataset training.

Examples:
  python tools/validate_slr_neck.py --smoke --complexity \
    --output reports/slr_validation_cpu.json
  CUDA_VISIBLE_DEVICES=1 python tools/validate_slr_neck.py --amp-smoke
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9922 tools/validate_slr_neck.py --ddp-smoke --amp

Conv/Linear MACs are the same reproducible lower bound used by the other
project validators. SLR's functional attention MACs are reported separately.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

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
    'PResNet18 + SLR': ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_slr.yml',
    'HRNetV2-W18 + Original': ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml',
    'HRNetV2-W18 + SLR': ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_slr.yml',
}
SLR_NAMES = ('PResNet18 + SLR', 'HRNetV2-W18 + SLR')


def is_hrnet(name):
    return name.startswith('HRNetV2-W18')


def build(name, training=False):
    overrides = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
                 if is_hrnet(name) else
                 {'PResNet': {'pretrained': False}})
    config = core.YAMLConfig(str(CONFIGS[name]), **overrides)
    model = config.model
    model.multi_scale = None
    model.train(training)
    return config, model


def recursive_tensor_loss(value):
    if torch.is_tensor(value) and value.is_floating_point():
        return value.float().square().mean()
    if isinstance(value, dict):
        terms = [recursive_tensor_loss(child) for child in value.values()]
    elif isinstance(value, (list, tuple)):
        terms = [recursive_tensor_loss(child) for child in value]
    else:
        terms = []
    terms = [term for term in terms if term is not None]
    return sum(terms) if terms else None


def optimizer_rows(config, model):
    assignments = {}
    for group_index, group in enumerate(config.optimizer.param_groups):
        for parameter in group['params']:
            assignments[id(parameter)] = (
                group_index, group['lr'], group['weight_decay'])
    rows = []
    for name, parameter in model.named_parameters():
        if name.startswith('encoder.slr.'):
            group, lr, decay = assignments[id(parameter)]
            rows.append({'name': name, 'group': group,
                         'lr': lr, 'weight_decay': decay})
    return rows


def smoke():
    report = {'models': {}, 'optimizer': {}}
    for name in SLR_NAMES:
        config, model = build(name)
        channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
        detail_channels = 18 if is_hrnet(name) else 64
        dynamic = {}
        model.backbone.eval()
        model.encoder.train()
        for size in (480, 640, 800):
            image = torch.randn(1, 3, size, size)
            with torch.no_grad():
                backbone_output = model.backbone(image)
                encoded = model.encoder(backbone_output)
            shapes = [list(value.shape) for value in encoded]
            expected = [[1, 256, size // 8, size // 8],
                        [1, 256, size // 16, size // 16],
                        [1, 256, size // 32, size // 32]]
            if shapes != expected:
                raise RuntimeError(f'{name} dynamic shape mismatch at {size}: {shapes}')
            detail_shape = list(backbone_output['detail'].shape)
            expected_detail = [1, detail_channels, size // 4, size // 4]
            if detail_shape != expected_detail:
                raise RuntimeError(
                    f'{name} detail mismatch: {detail_shape} != {expected_detail}')
            dynamic[str(size)] = {'detail': detail_shape, 'encoder': shapes}

        model.eval()
        with torch.inference_mode():
            predictions = model(torch.randn(1, 3, 640, 640))
        prediction_shapes = {key: list(value.shape)
                             for key, value in predictions.items()
                             if torch.is_tensor(value)}
        if prediction_shapes != {
                'pred_logits': [1, 300, 1], 'pred_boxes': [1, 300, 4]}:
            raise RuntimeError(f'{name} prediction shapes changed')

        model.encoder.train()
        features = [torch.randn(1, channels[0], 16, 16),
                    torch.randn(1, channels[1], 8, 8),
                    torch.randn(1, channels[2], 4, 4)]
        detail = torch.randn(1, detail_channels, 32, 32)
        outputs = model.encoder({'features': features, 'detail': detail})
        loss = sum(value.square().mean() for value in outputs)
        loss.backward()
        bad_gradients = [parameter_name for parameter_name, parameter
                         in model.encoder.slr.named_parameters()
                         if parameter.grad is None
                         or not torch.isfinite(parameter.grad).all()
                         or parameter.grad.abs().sum() == 0]
        if bad_gradients:
            raise RuntimeError(f'{name} bad SLR gradients: {bad_gradients}')

        rows = optimizer_rows(config, model)
        for row in rows:
            expected_decay = (0.0 if 'bias' in row['name']
                              or '.norm.' in row['name'] else 1e-4)
            if row['lr'] != 3e-4 or row['weight_decay'] != expected_decay:
                raise RuntimeError(f'{name} optimizer mismatch: {row}')
        report['models'][name] = {
            'forward': 'PASS', 'backward': 'PASS',
            'dynamic_shapes': dynamic,
            'prediction_shapes': prediction_shapes,
            'alpha_eff': model.encoder.slr.effective_alpha().item(),
        }
        report['optimizer'][name] = rows
        print(f'{name}: 480/640/800 forward, 640 detector, backward: PASS')
    return report


def conv_linear_macs(model, image):
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
    report = {}
    for name in CONFIGS:
        torch.manual_seed(0)
        _, model = build(name)
        model = model.cpu().eval()
        image = torch.randn(1, 3, 640, 640)
        macs, output = conv_linear_macs(model, image)
        entry = {
            'whole_params': sum(parameter.numel()
                                for parameter in model.parameters()),
            'whole_conv_linear_macs': macs,
            'whole_conv_linear_flops_2_per_mac': 2 * macs,
            'prediction_shapes': {key: list(value.shape)
                                  for key, value in output.items()
                                  if torch.is_tensor(value)},
        }
        if name in SLR_NAMES:
            query_dim = model.encoder.slr.query_dim
            height = width = 640 // 8
            entry['slr_functional_attention_macs'] = (
                (4 * query_dim + 4 * query_dim + 4 * 2)
                * height * width)
        report[name] = entry
        del model, image, output
    for backbone in ('PResNet18', 'HRNetV2-W18'):
        original = report[f'{backbone} + Original']
        candidate = report[f'{backbone} + SLR']
        candidate['added_params_vs_original'] = (
            candidate['whole_params'] - original['whole_params'])
        candidate['added_conv_linear_macs_vs_original'] = (
            candidate['whole_conv_linear_macs']
            - original['whole_conv_linear_macs'])
    print(json.dumps({'complexity': report}, indent=2))
    return report


def check_slr_gradients(model, name):
    bad = [parameter_name for parameter_name, parameter
           in model.encoder.slr.named_parameters()
           if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
    if bad:
        raise RuntimeError(f'{name} missing/nonfinite SLR gradients: {bad}')


def amp_smoke():
    if not torch.cuda.is_available():
        raise RuntimeError('--amp-smoke requires CUDA')
    device = torch.device('cuda', 0)
    for name in SLR_NAMES:
        _, model = build(name, training=True)
        model = model.to(device)
        image = torch.randn(1, 3, 128, 128, device=device)
        targets = [{'labels': torch.tensor([0], device=device),
                    'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device)}]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(image, targets)
            loss = recursive_tensor_loss(output)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f'{name} AMP loss is invalid')
        loss.backward()
        check_slr_gradients(model, name)
        print(f'{name}: CUDA AMP forward/backward: PASS')
        del model, image, output, loss
        torch.cuda.empty_cache()


def ddp_smoke(amp):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 3:
        raise RuntimeError(
            f'--ddp-smoke requires --nproc_per_node=3; got {world_size}')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError('Three visible CUDA GPUs are required for DDP smoke')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    try:
        for name in SLR_NAMES:
            _, model = build(name, training=True)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.cuda(local_rank)
            ddp = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False, gradient_as_bucket_view=True)
            for _ in range(2):
                ddp.zero_grad(set_to_none=True)
                image = torch.randn(1, 3, 128, 128, device=local_rank)
                targets = [{
                    'labels': torch.tensor([0], device=local_rank),
                    'boxes': torch.tensor(
                        [[0.5, 0.5, 0.1, 0.1]], device=local_rank)}]
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=amp):
                    output = ddp(image, targets)
                    loss = recursive_tensor_loss(output)
                loss.backward()
                check_slr_gradients(ddp.module, name)
            dist.barrier()
            if local_rank == 0:
                print(f'{name}: three-GPU SyncBN DDP two iterations: PASS')
            del ddp, model, image, output, loss
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--complexity', action='store_true')
    parser.add_argument('--amp-smoke', action='store_true')
    parser.add_argument('--ddp-smoke', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not any((args.smoke, args.complexity,
                args.amp_smoke, args.ddp_smoke)):
        parser.error('select at least one validation action')
    rank = int(os.environ.get('RANK', '0'))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    report = {}
    if args.smoke:
        report['smoke'] = smoke()
    if args.complexity:
        report['complexity'] = complexity()
    if args.amp_smoke:
        amp_smoke()
        report['amp_smoke'] = 'PASS'
    if args.ddp_smoke:
        ddp_smoke(args.amp)
        report['ddp_smoke'] = 'PASS'
    if args.output and rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
