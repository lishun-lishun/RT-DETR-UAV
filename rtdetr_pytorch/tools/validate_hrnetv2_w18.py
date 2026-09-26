"""Short HRNetV2-W18 validation and complexity utility (never trains data).

Examples:
  python tools/validate_hrnetv2_w18.py --smoke --complexity
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9919 tools/validate_hrnetv2_w18.py --ddp-smoke

Conv/Linear MACs are a reproducible lower bound: normalization, activation,
interpolation, softmax and functional attention operations are not counted.
Reported FLOPs use the common 2 FLOPs per multiply-accumulate convention.
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

from tools.analyze_dut_models import import_model_source


core = import_model_source(selective=True)
importlib.import_module('src.nn.backbone.hrnet')

from src.nn.backbone.hrnet import HRNetV2W18  # noqa: E402


BASELINE = ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml'
HRNET = ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml'


def build(config_path):
    overrides = ({'PResNet': {'pretrained': False}}
                 if config_path == BASELINE else
                 {'HRNetV2W18': {'pretrained': False,
                                 'pretrained_path': None}})
    config = core.YAMLConfig(str(config_path), **overrides)
    model = config.model.eval()
    model.multi_scale = None
    return model


def macs_for(module, image):
    macs = 0

    def hook(layer, inputs, output):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += (output.numel() * (layer.in_channels // layer.groups)
                     * layer.kernel_size[0] * layer.kernel_size[1])
        elif isinstance(layer, nn.Linear):
            macs += output.numel() * layer.in_features

    handles = [layer.register_forward_hook(hook) for layer in module.modules()
               if isinstance(layer, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode():
            output = module(image)
    finally:
        for handle in handles:
            handle.remove()
    return macs, output


def complexity():
    results = {}
    for name, path in (('PResNet18 baseline', BASELINE),
                       ('HRNetV2-W18', HRNET)):
        torch.manual_seed(0)
        model = build(path).cpu()
        image = torch.randn(1, 3, 640, 640)
        backbone_macs, features = macs_for(model.backbone, image)
        whole_macs, output = macs_for(model, image)
        results[name] = {
            'backbone_params': sum(p.numel() for p in model.backbone.parameters()),
            'whole_model_params': sum(p.numel() for p in model.parameters()),
            'backbone_conv_linear_macs': backbone_macs,
            'backbone_conv_linear_flops': 2 * backbone_macs,
            'whole_model_conv_linear_macs': whole_macs,
            'whole_model_conv_linear_flops': 2 * whole_macs,
            'feature_shapes': [list(value.shape) for value in features],
            'prediction_shapes': {key: list(value.shape)
                                  for key, value in output.items()
                                  if torch.is_tensor(value)},
        }
        del model, image, features, output
    print(json.dumps(results, indent=2))


def smoke():
    backbone = HRNetV2W18(pretrained=False).train()
    image = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        features = backbone.eval()(image)
    print('HRNet features:', [tuple(value.shape) for value in features])
    model = build(HRNET)
    with torch.inference_mode():
        output = model(image)
    if not all(torch.isfinite(value).all() for value in output.values()
               if torch.is_tensor(value)):
        raise RuntimeError('Full detector output contains NaN or Inf')
    print('Full detector:', {key: tuple(value.shape)
                             for key, value in output.items()
                             if torch.is_tensor(value)})

    model.train()
    small = torch.randn(1, 3, 128, 128)
    targets = [{'labels': torch.tensor([0]),
                'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]])}]
    training_output = model(small, targets)
    loss = (training_output['pred_logits'].square().mean()
            + training_output['pred_boxes'].square().mean())
    loss.backward()
    missing = [name for name, parameter in model.backbone.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    if missing:
        raise RuntimeError(f'Backbone parameters without gradients: {missing}')
    print('Full detector backward: PASS '
          '(all trainable HRNet parameters have gradients)')

    if torch.cuda.is_available():
        amp_model = HRNetV2W18(pretrained=False).cuda().train()
        amp_image = torch.randn(1, 3, 256, 256, device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            amp_loss = sum(value.square().mean()
                           for value in amp_model(amp_image))
        amp_loss.backward()
        if not torch.isfinite(amp_loss):
            raise RuntimeError('AMP loss is not finite')
        print('CUDA AMP forward/backward: PASS')
    else:
        print('CUDA AMP forward/backward: NOT RUN (CUDA unavailable)')


def ddp_smoke():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 3:
        raise RuntimeError(
            f'--ddp-smoke requires torchrun --nproc_per_node=3; got {world_size}')
    local_rank = int(os.environ['LOCAL_RANK'])
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError('Three visible CUDA GPUs are required for DDP smoke')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    try:
        model = HRNetV2W18(pretrained=False).cuda(local_rank).train()
        ddp = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True)
        image = torch.randn(1, 3, 128, 128, device=local_rank)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = sum(value.square().mean() for value in ddp(image))
        loss.backward()
        missing = [name for name, parameter in ddp.module.named_parameters()
                   if parameter.requires_grad and parameter.grad is None]
        if missing:
            raise RuntimeError(f'DDP parameters without gradients: {missing}')
        dist.barrier()
        if local_rank == 0:
            print('Three-GPU NCCL DDP startup/forward/backward: PASS')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--complexity', action='store_true')
    parser.add_argument('--ddp-smoke', action='store_true')
    args = parser.parse_args()
    if not (args.smoke or args.complexity or args.ddp_smoke):
        parser.error('select at least one validation action')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.smoke:
        smoke()
    if args.complexity:
        complexity()
    if args.ddp_smoke:
        ddp_smoke()


if __name__ == '__main__':
    main()
