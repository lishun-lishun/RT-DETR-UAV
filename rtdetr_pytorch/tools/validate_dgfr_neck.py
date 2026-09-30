"""Validate DGFR-Neck without running dataset training.

Examples:
  python tools/validate_dgfr_neck.py --smoke --complexity \
    --output reports/dgfr_validation_cpu.json
  CUDA_VISIBLE_DEVICES=1 python tools/validate_dgfr_neck.py --amp-smoke
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9924 tools/validate_dgfr_neck.py --ddp-smoke --amp

The complexity result is a reproducible Conv/Linear MAC lower bound. It omits
normalization, interpolation, activations, concatenation, LayerScale residual
arithmetic, and other functional operators.
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

from tools.analyze_dut_models import (  # noqa: E402
    differences,
    fresh_config,
    import_model_source,
)


core = import_model_source(selective=True)
importlib.import_module('src.nn.backbone.hrnet')
importlib.import_module('src.optim')


CONFIGS = {
    'PResNet18 + Original': (
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml'),
    'PResNet18 + DGFR': (
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_dgfr.yml'),
    'HRNetV2-W18 + Original': (
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml'),
    'HRNetV2-W18 + DGFR': (
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_dgfr.yml'),
}
CANDIDATES = ('PResNet18 + DGFR', 'HRNetV2-W18 + DGFR')
REFERENCE_FOR = {
    'PResNet18 + DGFR': 'PResNet18 + Original',
    'HRNetV2-W18 + DGFR': 'HRNetV2-W18 + Original',
}
DEBUG_KEYS = {
    *(f'gamma{level}_{stat}' for level in (3, 4, 5)
      for stat in ('mean', 'min', 'max')),
    *(f'e{level}_to_o{level}_norm_ratio' for level in (3, 4, 5)),
    *(f'y{level}_norm' for level in (3, 4, 5)),
}


def is_hrnet(name):
    return name.startswith('HRNetV2-W18')


def build(name, training=False, dgfr_override=None):
    overrides = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
                 if is_hrnet(name) else
                 {'PResNet': {'pretrained': False}})
    if dgfr_override is not None:
        overrides['DGFR'] = dgfr_override
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


def tensors_close(expected, actual):
    return (len(expected) == len(actual)
            and all(torch.allclose(left, right, atol=1e-6, rtol=1e-5)
                    for left, right in zip(expected, actual)))


def fairness_audit():
    report = {}
    expected_options = {
        'enabled': True,
        'fusion_channels': 64,
        'gamma_max': 0.25,
        'gamma_init': 0.05,
    }
    for name in CANDIDATES:
        reference_name = REFERENCE_FOR[name]
        reference = fresh_config(CONFIGS[reference_name])
        candidate = fresh_config(CONFIGS[name])
        diff = differences(reference, candidate)
        unexpected = [key for key in diff if not (
            key in ('__include__', 'output_dir', 'DGFR')
            or key.startswith('DGFR.'))]
        if unexpected:
            raise RuntimeError(
                f'FAIRNESS ERROR: {name} differs at {unexpected}')
        configured = candidate.get('DGFR', {})
        for key, expected in expected_options.items():
            if configured.get(key) != expected:
                raise RuntimeError(
                    f'FAIRNESS ERROR: {name} DGFR.{key}='
                    f'{configured.get(key)!r}, expected {expected!r}')
        report[name] = {
            'reference': reference_name,
            'allowed_namespace': 'DGFR',
            'resolved_differences': diff,
            'fixed_options': expected_options,
            'status': 'PASS',
        }
    print('Resolved-config fairness (two candidate/reference pairs): PASS')
    return report


def disabled_equivalence():
    report = {}
    image = torch.randn(1, 3, 128, 128)
    for candidate_name in CANDIDATES:
        reference_name = REFERENCE_FOR[candidate_name]
        torch.manual_seed(17)
        _, reference = build(reference_name)
        torch.manual_seed(17)
        _, disabled = build(
            candidate_name, dgfr_override={'enabled': False})
        if disabled.encoder.dgfr_enabled or hasattr(disabled.encoder, 'dgfr'):
            raise RuntimeError(
                f'{candidate_name}: disabled DGFR was still constructed')
        reference_state = reference.state_dict()
        disabled_state = disabled.state_dict()
        if reference_state.keys() != disabled_state.keys():
            raise RuntimeError(
                f'{candidate_name}: disabled state keys differ from original')
        changed = [key for key in reference_state
                   if not torch.equal(reference_state[key], disabled_state[key])]
        if changed:
            raise RuntimeError(
                f'{candidate_name}: disabled seeded weights differ: {changed[:10]}')
        disabled.load_state_dict(reference_state, strict=True)
        for model in (reference, disabled):
            model.eval()
            model.encoder.eval_spatial_size = None
            model.decoder.eval_spatial_size = None
        with torch.inference_mode():
            reference_backbone = reference.backbone(image)
            disabled_backbone = disabled.backbone(image)
            reference_neck = reference.encoder(reference_backbone)
            disabled_neck = disabled.encoder(disabled_backbone)
            reference_prediction = reference(image)
            disabled_prediction = disabled(image)
        if not tensors_close(reference_backbone, disabled_backbone):
            raise RuntimeError(f'{candidate_name}: backbone equivalence failed')
        if not tensors_close(reference_neck, disabled_neck):
            raise RuntimeError(f'{candidate_name}: O3/O4/O5 equivalence failed')
        for key in ('pred_logits', 'pred_boxes'):
            if not torch.allclose(
                    reference_prediction[key], disabled_prediction[key],
                    atol=1e-6, rtol=1e-5):
                raise RuntimeError(
                    f'{candidate_name}: disabled {key} equivalence failed')
        report[candidate_name] = {
            'state_dict': 'EXACT',
            'backbone': 'PASS',
            'O3_O4_O5': 'PASS',
            'pred_logits': 'PASS',
            'pred_boxes': 'PASS',
        }
        print(f'{candidate_name}: DGFR-disabled baseline equivalence: PASS')
        del reference, disabled
    return report


def common_seeded_weights_audit(name):
    reference_name = REFERENCE_FOR[name]
    torch.manual_seed(31)
    _, reference = build(reference_name)
    reference_state = reference.state_dict()
    torch.manual_seed(31)
    _, candidate = build(name)
    candidate_state = candidate.state_dict()
    common = set(reference_state).intersection(candidate_state)
    if common != set(reference_state):
        missing = sorted(set(reference_state) - common)
        raise RuntimeError(f'{name} removed original state keys: {missing}')
    changed = [key for key in sorted(common)
               if not torch.equal(reference_state[key], candidate_state[key])]
    if changed:
        raise RuntimeError(
            f'{name} shifted common seeded weights: {changed[:10]}')
    added = sorted(set(candidate_state) - set(reference_state))
    del reference, candidate, reference_state, candidate_state
    return {'status': 'PASS', 'common_tensors': len(common), 'added': added}


def dynamic_shapes(model, name):
    result = {}
    model.backbone.eval()
    model.encoder.eval()
    saved_eval_size = model.encoder.eval_spatial_size
    model.encoder.eval_spatial_size = None
    try:
        for size in (480, 640, 800):
            image = torch.randn(1, 3, size, size)
            with torch.inference_mode():
                backbone_output = model.backbone(image)
                encoded = model.encoder(backbone_output)
            backbone_shapes = [list(value.shape) for value in backbone_output]
            encoder_shapes = [list(value.shape) for value in encoded]
            expected_encoder = [
                [1, 256, size // 8, size // 8],
                [1, 256, size // 16, size // 16],
                [1, 256, size // 32, size // 32],
            ]
            if encoder_shapes != expected_encoder:
                raise RuntimeError(
                    f'{name} dynamic mismatch at {size}: {encoder_shapes}')
            if not all(torch.isfinite(value).all() for value in encoded):
                raise RuntimeError(f'{name} nonfinite dynamic output at {size}')
            result[str(size)] = {
                'backbone': backbone_shapes,
                'encoder': encoder_shapes,
            }
            del image, backbone_output, encoded
    finally:
        model.encoder.eval_spatial_size = saved_eval_size
    return result


def detector_640(model, name):
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 640, 640))
    shapes = {key: list(value.shape) for key, value in output.items()
              if torch.is_tensor(value)}
    expected = {'pred_logits': [1, 300, 1], 'pred_boxes': [1, 300, 4]}
    if shapes != expected:
        raise RuntimeError(f'{name} detector shapes changed: {shapes}')
    if not all(torch.isfinite(value).all() for value in output.values()
               if torch.is_tensor(value)):
        raise RuntimeError(f'{name} detector output contains NaN/Inf')
    return shapes


def check_dgfr_gradients(model, name, require_nonzero=False):
    parameters = [(parameter_name, parameter)
                  for parameter_name, parameter
                  in model.named_parameters()
                  if parameter_name.startswith('encoder.dgfr.')]
    if not parameters:
        raise RuntimeError(f'{name} has no DGFR parameters')
    problems = []
    for parameter_name, parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            problems.append(parameter_name + ': missing')
        elif not torch.isfinite(gradient).all():
            problems.append(parameter_name + ': nonfinite')
        elif require_nonzero and gradient.abs().sum().item() == 0.0:
            problems.append(parameter_name + ': zero')
    if problems:
        raise RuntimeError(f'{name} invalid DGFR gradients: {problems}')


def debug_rows(dgfr):
    if set(dgfr.last_debug_stats) != DEBUG_KEYS:
        raise RuntimeError(
            f'DGFR debug keys differ: {sorted(dgfr.last_debug_stats)}')
    result = {}
    for key, value in dgfr.last_debug_stats.items():
        if value.requires_grad or not torch.isfinite(value).all():
            raise RuntimeError(f'Invalid detached debug statistic: {key}')
        result[key] = value.detach().float().mean().item()
    return result


def synthetic_encoder_backward(model, name):
    channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
    model.zero_grad(set_to_none=True)
    model.encoder.train()
    model.encoder.eval_spatial_size = None
    model.encoder.dgfr.debug = True
    model.encoder.dgfr.debug_interval = 1
    features = [
        torch.randn(2, channels[0], 16, 16, requires_grad=True),
        torch.randn(2, channels[1], 8, 8, requires_grad=True),
        torch.randn(2, channels[2], 4, 4, requires_grad=True),
    ]
    output = model.encoder(features)
    loss = sum(value.float().square().mean() for value in output)
    if not torch.isfinite(loss):
        raise RuntimeError(f'{name} synthetic loss is NaN/Inf')
    loss.backward()
    check_dgfr_gradients(model, name, require_nonzero=True)
    gammas = model.encoder.dgfr.effective_gamma()
    return {
        'loss': loss.detach().item(),
        'new_parameter_gradients': 'PASS',
        'encoder_shapes': [list(value.shape) for value in output],
        'effective_gamma': [
            {'mean': value.detach().mean().item(),
             'min': value.detach().min().item(),
             'max': value.detach().max().item()}
            for value in gammas
        ],
        'debug_statistics': debug_rows(model.encoder.dgfr),
    }


def optimizer_rows(config, model, name):
    assignment = {}
    for group_index, group in enumerate(config.optimizer.param_groups):
        for parameter in group['params']:
            assignment[id(parameter)] = (
                group_index, group['lr'], group['weight_decay'])
    rows = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter_name.startswith('encoder.dgfr.'):
            continue
        if id(parameter) not in assignment:
            raise RuntimeError(
                f'{name} parameter missing from optimizer: {parameter_name}')
        group, lr, weight_decay = assignment[id(parameter)]
        row = {
            'parameter': parameter_name,
            'optimizer_group': group,
            'lr': lr,
            'weight_decay': weight_decay,
        }
        expected_decay = (0.0 if parameter_name.endswith('.bias')
                          or '.norm.' in parameter_name else 1e-4)
        if lr != 3e-4 or weight_decay != expected_decay:
            raise RuntimeError(f'{name} optimizer mismatch: {row}')
        rows.append(row)
    if not rows:
        raise RuntimeError(f'{name} has no optimizer rows')
    return rows


def smoke():
    report = {
        'fairness': fairness_audit(),
        'disabled_equivalence': disabled_equivalence(),
        'models': {},
        'optimizer': {},
        'common_seeded_weights': {},
    }
    for name in CANDIDATES:
        config, model = build(name)
        dynamic = dynamic_shapes(model, name)
        prediction_shapes = detector_640(model, name)
        backward = synthetic_encoder_backward(model, name)
        rows = optimizer_rows(config, model, name)
        report['models'][name] = {
            'dynamic_shapes': dynamic,
            'prediction_shapes': prediction_shapes,
            'synthetic_backward': backward,
        }
        report['optimizer'][name] = rows
        del config, model
        report['common_seeded_weights'][name] = common_seeded_weights_audit(name)
        print(f'{name}: 480/640/800 backbone+encoder, 640 detector, '
              'backward, debug, optimizer, common weights: PASS')
        for row in rows:
            print('  {parameter} | group={optimizer_group} | lr={lr} | '
                  'weight_decay={weight_decay}'.format(**row))
    return report


def conv_linear_macs(model, image):
    total = 0

    def hook(layer, _inputs, output):
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
        report[name] = {
            'whole_params': sum(parameter.numel()
                                for parameter in model.parameters()),
            'whole_conv_linear_macs_lower_bound': macs,
            'whole_conv_linear_flops_lower_bound_2_per_mac': 2 * macs,
            'prediction_shapes': {
                key: list(value.shape) for key, value in output.items()
                if torch.is_tensor(value)
            },
            'omitted_functional_ops': (
                'normalization/interpolation/activation/concat/LayerScale/'
                'elementwise and other non-Conv/Linear operators'),
        }
        del model, image, output
    for name in CANDIDATES:
        reference = report[REFERENCE_FOR[name]]
        candidate = report[name]
        candidate['added_params_vs_original'] = (
            candidate['whole_params'] - reference['whole_params'])
        candidate['added_conv_linear_macs_vs_original'] = (
            candidate['whole_conv_linear_macs_lower_bound']
            - reference['whole_conv_linear_macs_lower_bound'])
    print(json.dumps({'complexity': report}, indent=2))
    return report


def amp_smoke():
    if not torch.cuda.is_available():
        raise RuntimeError('--amp-smoke requires CUDA')
    device = torch.device('cuda', 0)
    for name in CANDIDATES:
        _, model = build(name, training=True)
        model = model.to(device)
        image = torch.randn(1, 3, 128, 128, device=device)
        targets = [{
            'labels': torch.tensor([0], device=device),
            'boxes': torch.tensor(
                [[0.5, 0.5, 0.1, 0.1]], device=device),
        }]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(image, targets)
            loss = recursive_tensor_loss(output)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f'{name} AMP loss is invalid')
        loss.backward()
        check_dgfr_gradients(model, name)
        if not all(torch.isfinite(value).all() for value in output.values()
                   if torch.is_tensor(value)):
            raise RuntimeError(f'{name} AMP output contains NaN/Inf')
        print(f'{name}: CUDA AMP full forward/backward: PASS')
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
        for name in CANDIDATES:
            _, model = build(name, training=True)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.cuda(local_rank)
            ddp = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False, gradient_as_bucket_view=True)
            for step in range(2):
                ddp.zero_grad(set_to_none=True)
                image = torch.randn(1, 3, 128, 128, device=local_rank)
                targets = [{
                    'labels': torch.tensor([0], device=local_rank),
                    'boxes': torch.tensor(
                        [[0.5, 0.5, 0.1, 0.1]], device=local_rank),
                }]
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=amp):
                    output = ddp(image, targets)
                    loss = recursive_tensor_loss(output)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(
                        f'{name} DDP step {step} has invalid loss')
                loss.backward()
                check_dgfr_gradients(ddp.module, name)
            dist.barrier()
            if local_rank == 0:
                print(f'{name}: three-GPU DDP two iterations: PASS')
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
