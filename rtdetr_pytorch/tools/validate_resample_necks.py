"""Validate the independent LPRU/SPDR experiments without dataset training.

CPU validation and reproducible Conv/Linear MAC lower bounds::

  python tools/validate_resample_necks.py --smoke --complexity \
    --output reports/resample_necks_cpu.json

CUDA AMP and strict three-rank DDP smoke tests::

  CUDA_VISIBLE_DEVICES=1 python tools/validate_resample_necks.py --amp-smoke
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9925 tools/validate_resample_necks.py --ddp-smoke --amp

The MAC result counts executed Conv2d/Linear multiply-accumulates.  It omits
PixelShuffle/PixelUnshuffle (data rearrangements), normalization,
interpolation, activation, concatenation and elementwise residual arithmetic.
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
    'PResNet18 + Original':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'HRNetV2-W18 + Original':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml',
    'PResNet18 + LPRU':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_lpru.yml',
    'HRNetV2-W18 + LPRU':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_lpru.yml',
    'PResNet18 + SPDR':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_spdr.yml',
    'HRNetV2-W18 + SPDR':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_spdr.yml',
}
CANDIDATES = (
    'PResNet18 + LPRU',
    'HRNetV2-W18 + LPRU',
    'PResNet18 + SPDR',
    'HRNetV2-W18 + SPDR',
)
REFERENCE_FOR = {
    name: ('HRNetV2-W18 + Original' if name.startswith('HRNet')
           else 'PResNet18 + Original')
    for name in CANDIDATES
}


def is_hrnet(name):
    return name.startswith('HRNetV2-W18')


def method_for(name):
    return 'LPRU' if name.endswith('LPRU') else 'SPDR'


def prefixes_for(name):
    return (('encoder.lpru54.', 'encoder.lpru43.')
            if method_for(name) == 'LPRU' else
            ('encoder.spdr34.', 'encoder.spdr45.'))


def build(name, training=False, method_override=None):
    overrides = ({'HRNetV2W18': {
        'pretrained': False, 'pretrained_path': None}}
        if is_hrnet(name) else {'PResNet': {'pretrained': False}})
    if method_override is not None:
        overrides[method_for(name)] = method_override
    config = core.YAMLConfig(str(CONFIGS[name]), **overrides)
    model = config.model
    model.multi_scale = None
    model.train(training)
    return config, model


def recursive_tensor_loss(value):
    if torch.is_tensor(value) and value.is_floating_point():
        return value.float().square().mean()
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        children = ()
    terms = [recursive_tensor_loss(child) for child in children]
    terms = [term for term in terms if term is not None]
    return sum(terms) if terms else None


def tensor_lists_close(left, right):
    return (len(left) == len(right)
            and all(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                    for a, b in zip(left, right)))


def new_parameters(model, name):
    prefixes = prefixes_for(name)
    return [(parameter_name, parameter)
            for parameter_name, parameter in model.named_parameters()
            if parameter_name.startswith(prefixes)]


def check_new_gradients(model, name, require_nonzero):
    parameters = new_parameters(model, name)
    if not parameters:
        raise RuntimeError(f'{name}: no new Neck parameters found')
    problems = []
    for parameter_name, parameter in parameters:
        if parameter.grad is None:
            problems.append(parameter_name + ':missing')
        elif not torch.isfinite(parameter.grad).all():
            problems.append(parameter_name + ':nonfinite')
        elif require_nonzero and parameter.grad.abs().sum().item() == 0.0:
            problems.append(parameter_name + ':zero')
    if problems:
        raise RuntimeError(f'{name}: invalid gradients: {problems}')


def fairness_audit():
    expected = {
        'LPRU': {'enabled': True, 'alpha_max': 0.5, 'alpha_init': 0.05},
        'SPDR': {'enabled': True, 'beta_max': 0.5, 'beta_init': 0.05},
    }
    result = {}
    for name in CANDIDATES:
        method = method_for(name)
        reference_name = REFERENCE_FOR[name]
        reference = fresh_config(CONFIGS[reference_name])
        candidate = fresh_config(CONFIGS[name])
        diff = differences(reference, candidate)
        unexpected = [key for key in diff if not (
            key in ('__include__', 'output_dir', method)
            or key.startswith(method + '.'))]
        if unexpected:
            raise RuntimeError(
                f'FAIRNESS ERROR: {name} differs at {unexpected}')
        if candidate.get(method) != expected[method]:
            raise RuntimeError(
                f'FAIRNESS ERROR: {name} {method}={candidate.get(method)!r}; '
                f'expected {expected[method]!r}')
        result[name] = {
            'reference': reference_name,
            'allowed_namespace': method,
            'resolved_differences': diff,
            'fixed_options': expected[method],
            'status': 'PASS',
        }
    print('Resolved-config fairness (four candidate/reference pairs): PASS')
    return result


def final_config_set_audit():
    actual = {path.name for path in (ROOT / 'configs/rtdetr').glob(
        'rtdetr*_dut_anti_uav*.yml')}
    expected = {path.name for path in CONFIGS.values()}
    if actual != expected:
        raise RuntimeError(
            'Final top-level DUT config set is not exactly six: '
            f'extra={sorted(actual - expected)}, missing={sorted(expected - actual)}')
    return {'status': 'PASS', 'files': sorted(actual)}


def disabled_equivalence():
    image = torch.randn(1, 3, 128, 128)
    result = {}
    for name in CANDIDATES:
        reference_name = REFERENCE_FOR[name]
        torch.manual_seed(17)
        _, reference = build(reference_name)
        torch.manual_seed(17)
        _, disabled = build(name, method_override={'enabled': False})
        if disabled.encoder.lpru_enabled or disabled.encoder.spdr_enabled:
            raise RuntimeError(f'{name}: disabled Neck is still enabled')
        forbidden = ('lpru54', 'lpru43', 'spdr34', 'spdr45')
        if any(hasattr(disabled.encoder, attr) for attr in forbidden):
            raise RuntimeError(f'{name}: disabled Neck was still constructed')
        reference_state = reference.state_dict()
        disabled_state = disabled.state_dict()
        if reference_state.keys() != disabled_state.keys():
            raise RuntimeError(f'{name}: disabled state keys differ')
        changed = [key for key in reference_state
                   if not torch.equal(reference_state[key], disabled_state[key])]
        if changed:
            raise RuntimeError(
                f'{name}: disabled seeded weights differ: {changed[:10]}')
        disabled.load_state_dict(reference_state, strict=True)
        for model in (reference, disabled):
            model.eval()
            model.encoder.eval_spatial_size = None
            model.decoder.eval_spatial_size = None
        with torch.inference_mode():
            ref_backbone = reference.backbone(image)
            new_backbone = disabled.backbone(image)
            ref_neck = reference.encoder(ref_backbone)
            new_neck = disabled.encoder(new_backbone)
            ref_prediction = reference(image)
            new_prediction = disabled(image)
        if not tensor_lists_close(ref_backbone, new_backbone):
            raise RuntimeError(f'{name}: backbone equivalence failed')
        if not tensor_lists_close(ref_neck, new_neck):
            raise RuntimeError(f'{name}: N3/N4/N5 equivalence failed')
        for key in ('pred_logits', 'pred_boxes'):
            if not torch.allclose(ref_prediction[key], new_prediction[key],
                                  atol=1e-6, rtol=1e-5):
                raise RuntimeError(f'{name}: {key} equivalence failed')
        result[name] = {
            'state_dict': 'EXACT',
            'backbone': 'PASS',
            'N3_N4_N5': 'PASS',
            'pred_logits': 'PASS',
            'pred_boxes': 'PASS',
        }
        print(f'{name}: disabled original equivalence: PASS')
        del reference, disabled
    return result


def common_seeded_weights(name):
    torch.manual_seed(31)
    _, reference = build(REFERENCE_FOR[name])
    torch.manual_seed(31)
    _, candidate = build(name)
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    common = set(reference_state).intersection(candidate_state)
    if common != set(reference_state):
        raise RuntimeError(
            f'{name}: removed keys {sorted(set(reference_state) - common)}')
    changed = [key for key in sorted(common)
               if not torch.equal(reference_state[key], candidate_state[key])]
    if changed:
        raise RuntimeError(
            f'{name}: shifted original seeded weights: {changed[:10]}')
    return {
        'status': 'PASS',
        'common_tensors': len(common),
        'added_tensors': sorted(set(candidate_state) - set(reference_state)),
    }


def dynamic_detector_forward(model, name):
    model.eval()
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    captured = []
    handle = model.encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.append(
            [list(value.shape) for value in output]))
    result = {}
    try:
        for size in (480, 640, 800):
            image = torch.randn(1, 3, size, size)
            with torch.inference_mode():
                output = model(image)
            expected_neck = [
                [1, 256, size // 8, size // 8],
                [1, 256, size // 16, size // 16],
                [1, 256, size // 32, size // 32],
            ]
            if captured[-1] != expected_neck:
                raise RuntimeError(
                    f'{name}@{size}: Neck shapes {captured[-1]}')
            prediction = {key: list(output[key].shape)
                          for key in ('pred_logits', 'pred_boxes')}
            expected_prediction = {
                'pred_logits': [1, 300, 1],
                'pred_boxes': [1, 300, 4],
            }
            if prediction != expected_prediction:
                raise RuntimeError(
                    f'{name}@{size}: prediction shapes {prediction}')
            if not all(torch.isfinite(value).all()
                       for value in output.values() if torch.is_tensor(value)):
                raise RuntimeError(f'{name}@{size}: NaN/Inf output')
            result[str(size)] = {
                'input': [1, 3, size, size],
                'N3_N4_N5': captured[-1],
                **prediction,
                'status': 'PASS',
            }
    finally:
        handle.remove()
    return result


def synthetic_backward(model, name):
    channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
    model.zero_grad(set_to_none=True)
    model.encoder.train()
    model.encoder.eval_spatial_size = None
    features = [
        torch.randn(2, channels[0], 16, 16, requires_grad=True),
        torch.randn(2, channels[1], 8, 8, requires_grad=True),
        torch.randn(2, channels[2], 4, 4, requires_grad=True),
    ]
    output = model.encoder(features)
    loss = sum(value.float().square().mean() for value in output)
    if not torch.isfinite(loss):
        raise RuntimeError(f'{name}: nonfinite synthetic loss')
    loss.backward()
    check_new_gradients(model, name, require_nonzero=True)
    return {
        'loss': loss.detach().item(),
        'N3_N4_N5': [list(value.shape) for value in output],
        'new_parameter_gradients': 'ALL PRESENT, FINITE, NONZERO',
    }


def optimizer_rows(config, model, name):
    assignment = {}
    for group_index, group in enumerate(config.optimizer.param_groups):
        for parameter in group['params']:
            assignment[id(parameter)] = (
                group_index, group['lr'], group['weight_decay'])
    rows = []
    for parameter_name, parameter in new_parameters(model, name):
        if id(parameter) not in assignment:
            raise RuntimeError(
                f'{name}: missing optimizer parameter {parameter_name}')
        group, lr, weight_decay = assignment[id(parameter)]
        expected_decay = 0.0 if parameter_name.endswith('.bias') else 1e-4
        row = {
            'parameter': parameter_name,
            'optimizer_group': group,
            'lr': lr,
            'weight_decay': weight_decay,
        }
        if lr != 3e-4 or weight_decay != expected_decay:
            raise RuntimeError(f'{name}: optimizer mismatch {row}')
        rows.append(row)
    if not rows:
        raise RuntimeError(f'{name}: no new optimizer parameters')
    return rows


def smoke():
    report = {
        'final_config_set': final_config_set_audit(),
        'fairness': fairness_audit(),
        'disabled_equivalence': disabled_equivalence(),
        'models': {},
        'optimizer': {},
        'common_seeded_weights': {},
    }
    for name in CANDIDATES:
        config, model = build(name)
        report['models'][name] = {
            'dynamic_detector_forward': dynamic_detector_forward(model, name),
            'backward': synthetic_backward(model, name),
        }
        rows = optimizer_rows(config, model, name)
        report['optimizer'][name] = rows
        del config, model
        report['common_seeded_weights'][name] = common_seeded_weights(name)
        print(f'{name}: 480/640/800 detector, backward, optimizer, '
              'common weights: PASS')
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
            'omitted_ops': (
                'PixelShuffle/PixelUnshuffle/normalization/interpolation/'
                'activation/concat/elementwise and other non-Conv/Linear ops'),
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


def training_sample(device):
    return (
        torch.randn(1, 3, 128, 128, device=device),
        [{
            'labels': torch.tensor([0], device=device),
            'boxes': torch.tensor(
                [[0.5, 0.5, 0.1, 0.1]], device=device),
        }],
    )


def amp_smoke():
    if not torch.cuda.is_available():
        raise RuntimeError('--amp-smoke requires CUDA')
    device = torch.device('cuda', 0)
    report = {}
    for name in CANDIDATES:
        _, model = build(name, training=True)
        model = model.to(device)
        image, targets = training_sample(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(image, targets)
            loss = recursive_tensor_loss(output)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f'{name}: invalid AMP loss')
        loss.backward()
        check_new_gradients(model, name, require_nonzero=False)
        if not all(torch.isfinite(value).all()
                   for value in output.values() if torch.is_tensor(value)):
            raise RuntimeError(f'{name}: AMP output contains NaN/Inf')
        report[name] = {'status': 'PASS', 'loss': loss.detach().item()}
        print(f'{name}: CUDA AMP full forward/backward: PASS')
        del model, image, output, loss
        torch.cuda.empty_cache()
    return report


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
    report = {}
    try:
        for name in CANDIDATES:
            device = torch.device('cuda', local_rank)
            _, model = build(name, training=True)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
            ddp = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False, gradient_as_bucket_view=True)
            losses = []
            for _step in range(2):
                ddp.zero_grad(set_to_none=True)
                image, targets = training_sample(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=amp):
                    output = ddp(image, targets)
                    loss = recursive_tensor_loss(output)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f'{name}: invalid DDP loss')
                loss.backward()
                check_new_gradients(ddp.module, name, require_nonzero=False)
                losses.append(loss.detach().item())
            sentinel = torch.ones((), device=device)
            dist.all_reduce(sentinel)
            if sentinel.item() != world_size:
                raise RuntimeError(f'{name}: DDP all-reduce failed')
            report[name] = {
                'status': 'PASS', 'steps': 2, 'losses': losses,
                'world_size': world_size, 'amp': amp,
                'find_unused_parameters': False,
            }
            if local_rank == 0:
                print(f'{name}: 3-GPU DDP two-step forward/backward: PASS')
            del ddp, model, image, output, loss
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        dist.destroy_process_group()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--complexity', action='store_true')
    parser.add_argument('--amp-smoke', action='store_true')
    parser.add_argument('--ddp-smoke', action='store_true')
    parser.add_argument('--amp', action='store_true',
                        help='Enable autocast in --ddp-smoke')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not any((args.smoke, args.complexity,
                args.amp_smoke, args.ddp_smoke)):
        parser.error('select at least one validation action')

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    report = {
        'protocol': {
            'candidate_order': list(CANDIDATES),
            'pretrained_disabled_for_smoke_only': True,
            'formal_training_launched': False,
        },
    }
    if args.smoke:
        report['smoke'] = smoke()
    if args.complexity:
        report['complexity'] = complexity()
    if args.amp_smoke:
        report['amp_smoke'] = amp_smoke()
    if args.ddp_smoke:
        report['ddp_smoke'] = ddp_smoke(args.amp)

    rank = int(os.environ.get('RANK', '0'))
    if args.output and rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        print(f'Saved {args.output.resolve()}')


if __name__ == '__main__':
    main()
