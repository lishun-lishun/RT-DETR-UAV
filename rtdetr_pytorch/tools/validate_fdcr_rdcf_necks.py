"""Audit the independent post-CCFF FDCR and RDCF experiments.

This tool never constructs a dataset, optimizer step, scheduler, checkpoint,
or formal training loop.  It only builds models from the real YAML files and
runs deterministic synthetic checks.

Resource-conscious CPU smoke (useful on a development machine)::

  python tools/validate_fdcr_rdcf_necks.py --smoke --resolutions 128 \
    --output reports/fdcr_rdcf_cpu.json

Required dynamic detector resolutions and 640x640 complexity audit::

  python tools/validate_fdcr_rdcf_necks.py --smoke --resolutions 480 640 800 \
    --complexity --output reports/fdcr_rdcf_cpu.json

CUDA AMP and strict three-rank DDP entry points::

  CUDA_VISIBLE_DEVICES=1 python tools/validate_fdcr_rdcf_necks.py --amp-smoke
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9926 tools/validate_fdcr_rdcf_necks.py --ddp-smoke --amp

The MAC audit reports executed Conv2d/Linear multiply-accumulates.  It is a
reproducible lower bound: normalization, pooling, activation, interpolation,
concatenation, elementwise residual arithmetic and other non-Conv/Linear
operations are deliberately excluded.
"""

import argparse
import copy
import gc
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


# Selective loading avoids importing dataset/evaluator packages such as
# pycocotools.  These audits use only the real model/config/optimizer source.
core = import_model_source(selective=True)
importlib.import_module('src.nn.backbone.hrnet')
importlib.import_module('src.optim')


CONFIGS = {
    'PResNet18 + Original':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'HRNetV2-W18 + Original':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml',
    'PResNet18 + FDCR':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_fdcr.yml',
    'HRNetV2-W18 + FDCR':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_fdcr.yml',
    'PResNet18 + RDCF':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_rdcf.yml',
    'HRNetV2-W18 + RDCF':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_rdcf.yml',
}

SPDR_CONFIGS = {
    ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_spdr.yml',
    ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_spdr.yml',
}

RETAINED_SPDR = {
    'PResNet18 + SPDR':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_spdr.yml',
    'HRNetV2-W18 + SPDR':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_spdr.yml',
}

CANDIDATES = (
    'PResNet18 + FDCR',
    'HRNetV2-W18 + FDCR',
    'PResNet18 + RDCF',
    'HRNetV2-W18 + RDCF',
)

REFERENCE_FOR = {
    name: ('HRNetV2-W18 + Original' if name.startswith('HRNet')
           else 'PResNet18 + Original')
    for name in CANDIDATES
}

EXPECTED_OPTIONS = {
    'FDCR': {
        'enabled': True,
        'gamma_max': 0.30,
        'gamma_init': 0.05,
    },
    'RDCF': {
        'enabled': True,
        'eta_max': 0.30,
        'eta_init': 0.05,
        'deploy': False,
    },
}


def is_hrnet(name):
    return name.startswith('HRNetV2-W18')


def method_for(name):
    if name.endswith('FDCR'):
        return 'FDCR'
    if name.endswith('RDCF'):
        return 'RDCF'
    raise ValueError(f'{name!r} is not an FDCR/RDCF candidate')


def prefixes_for(name):
    method = method_for(name).lower()
    return (f'encoder.{method}3.', f'encoder.{method}4.')


def backbone_overrides(name):
    if is_hrnet(name):
        return {'HRNetV2W18': {
            'pretrained': False,
            'pretrained_path': None,
        }}
    return {'PResNet': {'pretrained': False}}


def build(name, training=False, method_override=None):
    """Build one real detector without downloading pretrained weights."""
    overrides = backbone_overrides(name)
    if method_override is not None:
        overrides[method_for(name)] = copy.deepcopy(method_override)
    config = core.YAMLConfig(str(CONFIGS[name]), **overrides)
    model = config.model
    model.multi_scale = None
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    model.train(training)
    return config, model


def build_from_path(name, path, training=False):
    """Build a retained non-candidate model for regression smoke only."""
    config = core.YAMLConfig(str(path), **backbone_overrides(name))
    model = config.model
    model.multi_scale = None
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    model.train(training)
    return config, model


def new_parameters(model, name):
    prefixes = prefixes_for(name)
    return [(parameter_name, parameter)
            for parameter_name, parameter in model.named_parameters()
            if parameter_name.startswith(prefixes)]


def assert_allclose(left, right, context, atol=1e-6, rtol=1e-5):
    if left.shape != right.shape or not torch.allclose(
            left, right, atol=atol, rtol=rtol):
        maximum = (left.float() - right.float()).abs().max().item()
        raise RuntimeError(
            f'{context}: tensors differ (max_abs={maximum}, '
            f'atol={atol}, rtol={rtol})')


def assert_tensor_lists_close(left, right, context,
                              atol=1e-6, rtol=1e-5):
    if len(left) != len(right):
        raise RuntimeError(
            f'{context}: tensor-list lengths differ: {len(left)} != '
            f'{len(right)}')
    for index, (expected, actual) in enumerate(zip(left, right)):
        assert_allclose(
            expected, actual, f'{context}/level-{index}', atol, rtol)


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


def check_new_gradients(model, name, require_nonzero=True):
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


def final_config_set_audit():
    actual = {path.name for path in (ROOT / 'configs/rtdetr').glob(
        'rtdetr*_dut_anti_uav*.yml')}
    expected = {path.name for path in CONFIGS.values()} | {
        path.name for path in SPDR_CONFIGS
    }
    if actual != expected:
        raise RuntimeError(
            'Final top-level DUT config set is not exactly the two '
            'baselines, two retained SPDR configs and four new candidates: '
            f'extra={sorted(actual - expected)}, '
            f'missing={sorted(expected - actual)}')
    return {'status': 'PASS', 'files': sorted(actual)}


def candidate_activation_audit():
    result = {}
    for name in CANDIDATES:
        method = method_for(name)
        config, model = build(name)
        encoder = model.encoder
        expected_fdcr = method == 'FDCR'
        expected_rdcf = method == 'RDCF'
        if encoder.fdcr_enabled != expected_fdcr:
            raise RuntimeError(f'{name}: FDCR enabled state is wrong')
        if encoder.rdcf_enabled != expected_rdcf:
            raise RuntimeError(f'{name}: RDCF enabled state is wrong')
        if encoder.spdr_enabled:
            raise RuntimeError(f'{name}: SPDR must be disabled')
        for prefix in ('fdcr', 'rdcf'):
            expected = prefix.upper() == method
            for level in (3, 4):
                if hasattr(encoder, f'{prefix}{level}') != expected:
                    raise RuntimeError(
                        f'{name}: incorrect {prefix}{level} construction')
        first = getattr(encoder, method.lower() + '3')
        second = getattr(encoder, method.lower() + '4')
        if first is second or not set(map(id, first.parameters())).isdisjoint(
                set(map(id, second.parameters()))):
            raise RuntimeError(f'{name}: N3/N4 blocks share parameters')
        result[name] = {
            'method': method,
            'fdcr_enabled': encoder.fdcr_enabled,
            'rdcf_enabled': encoder.rdcf_enabled,
            'spdr_enabled': encoder.spdr_enabled,
            'independent_N3_N4_parameters': True,
            'status': 'PASS',
        }
        del model, config
        gc.collect()
    return result


def fairness_audit():
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
        if candidate.get(method) != EXPECTED_OPTIONS[method]:
            raise RuntimeError(
                f'FAIRNESS ERROR: {name} {method}='
                f'{candidate.get(method)!r}; expected '
                f'{EXPECTED_OPTIONS[method]!r}')
        other = 'RDCF' if method == 'FDCR' else 'FDCR'
        if candidate.get(other, {}).get('enabled', False):
            raise RuntimeError(f'{name}: {other} must remain disabled')
        if candidate.get('SPDR', {}).get('enabled', False):
            raise RuntimeError(f'{name}: SPDR must remain disabled')
        result[name] = {
            'reference': reference_name,
            'allowed_namespace': method,
            'resolved_differences': diff,
            'fixed_options': EXPECTED_OPTIONS[method],
            'status': 'PASS',
        }
    print('Resolved-config fairness (four candidate/reference pairs): PASS')
    return result


def disabled_equivalence(input_size=128):
    """Compare complete PResNet/HRNet detectors with both methods off."""
    result = {}
    image = torch.randn(1, 3, input_size, input_size)
    for name in CANDIDATES:
        reference_name = REFERENCE_FOR[name]
        torch.manual_seed(17)
        _, reference = build(reference_name)
        torch.manual_seed(17)
        _, disabled = build(name, method_override={'enabled': False})
        if disabled.encoder.fdcr_enabled or disabled.encoder.rdcf_enabled:
            raise RuntimeError(f'{name}: disabled Neck is still enabled')
        forbidden = ('fdcr3', 'fdcr4', 'rdcf3', 'rdcf4')
        if any(hasattr(disabled.encoder, attr) for attr in forbidden):
            raise RuntimeError(f'{name}: disabled Neck was still constructed')

        reference_state = reference.state_dict()
        disabled_state = disabled.state_dict()
        if reference_state.keys() != disabled_state.keys():
            raise RuntimeError(f'{name}: disabled state keys differ')
        changed = [key for key in reference_state
                   if not torch.equal(reference_state[key],
                                      disabled_state[key])]
        if changed:
            raise RuntimeError(
                f'{name}: disabled seeded weights differ: {changed[:10]}')
        disabled.load_state_dict(reference_state, strict=True)

        with torch.inference_mode():
            reference_backbone = reference.backbone(image)
            disabled_backbone = disabled.backbone(image)
            reference_neck = reference.encoder(reference_backbone)
            disabled_neck = disabled.encoder(disabled_backbone)
            reference_output = reference(image)
            disabled_output = disabled(image)
        assert_tensor_lists_close(
            reference_backbone, disabled_backbone,
            f'{name}/disabled-backbone')
        assert_tensor_lists_close(
            reference_neck, disabled_neck,
            f'{name}/disabled-N3-N4-N5')
        for key in ('pred_logits', 'pred_boxes'):
            assert_allclose(
                reference_output[key], disabled_output[key],
                f'{name}/disabled-{key}')
        result[name] = {
            'input_size': input_size,
            'state_dict': 'BIT-EXACT',
            'backbone': 'PASS',
            'N3_N4_N5': 'PASS',
            'pred_logits': 'PASS',
            'pred_boxes': 'PASS',
            'tolerance': {'atol': 1e-6, 'rtol': 1e-5},
        }
        print(f'{name}: disabled full-detector equivalence: PASS')
        del reference, disabled, reference_backbone, disabled_backbone
        del reference_neck, disabled_neck, reference_output, disabled_output
        gc.collect()
    return result


def retained_spdr_regression(input_size=128):
    """Build and execute both retained SPDR detectors after cleanup."""
    result = {}
    for name, path in RETAINED_SPDR.items():
        _config, model = build_from_path(name, path)
        encoder = model.encoder
        if not encoder.spdr_enabled:
            raise RuntimeError(f'{name}: retained SPDR is not enabled')
        if encoder.fdcr_enabled or encoder.rdcf_enabled:
            raise RuntimeError(f'{name}: new candidates leaked into SPDR')
        with torch.inference_mode():
            output = model(torch.randn(1, 3, input_size, input_size))
        shapes = {key: list(output[key].shape)
                  for key in ('pred_logits', 'pred_boxes')}
        expected = {
            'pred_logits': [1, 300, 1],
            'pred_boxes': [1, 300, 4],
        }
        if shapes != expected:
            raise RuntimeError(f'{name}: prediction shapes {shapes}')
        if not all(torch.isfinite(value).all()
                   for value in output.values() if torch.is_tensor(value)):
            raise RuntimeError(f'{name}: NaN/Inf output')
        result[name] = {
            'config': str(path.relative_to(ROOT)),
            'input': [1, 3, input_size, input_size],
            **shapes,
            'spdr_enabled': True,
            'fdcr_enabled': False,
            'rdcf_enabled': False,
            'status': 'PASS',
        }
        del model, _config, output
        gc.collect()
    print('Retained PResNet18/HRNetV2-W18 SPDR 128 forward: PASS')
    return result


def common_seeded_weights_and_n5(name, input_size=128):
    """Prove all shared weights and post-CCFF N5 are unmodified."""
    torch.manual_seed(31)
    _, reference = build(REFERENCE_FOR[name])
    torch.manual_seed(31)
    _, candidate = build(name)
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    common = set(reference_state).intersection(candidate_state)
    if common != set(reference_state):
        raise RuntimeError(
            f'{name}: removed keys '
            f'{sorted(set(reference_state) - common)[:10]}')
    changed = [key for key in sorted(common)
               if not torch.equal(reference_state[key], candidate_state[key])]
    if changed:
        raise RuntimeError(
            f'{name}: shifted original seeded weights: {changed[:10]}')

    image = torch.randn(1, 3, input_size, input_size)
    with torch.inference_mode():
        baseline_features = reference.backbone(image)
        candidate_features = candidate.backbone(image)
        baseline_neck = reference.encoder(baseline_features)
        candidate_neck = candidate.encoder(candidate_features)
    if not torch.equal(baseline_neck[2], candidate_neck[2]):
        maximum = (baseline_neck[2] - candidate_neck[2]).abs().max().item()
        raise RuntimeError(f'{name}: N5 changed (max_abs={maximum})')
    result = {
        'status': 'PASS',
        'common_tensors': len(common),
        'added_tensors': sorted(set(candidate_state) - set(reference_state)),
        'common_seeded_weights': 'BIT-EXACT',
        'N5_identity': 'BIT-EXACT',
    }
    del reference, candidate, baseline_features, candidate_features
    del baseline_neck, candidate_neck
    gc.collect()
    return result


def dynamic_detector_forward(model, name, resolutions):
    model.eval()
    captured = []
    handle = model.encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.append(
            [list(value.shape) for value in output]))
    result = {}
    try:
        for size in resolutions:
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
            del image, output
    finally:
        handle.remove()
    return result


def synthetic_backward(model, name):
    channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
    model.zero_grad(set_to_none=True)
    model.encoder.train()
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


def full_detector_backward(model, name, input_size=128):
    """Use a real detector training forward and recursively summed loss."""
    model.zero_grad(set_to_none=True)
    model.train()
    image = torch.randn(1, 3, input_size, input_size)
    targets = [{
        'labels': torch.tensor([0], dtype=torch.long),
        'boxes': torch.tensor(
            [[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
    }]
    output = model(image, targets)
    loss = recursive_tensor_loss(output)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f'{name}: invalid full-detector dummy loss')
    loss.backward()
    check_new_gradients(model, name, require_nonzero=True)
    return {
        'input': [1, 3, input_size, input_size],
        'loss': loss.detach().item(),
        'new_parameter_gradients': 'ALL PRESENT, FINITE, NONZERO',
        'formal_optimizer_step': False,
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
            'expected_weight_decay': expected_decay,
        }
        if lr != 3e-4 or weight_decay != expected_decay:
            raise RuntimeError(f'{name}: optimizer mismatch {row}')
        rows.append(row)
    if not rows:
        raise RuntimeError(f'{name}: no new optimizer parameters')
    return rows


def cpu_autocast_smoke(model, name):
    channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
    model.zero_grad(set_to_none=True)
    model.encoder.train()
    features = [
        torch.randn(1, channels[0], 16, 16, requires_grad=True),
        torch.randn(1, channels[1], 8, 8, requires_grad=True),
        torch.randn(1, channels[2], 4, 4, requires_grad=True),
    ]
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = model.encoder(features)
        loss = sum(value.float().square().mean() for value in output)
    if not torch.isfinite(loss) or not all(
            torch.isfinite(value).all() for value in output):
        raise RuntimeError(f'{name}: CPU autocast produced NaN/Inf')
    loss.backward()
    check_new_gradients(model, name, require_nonzero=False)
    return {
        'status': 'PASS',
        'dtype': 'torch.bfloat16',
        'loss': loss.detach().item(),
        'gradients': 'ALL PRESENT AND FINITE',
    }


def rdcf_deploy_equivalence(name):
    if method_for(name) != 'RDCF':
        raise ValueError('RDCF deploy audit requires an RDCF candidate')
    channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
    torch.manual_seed(43)
    _, training_model = build(name)
    features = [
        torch.randn(1, channels[0], 16, 16),
        torch.randn(1, channels[1], 8, 8),
        torch.randn(1, channels[2], 4, 4),
    ]
    with torch.inference_mode():
        training_output = training_model.encoder(features)
    training_model.encoder.rdcf3.switch_to_deploy()
    training_model.encoder.rdcf4.switch_to_deploy()
    with torch.inference_mode():
        converted_output = training_model.encoder(features)
    assert_tensor_lists_close(
        training_output, converted_output,
        f'{name}/train-vs-converted-deploy', atol=1e-5, rtol=1e-4)
    if not torch.equal(training_output[2], converted_output[2]):
        raise RuntimeError(f'{name}: deploy conversion changed N5')

    converted_state = training_model.state_dict()
    deploy_options = copy.deepcopy(EXPECTED_OPTIONS['RDCF'])
    deploy_options['deploy'] = True
    _, deploy_model = build(name, method_override=deploy_options)
    incompatible = deploy_model.load_state_dict(converted_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f'{name}: strict deploy load mismatch: {incompatible}')
    with torch.inference_mode():
        loaded_output = deploy_model.encoder(features)
    assert_tensor_lists_close(
        converted_output, loaded_output,
        f'{name}/converted-vs-strict-loaded', atol=1e-5, rtol=1e-4)
    state_keys = set(converted_state)
    if any('.dw_3x3.' in key or '.dw_1x9.' in key
           or '.dw_9x1.' in key for key in state_keys):
        raise RuntimeError(f'{name}: training branches remain after conversion')
    if not any('.reparam_conv.' in key for key in state_keys):
        raise RuntimeError(f'{name}: deploy kernel is absent')
    result = {
        'status': 'PASS',
        'tolerance': {'atol': 1e-5, 'rtol': 1e-4},
        'N5_identity': 'BIT-EXACT',
        'strict_state_load': True,
        'training_branches_removed': True,
        'reparam_9x9_present': True,
    }
    del training_model, deploy_model, training_output, converted_output
    del loaded_output, converted_state
    gc.collect()
    return result


def smoke(resolutions, equivalence_input_size):
    report = {
        'final_config_set': final_config_set_audit(),
        'retained_SPDR_regression': retained_spdr_regression(
            input_size=equivalence_input_size),
        'candidate_activation': candidate_activation_audit(),
        'fairness': fairness_audit(),
        'disabled_equivalence': disabled_equivalence(
            input_size=equivalence_input_size),
        'models': {},
        'optimizer': {},
        'common_seeded_weights_and_N5': {},
        'RDCF_train_deploy': {},
    }
    for name in CANDIDATES:
        config, model = build(name)
        report['models'][name] = {
            'dynamic_detector_forward': dynamic_detector_forward(
                model, name, resolutions),
            'encoder_backward': synthetic_backward(model, name),
            'cpu_autocast': cpu_autocast_smoke(model, name),
            'full_detector_backward': full_detector_backward(
                model, name, input_size=equivalence_input_size),
        }
        rows = optimizer_rows(config, model, name)
        report['optimizer'][name] = rows
        del config, model
        gc.collect()
        report['common_seeded_weights_and_N5'][name] = (
            common_seeded_weights_and_n5(
                name, input_size=equivalence_input_size))
        print(f'{name}: detector resolutions={list(resolutions)}, backward, '
              'CPU autocast, optimizer, common weights and N5: PASS')
    for name in CANDIDATES:
        if method_for(name) == 'RDCF':
            report['RDCF_train_deploy'][name] = (
                rdcf_deploy_equivalence(name))
            print(f'{name}: training/deploy equivalence + strict load: PASS')
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


def one_complexity_row(model):
    model = model.cpu().eval()
    image = torch.randn(1, 3, 640, 640)
    macs, output = conv_linear_macs(model, image)
    row = {
        'input': [1, 3, 640, 640],
        'whole_params': sum(parameter.numel()
                            for parameter in model.parameters()),
        'whole_conv_linear_macs_lower_bound': macs,
        'whole_conv_linear_flops_lower_bound_2_per_mac': 2 * macs,
        'prediction_shapes': {
            key: list(value.shape) for key, value in output.items()
            if torch.is_tensor(value)
        },
        'omitted_ops': (
            'normalization/pooling/interpolation/activation/concat/'
            'elementwise and all other non-Conv/Linear operations'),
    }
    del image, output
    return row


def complexity():
    """Measure two baselines, four candidates and both deploy RDCF forms."""
    report = {}
    for name in CONFIGS:
        torch.manual_seed(0)
        _, model = build(name)
        report[name] = one_complexity_row(model)
        if name in CANDIDATES and method_for(name) == 'RDCF':
            model.encoder.rdcf3.switch_to_deploy()
            model.encoder.rdcf4.switch_to_deploy()
            deploy_name = name + ' (deploy)'
            report[deploy_name] = one_complexity_row(model)
        del model
        gc.collect()

    for name in CANDIDATES:
        reference = report[REFERENCE_FOR[name]]
        candidate = report[name]
        candidate['added_params_vs_original'] = (
            candidate['whole_params'] - reference['whole_params'])
        candidate['added_conv_linear_macs_vs_original'] = (
            candidate['whole_conv_linear_macs_lower_bound']
            - reference['whole_conv_linear_macs_lower_bound'])
        if method_for(name) == 'RDCF':
            deploy = report[name + ' (deploy)']
            deploy['added_params_vs_original'] = (
                deploy['whole_params'] - reference['whole_params'])
            deploy['added_conv_linear_macs_vs_original'] = (
                deploy['whole_conv_linear_macs_lower_bound']
                - reference['whole_conv_linear_macs_lower_bound'])
            deploy['params_vs_training_form'] = (
                deploy['whole_params'] - candidate['whole_params'])
            deploy['macs_vs_training_form'] = (
                deploy['whole_conv_linear_macs_lower_bound']
                - candidate['whole_conv_linear_macs_lower_bound'])
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
                check_new_gradients(
                    ddp.module, name, require_nonzero=False)
                losses.append(loss.detach().item())
            dist.barrier()
            report[name] = {
                'status': 'PASS',
                'steps': 2,
                'losses': losses,
                'world_size': world_size,
                'amp': amp,
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--complexity', action='store_true')
    parser.add_argument('--amp-smoke', action='store_true')
    parser.add_argument('--ddp-smoke', action='store_true')
    parser.add_argument('--amp', action='store_true',
                        help='Enable autocast in --ddp-smoke')
    parser.add_argument(
        '--resolutions', type=int, nargs='+', default=[480, 640, 800],
        help='Detector sizes used by --smoke (default: 480 640 800).')
    parser.add_argument(
        '--equivalence-input-size', type=int, default=128,
        help='Compact square input for full disabled/N5 equivalence.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not any((args.smoke, args.complexity,
                args.amp_smoke, args.ddp_smoke)):
        parser.error('select at least one validation action')
    if any(size < 32 or size % 32 for size in args.resolutions):
        parser.error('--resolutions values must be positive multiples of 32')
    if (args.equivalence_input_size < 32
            or args.equivalence_input_size % 32):
        parser.error('--equivalence-input-size must be a multiple of 32')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    report = {
        'protocol': {
            'candidate_order': list(CANDIDATES),
            'pretrained_disabled_for_synthetic_audit_only': True,
            'dataset_loaded': False,
            'formal_training_launched': False,
            'conv_linear_MAC_definition': (
                'executed Conv2d/Linear multiply-accumulates only'),
        },
    }
    if args.smoke:
        report['smoke'] = smoke(
            tuple(args.resolutions), args.equivalence_input_size)
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
