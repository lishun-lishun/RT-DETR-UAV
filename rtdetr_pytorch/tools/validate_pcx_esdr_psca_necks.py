"""Validate the six independent PCX/ESDR/PSCA RT-DETR experiments.

This is a synthetic audit only: it never loads DUT-Anti-UAV, performs an
optimizer step, or starts formal training.  The default full gate is::

  python tools/validate_pcx_esdr_psca_necks.py --smoke \
    --resolutions 480 640 800 --complexity \
    --output reports/pcx_esdr_psca_cpu.json

For a resource-conscious development-machine audit use ``--resolutions 128``
and omit ``--complexity``.  The JSON records the sizes actually executed.
Server-only CUDA gates are exposed separately::

  CUDA_VISIBLE_DEVICES=1 python tools/validate_pcx_esdr_psca_necks.py \
    --amp-smoke
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 \
    --master_port=9931 tools/validate_pcx_esdr_psca_necks.py \
    --ddp-smoke --amp

MAC counts cover executed Conv2d/Linear operations.  PSCA's functional QK and
AV matrix multiplications are reported explicitly in separate fields.
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


# Avoid dataset/evaluator imports (and therefore a pycocotools requirement).
core = import_model_source(selective=True)
importlib.import_module('src.nn.backbone.hrnet')
importlib.import_module('src.optim')
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder  # noqa: E402


CONFIGS = {
    'PResNet18 + Original':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml',
    'HRNetV2-W18 + Original':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml',
    'PResNet18 + PCX':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pcx.yml',
    'HRNetV2-W18 + PCX':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_pcx.yml',
    'PResNet18 + ESDR':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_esdr.yml',
    'HRNetV2-W18 + ESDR':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_esdr.yml',
    'PResNet18 + PSCA':
        ROOT / 'configs/rtdetr/rtdetr_r18vd_dut_anti_uav_psca.yml',
    'HRNetV2-W18 + PSCA':
        ROOT / 'configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav_psca.yml',
}

CANDIDATES = (
    'PResNet18 + PCX',
    'HRNetV2-W18 + PCX',
    'PResNet18 + ESDR',
    'HRNetV2-W18 + ESDR',
    'PResNet18 + PSCA',
    'HRNetV2-W18 + PSCA',
)

REFERENCE_FOR = {
    name: ('HRNetV2-W18 + Original' if name.startswith('HRNet')
           else 'PResNet18 + Original')
    for name in CANDIDATES
}

EXPECTED_OPTIONS = {
    'PCX': {
        'enabled': True,
        'exchange_ratio': 0.25,
        'channel_shuffle': True,
    },
    'ESDR': {
        'enabled': True,
        'beta_max': 0.20,
        'beta_init': 0.02,
    },
    'PSCA': {
        'enabled': True,
        'context_ratio': 0.25,
        'attention_dim': 32,
        'p3_pool_stride': 4,
        'p4_pool_stride': 2,
        'alpha_max': 0.20,
        'alpha_init': 0.02,
        'channel_shuffle': True,
    },
}

METHODS = tuple(EXPECTED_OPTIONS)


def is_hrnet(name):
    return name.startswith('HRNetV2-W18')


def method_for(name):
    for method in METHODS:
        if name.endswith(method):
            return method
    raise ValueError(f'{name!r} is not a PCX/ESDR/PSCA candidate')


def prefixes_for(name):
    method = method_for(name)
    if method == 'PCX':
        return ('encoder.pcx.',)
    if method == 'ESDR':
        return ('encoder.esdr34.', 'encoder.esdr45.')
    return ('encoder.psca3.', 'encoder.psca4.')


def local_prefixes_for(name):
    return tuple(prefix.split('encoder.', 1)[1]
                 for prefix in prefixes_for(name))


def backbone_overrides(name):
    if is_hrnet(name):
        return {'HRNetV2W18': {
            'pretrained': False,
            'pretrained_path': None,
        }}
    return {'PResNet': {'pretrained': False}}


def build(name, training=False, method_override=None):
    """Build a real detector while disabling network weight downloads."""
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


def encoder_kwargs(channels):
    return dict(
        in_channels=channels,
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=8,
        dim_feedforward=64,
        expansion=0.5,
        depth_mult=0.34,
        num_encoder_layers=0,
        eval_spatial_size=None,
    )


def synthetic_features(channels, batch=1, requires_grad=False):
    return [
        torch.randn(batch, channels[0], 16, 16,
                    requires_grad=requires_grad),
        torch.randn(batch, channels[1], 8, 8,
                    requires_grad=requires_grad),
        torch.randn(batch, channels[2], 4, 4,
                    requires_grad=requires_grad),
    ]


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
        raise RuntimeError(f'{context}: tensor-list lengths differ')
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
    invalid = []
    for parameter_name, parameter in parameters:
        if parameter.grad is None:
            invalid.append(parameter_name + ':missing')
        elif not torch.isfinite(parameter.grad).all():
            invalid.append(parameter_name + ':nonfinite')
        elif require_nonzero and parameter.grad.abs().sum().item() == 0.0:
            invalid.append(parameter_name + ':zero')
    if invalid:
        raise RuntimeError(f'{name}: invalid gradients: {invalid}')


def required_config_audit():
    missing = [str(path.relative_to(ROOT)) for path in CONFIGS.values()
               if not path.is_file()]
    if missing:
        raise RuntimeError(f'Missing required experiment configs: {missing}')
    return {
        'status': 'PASS',
        'required_files': [str(path.relative_to(ROOT))
                           for path in CONFIGS.values()],
    }


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
                f'FAIRNESS ERROR: {name} {method} options are '
                f'{candidate.get(method)!r}')
        for other in METHODS:
            if other != method and candidate.get(other, {}).get(
                    'enabled', False):
                raise RuntimeError(f'{name}: {other} must remain disabled')
        result[name] = {
            'reference': reference_name,
            'allowed_differences': ['__include__', 'output_dir', method],
            'resolved_differences': diff,
            'fixed_options': EXPECTED_OPTIONS[method],
            'status': 'PASS',
        }
    return result


def mutual_exclusion_audit():
    combinations = (
        ('PCX', 'ESDR'),
        ('PCX', 'PSCA'),
        ('ESDR', 'PSCA'),
        ('PCX', 'ESDR', 'PSCA'),
    )
    result = {}
    for enabled in combinations:
        options = {method: {'enabled': True} for method in enabled}
        try:
            HybridEncoder(**encoder_kwargs([16, 32, 64]), **options)
        except ValueError as error:
            expected = 'PCX, ESDR and PSCA must be evaluated independently.'
            if expected not in str(error):
                raise RuntimeError(
                    f'{enabled}: wrong rejection message: {error}') from error
            result['+'.join(enabled)] = 'REJECTED'
        else:
            raise RuntimeError(f'{enabled}: invalid combination was accepted')
    historical = ('ACR', 'SLR', 'PAF', 'BOR', 'DGFR', 'SPDR', 'FDCR', 'RDCF')
    historical_result = {}
    for method in METHODS:
        for old_method in historical:
            enabled = {
                method: {'enabled': True},
                old_method: {'enabled': True},
            }
            try:
                HybridEncoder(**encoder_kwargs([16, 32, 64]), **enabled)
            except ValueError as error:
                if 'independent experiments and cannot mix' not in str(error):
                    raise RuntimeError(
                        f'{method}+{old_method}: wrong rejection message: '
                        f'{error}') from error
                historical_result[f'{method}+{old_method}'] = 'REJECTED'
            else:
                raise RuntimeError(
                    f'{method}+{old_method}: invalid combination was accepted')
    return {
        'status': 'PASS',
        'combinations': result,
        'historical_neck_combinations': historical_result,
    }


def candidate_activation_audit():
    expected_attributes = {
        'PCX': ('pcx',),
        'ESDR': ('esdr34', 'esdr45'),
        'PSCA': ('psca3', 'psca4'),
    }
    all_attributes = tuple(
        attribute for values in expected_attributes.values()
        for attribute in values)
    result = {}
    for name in CANDIDATES:
        method = method_for(name)
        config, model = build(name)
        encoder = model.encoder
        states = {
            candidate: getattr(encoder, candidate.lower() + '_enabled')
            for candidate in METHODS
        }
        if sum(states.values()) != 1 or not states[method]:
            raise RuntimeError(f'{name}: invalid enabled states {states}')
        expected = set(expected_attributes[method])
        actual = {attribute for attribute in all_attributes
                  if hasattr(encoder, attribute)}
        if actual != expected:
            raise RuntimeError(
                f'{name}: module attributes {actual}, expected {expected}')
        if method == 'PCX':
            blocks = (encoder.pcx.refine3, encoder.pcx.refine4,
                      encoder.pcx.refine5)
            parameter_ids = [set(map(id, block.parameters()))
                             for block in blocks]
            if any(parameter_ids[left].intersection(parameter_ids[right])
                   for left in range(3) for right in range(left + 1, 3)):
                raise RuntimeError(f'{name}: PCX levels share parameters')
        else:
            first, second = (getattr(encoder, value)
                             for value in expected_attributes[method])
            if first is second or not set(map(id, first.parameters())).isdisjoint(
                    set(map(id, second.parameters()))):
                raise RuntimeError(f'{name}: two levels share parameters')
        result[name] = {
            'method': method,
            'enabled_states': states,
            'constructed_attributes': sorted(actual),
            'independent_level_parameters': True,
            'status': 'PASS',
        }
        del config, model
        gc.collect()
    return result


def disabled_equivalence(input_size=128):
    """Compare complete disabled candidates to both formal baselines."""
    result = {}
    image = torch.randn(1, 3, input_size, input_size)
    forbidden = ('pcx', 'esdr34', 'esdr45', 'psca3', 'psca4')
    for name in CANDIDATES:
        method = method_for(name)
        torch.manual_seed(17)
        _, reference = build(REFERENCE_FOR[name])
        torch.manual_seed(17)
        _, disabled = build(name, method_override={'enabled': False})
        states = tuple(getattr(
            disabled.encoder, value.lower() + '_enabled')
            for value in METHODS)
        if any(states) or any(hasattr(disabled.encoder, value)
                              for value in forbidden):
            raise RuntimeError(f'{name}: disabled module was constructed')
        reference_state = reference.state_dict()
        disabled_state = disabled.state_dict()
        if reference_state.keys() != disabled_state.keys():
            raise RuntimeError(f'{name}: disabled state keys differ')
        changed = [key for key in reference_state if not torch.equal(
            reference_state[key], disabled_state[key])]
        if changed:
            raise RuntimeError(
                f'{name}: same-seed disabled weights differ: {changed[:10]}')

        reference.eval()
        disabled.eval()
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
            'method': method,
            'input_size': input_size,
            'same_seed_shared_state': 'BIT-EXACT',
            'N3_N4_N5': 'PASS',
            'pred_logits': 'PASS',
            'pred_boxes': 'PASS',
            'tolerance': {'atol': 1e-6, 'rtol': 1e-5},
        }
        del reference, disabled, reference_backbone, disabled_backbone
        del reference_neck, disabled_neck, reference_output, disabled_output
        gc.collect()
    return result


def _trace_pcx(encoder, baseline_neck, features):
    trace = {}

    def hook(_module, inputs, output):
        trace['input'] = [value.detach().clone() for value in inputs[0]]
        trace['output'] = [value.detach().clone() for value in output]

    handle = encoder.pcx.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            candidate_neck = encoder(features)
    finally:
        handle.remove()
    if 'input' not in trace:
        raise RuntimeError('PCX was not called')
    assert_tensor_lists_close(
        baseline_neck, trace['input'], 'PCX/post-CCFF-input')
    assert_tensor_lists_close(
        candidate_neck, trace['output'], 'PCX/output')
    return candidate_neck, {
        'placement': 'post-CCFF N3/N4/N5',
        'call_count': 1,
        'input_levels': 3,
        'output_levels': 3,
    }


def _trace_esdr(encoder, features):
    conv_traces = {}
    esdr_traces = {}
    handles = []

    def conv_hook(index):
        def capture(_module, inputs, output):
            conv_traces[index] = (inputs[0], output)
        return capture

    def esdr_hook(index):
        def capture(_module, inputs, output):
            esdr_traces[index] = (inputs[0], inputs[1], output)
        return capture

    for index, conv in enumerate(encoder.downsample_convs):
        handles.append(conv.register_forward_hook(conv_hook(index)))
    for index, module in enumerate((encoder.esdr34, encoder.esdr45)):
        handles.append(module.register_forward_hook(esdr_hook(index)))
    try:
        with torch.inference_mode():
            candidate_neck = encoder(features)
    finally:
        for handle in handles:
            handle.remove()
    if set(conv_traces) != {0, 1} or set(esdr_traces) != {0, 1}:
        raise RuntimeError('ESDR/downsample calls are not exactly 34 and 45')
    rows = []
    for index, transition in enumerate(('P3->P4', 'P4->P5')):
        conv_source, original_base = conv_traces[index]
        esdr_source, esdr_base, esdr_output = esdr_traces[index]
        if conv_source.data_ptr() != esdr_source.data_ptr():
            raise RuntimeError(f'ESDR {transition}: source was replaced')
        if original_base.data_ptr() != esdr_base.data_ptr():
            raise RuntimeError(f'ESDR {transition}: original base was replaced')
        rows.append({
            'transition': transition,
            'source_shape': list(esdr_source.shape),
            'base_shape': list(esdr_base.shape),
            'output_shape': list(esdr_output.shape),
            'original_downsample_object_preserved': True,
        })
    return candidate_neck, {
        'placement': 'bottom-up downsample only',
        'call_count': 2,
        'transitions': rows,
    }


def _trace_psca(encoder, baseline_neck, features):
    traces = {}
    handles = []

    def psca_hook(level):
        def capture(_module, inputs, output):
            traces[level] = (
                inputs[0].detach().clone(), output.detach().clone())
        return capture

    for level, module in ((3, encoder.psca3), (4, encoder.psca4)):
        handles.append(module.register_forward_hook(psca_hook(level)))
    try:
        with torch.inference_mode():
            candidate_neck = encoder(features)
    finally:
        for handle in handles:
            handle.remove()
    if set(traces) != {3, 4}:
        raise RuntimeError('PSCA was not called exactly on N3 and N4')
    assert_allclose(baseline_neck[0], traces[3][0], 'PSCA/N3-input')
    assert_allclose(baseline_neck[1], traces[4][0], 'PSCA/N4-input')
    assert_allclose(candidate_neck[0], traces[3][1], 'PSCA/N3-output')
    assert_allclose(candidate_neck[1], traces[4][1], 'PSCA/N4-output')
    if not torch.equal(baseline_neck[2], candidate_neck[2]):
        raise RuntimeError('PSCA changed N5')
    return candidate_neck, {
        'placement': 'post-CCFF N3/N4 only',
        'call_count': 2,
        'levels': [3, 4],
        'N5_identity': 'BIT-EXACT',
    }


def common_weights_and_insertion(name, input_size=128):
    """Audit RNG isolation and exact insertion semantics for one candidate."""
    torch.manual_seed(31)
    _, reference = build(REFERENCE_FOR[name])
    torch.manual_seed(31)
    _, candidate = build(name)
    reference.eval()
    candidate.eval()
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    common = set(reference_state).intersection(candidate_state)
    if common != set(reference_state):
        raise RuntimeError(
            f'{name}: removed original keys '
            f'{sorted(set(reference_state) - common)[:10]}')
    changed = [key for key in common if not torch.equal(
        reference_state[key], candidate_state[key])]
    if changed:
        raise RuntimeError(
            f'{name}: common same-seed weights changed: {changed[:10]}')

    image = torch.randn(1, 3, input_size, input_size)
    with torch.inference_mode():
        baseline_features = reference.backbone(image)
        candidate_features = candidate.backbone(image)
        baseline_neck = reference.encoder(baseline_features)
    assert_tensor_lists_close(
        baseline_features, candidate_features, f'{name}/backbone')
    method = method_for(name)
    if method == 'PCX':
        candidate_neck, insertion = _trace_pcx(
            candidate.encoder, baseline_neck, candidate_features)
    elif method == 'ESDR':
        candidate_neck, insertion = _trace_esdr(
            candidate.encoder, candidate_features)
    else:
        candidate_neck, insertion = _trace_psca(
            candidate.encoder, baseline_neck, candidate_features)
    result = {
        'status': 'PASS',
        'same_seed_common_weights': 'BIT-EXACT',
        'common_tensors': len(common),
        'added_tensors': sorted(set(candidate_state) - set(reference_state)),
        'insertion': insertion,
        'output_shapes': [list(value.shape) for value in candidate_neck],
    }
    del reference, candidate, reference_state, candidate_state
    del baseline_features, candidate_features, baseline_neck, candidate_neck
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
    features = synthetic_features(channels, batch=2, requires_grad=True)
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
        'optimizer_step': False,
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
    raw_scales = [row for row in rows if row['parameter'].endswith(
        ('raw_beta', 'raw_alpha'))]
    if method_for(name) in ('ESDR', 'PSCA') and not raw_scales:
        raise RuntimeError(f'{name}: raw LayerScale was not audited')
    return rows


def cpu_autocast_smoke(model, name):
    channels = [36, 72, 144] if is_hrnet(name) else [128, 256, 512]
    model.zero_grad(set_to_none=True)
    model.encoder.train()
    features = synthetic_features(channels, requires_grad=True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = model.encoder(features)
        loss = sum(value.float().square().mean() for value in output)
    if not torch.isfinite(loss) or not all(
            torch.isfinite(value).all() for value in output):
        raise RuntimeError(f'{name}: CPU BF16 produced NaN/Inf')
    loss.backward()
    check_new_gradients(model, name, require_nonzero=False)
    return {
        'status': 'PASS',
        'dtype': 'torch.bfloat16',
        'loss': loss.detach().item(),
        'gradients': 'ALL PRESENT AND FINITE',
    }


def smoke(resolutions, equivalence_input_size):
    report = {
        'required_configs': required_config_audit(),
        'fairness': fairness_audit(),
        'mutual_exclusion': mutual_exclusion_audit(),
        'candidate_activation': candidate_activation_audit(),
        'disabled_equivalence': disabled_equivalence(
            input_size=equivalence_input_size),
        'models': {},
        'optimizer': {},
        'same_seed_and_insertion': {},
    }
    for name in CANDIDATES:
        config, model = build(name)
        report['models'][name] = {
            'detector_forward': dynamic_detector_forward(
                model, name, resolutions),
            'encoder_dummy_backward': synthetic_backward(model, name),
            'full_detector_dummy_backward': full_detector_backward(
                model, name, input_size=equivalence_input_size),
            'cpu_bfloat16': cpu_autocast_smoke(model, name),
        }
        report['optimizer'][name] = optimizer_rows(config, model, name)
        del config, model
        gc.collect()
        report['same_seed_and_insertion'][name] = (
            common_weights_and_insertion(
                name, input_size=equivalence_input_size))
        print(f'{name}: forward/backward/BF16/optimizer/insertion PASS')
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


def psca_attention_macs(model, input_size):
    """Return explicit functional QK and AV MACs for both PSCA levels."""
    encoder = model.encoder
    if not getattr(encoder, 'psca_enabled', False):
        return {'QK': 0, 'AV': 0, 'total': 0, 'levels': {}}
    result = {'QK': 0, 'AV': 0, 'levels': {}}
    for level, stride, module in (
            ('N3', 8, encoder.psca3), ('N4', 16, encoder.psca4)):
        height = input_size // stride
        width = input_size // stride
        pooled_height = height // module.pool_stride
        pooled_width = width // module.pool_stride
        queries = height * width
        keys = pooled_height * pooled_width
        qk = queries * keys * module.attention_dim
        av = queries * keys * module.context_channels
        result['QK'] += qk
        result['AV'] += av
        result['levels'][level] = {
            'queries': queries,
            'keys_values': keys,
            'attention_dim': module.attention_dim,
            'context_channels': module.context_channels,
            'QK': qk,
            'AV': av,
        }
    result['total'] = result['QK'] + result['AV']
    return result


def one_complexity_row(model, input_size=640):
    model = model.cpu().eval()
    image = torch.randn(1, 3, input_size, input_size)
    macs, output = conv_linear_macs(model, image)
    attention = psca_attention_macs(model, input_size)
    row = {
        'input': [1, 3, input_size, input_size],
        'whole_params': sum(parameter.numel()
                            for parameter in model.parameters()),
        'conv_linear_macs': macs,
        'PSCA_attention_macs': attention,
        'conv_linear_plus_PSCA_attention_macs': macs + attention['total'],
        'flops_2_per_reported_mac': 2 * (macs + attention['total']),
        'prediction_shapes': {
            key: list(value.shape) for key, value in output.items()
            if torch.is_tensor(value)
        },
        'omitted_ops': (
            'normalization/pooling/interpolation/activation/softmax/concat/'
            'channel-shuffle/elementwise and other non-Conv/Linear/QK/AV ops'),
    }
    del image, output
    return row


def complexity(input_size=640):
    report = {}
    for name in CONFIGS:
        torch.manual_seed(0)
        _, model = build(name)
        report[name] = one_complexity_row(model, input_size=input_size)
        del model
        gc.collect()
    for name in CANDIDATES:
        reference = report[REFERENCE_FOR[name]]
        candidate = report[name]
        candidate['added_params_vs_original'] = (
            candidate['whole_params'] - reference['whole_params'])
        candidate['added_reported_macs_vs_original'] = (
            candidate['conv_linear_plus_PSCA_attention_macs']
            - reference['conv_linear_plus_PSCA_attention_macs'])
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
        print(f'{name}: CUDA AMP full forward/backward PASS')
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
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
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
                print(f'{name}: 3-GPU DDP two-step forward/backward PASS')
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
                        help='Enable FP16 autocast in --ddp-smoke.')
    parser.add_argument(
        '--resolutions', type=int, nargs='+', default=[480, 640, 800],
        help='Detector sizes for --smoke (default: 480 640 800).')
    parser.add_argument(
        '--equivalence-input-size', type=int, default=128,
        help='Compact size used by equivalence and backward checks.')
    parser.add_argument(
        '--complexity-input-size', type=int, default=640,
        help='Complexity input size; the formal audit uses 640.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not any((args.smoke, args.complexity,
                args.amp_smoke, args.ddp_smoke)):
        parser.error('select at least one validation action')
    sizes = list(args.resolutions) + [
        args.equivalence_input_size, args.complexity_input_size]
    if any(size < 32 or size % 32 for size in sizes):
        parser.error('all input sizes must be positive multiples of 32')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    report = {
        'protocol': {
            'candidate_order': list(CANDIDATES),
            'pretrained_disabled_for_synthetic_audit_only': True,
            'dataset_loaded': False,
            'optimizer_step_executed': False,
            'formal_training_launched': False,
            'conv_linear_MAC_definition': (
                'executed Conv2d/Linear multiply-accumulates'),
            'PSCA_attention_MAC_definition': (
                'functional QK and AV multiply-accumulates'),
            'executed_CPU_detector_resolutions': (
                list(args.resolutions) if args.smoke else []),
            'required_480_640_800_forward_completed': (
                args.smoke
                and {480, 640, 800}.issubset(set(args.resolutions))),
            'executed_complexity_input_size': (
                args.complexity_input_size if args.complexity else None),
            'CUDA_AMP_executed': args.amp_smoke,
            'three_GPU_DDP_executed': args.ddp_smoke,
        },
    }
    if args.smoke:
        report['smoke'] = smoke(
            tuple(args.resolutions), args.equivalence_input_size)
    if args.complexity:
        report['complexity'] = complexity(args.complexity_input_size)
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
