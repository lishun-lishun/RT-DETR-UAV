"""Structural and numerical smoke tests for HRDI, SAQS and SGR.

This script never trains or reads the DUT dataset.  Use ``--forward`` on the
server to additionally run the requested Bx3x640x640 end-to-end forwards.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _allow_import_without_pycocotools():
    """Install import-only stubs for local model tests, never data/evaluation."""
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                'pycocotools stub is import-only; dataset/evaluation is disabled')

    if importlib.util.find_spec('pycocotools') is None:
        package = types.ModuleType('pycocotools')
        package.__path__ = []
        mask = types.ModuleType('pycocotools.mask')
        coco = types.ModuleType('pycocotools.coco')
        cocoeval = types.ModuleType('pycocotools.cocoeval')
        coco.COCO = _Unavailable
        cocoeval.COCOeval = _Unavailable
        for name in ('encode', 'decode', 'area', 'toBbox', 'frPyObjects', 'merge'):
            setattr(mask, name, lambda *args, **kwargs: None)
        package.mask = mask
        sys.modules.update({
            'pycocotools': package,
            'pycocotools.mask': mask,
            'pycocotools.coco': coco,
            'pycocotools.cocoeval': cocoeval,
        })
    if importlib.util.find_spec('transformers') is None:
        transformers = types.ModuleType('transformers')
        transformers.RegNetModel = _Unavailable
        sys.modules['transformers'] = transformers


PRES = 'rtdetr_r18vd_dut_anti_uav'
HRNET = 'rtdetr_hrnetv2_w18_dut_anti_uav'
SUFFIXES = ('', '_hrdi', '_saqs', '_sgr', '_hrdi_saqs',
            '_hrdi_sgr', '_saqs_sgr', '_hrdi_saqs_sgr')


def _load(name, seed=123):
    from src.core import YAMLConfig

    torch.manual_seed(seed)
    cfg = YAMLConfig(str(ROOT / 'configs' / 'rtdetr' / f'{name}.yml'))
    # The smoke test verifies architecture, not external ImageNet downloads.
    cfg.yaml_cfg.get('PResNet', {})['pretrained'] = False
    cfg.yaml_cfg.get('HRNetV2W18', {})['pretrained'] = False
    model = cfg.model
    criterion = cfg.criterion
    return model, criterion


def _expected(name, token):
    return token in name.removeprefix(PRES).removeprefix(HRNET).split('_')


def _dynamic_eval(model):
    """Use runtime spatial shapes instead of the cached formal 640 grid."""
    model.eval()
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    return model


def test_all_yaml_builds():
    rows = []
    for family in (PRES, HRNET):
        for suffix in SUFFIXES:
            name = family + suffix
            model, criterion = _load(name)
            actual = (
                hasattr(model.encoder, 'hrdi'),
                hasattr(model.decoder, 'saqs_selector'),
                criterion.sgr_enabled,
            )
            expected = tuple(_expected(name, token)
                             for token in ('hrdi', 'saqs', 'sgr'))
            if actual != expected:
                raise AssertionError(
                    f'{name}: switches {actual}, expected {expected}')
            rows.append(name)
            del model, criterion
            gc.collect()
    print(f'[PASS] YAML/build/switch isolation: {len(rows)} configurations')


def test_common_initialization_and_identity(image_size):
    # SGR changes criterion only; SAQS has no parameters; HRDI has only extra
    # keys and alpha=0 must leave the detector output bit-exact at initialization.
    for family in (PRES, HRNET):
        baseline, _ = _load(family, seed=777)
        hrdi, _ = _load(family + '_hrdi', seed=777)
        base_state, candidate_state = baseline.state_dict(), hrdi.state_dict()
        for key, value in base_state.items():
            if key not in candidate_state or not torch.equal(
                    value, candidate_state[key]):
                raise AssertionError(f'{family}: common state differs at {key}')
        if set(candidate_state) == set(base_state):
            raise AssertionError(f'{family}: HRDI added no state')

        _dynamic_eval(baseline)
        _dynamic_eval(hrdi)
        generator = torch.Generator().manual_seed(101)
        sample = torch.randn(1, 3, image_size, image_size,
                             generator=generator)
        with torch.inference_mode():
            expected = baseline(sample)
            actual = hrdi(sample)
        for key in ('pred_logits', 'pred_boxes'):
            if not torch.equal(expected[key], actual[key]):
                error = (expected[key] - actual[key]).abs().max().item()
                raise AssertionError(
                    f'{family}: alpha=0 output differs for {key}: {error}')
        del baseline, hrdi, expected, actual, sample
        gc.collect()
    print('[PASS] common initialization and HRDI alpha=0 exact identity')


def test_forward_shapes(batch_size, image_size):
    names = [family + suffix for family in (PRES, HRNET)
             for suffix in ('', '_hrdi', '_saqs', '_sgr')]
    for name in names:
        model, _ = _load(name, seed=31)
        _dynamic_eval(model)
        sample = torch.randn(batch_size, 3, image_size, image_size)
        with torch.inference_mode():
            backbone = model.backbone(sample)
            pyramid = model.encoder(backbone)
            output = model.decoder(pyramid)
        expected_hw = [(image_size // stride, image_size // stride)
                       for stride in (8, 16, 32)]
        shapes = [tuple(level.shape) for level in pyramid]
        for level, spatial in zip(pyramid, expected_hw):
            if tuple(level.shape) != (batch_size, 256, *spatial):
                raise AssertionError(f'{name}: unexpected pyramid {shapes}')
        if tuple(output['pred_logits'].shape[:2]) != (batch_size, 300):
            raise AssertionError(f'{name}: bad pred_logits shape')
        if tuple(output['pred_boxes'].shape) != (batch_size, 300, 4):
            raise AssertionError(f'{name}: bad pred_boxes shape')
        print(f'[PASS] forward {name}: neck={shapes}, '
              f'queries={tuple(output["pred_boxes"].shape)}')
        del model, sample, backbone, pyramid, output
        gc.collect()


def test_module_gradients():
    from src.zoo.rtdetr.hrdi import HighResolutionDetailInjection
    from src.zoo.rtdetr.saqs import ScaleAwareQuerySelector
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion

    hrdi = HighResolutionDetailInjection(
        source='c2', source_channels=8, source_stride=4,
        targets=['P3', 'P4'])
    detail = torch.randn(2, 8, 32, 32, requires_grad=True)
    levels = [torch.randn(2, 256, 16 // (2 ** i), 16 // (2 ** i),
                          requires_grad=True) for i in range(3)]
    sum(value.sum() for value in hrdi(levels, detail)).backward()
    alpha_grads = [value.grad for value in hrdi.alphas.values()]
    if not all(value is not None and torch.isfinite(value).all()
               for value in alpha_grads):
        raise AssertionError('HRDI alpha gradient is missing/non-finite')

    selector = ScaleAwareQuerySelector(
        num_queries=12, feature_level_strides=[8, 16, 32],
        level_ratios=[0.5, 1 / 3, 1 / 6])
    # Reproduce formal AMP behavior: the score head can remain FP16 while
    # adding FP32 anchors promotes proposal logits to FP32.
    logits = torch.randn(2, 21, 1, dtype=torch.float16)
    boxes = torch.randn(2, 21, 4, dtype=torch.float32)
    memory = torch.randn(2, 21, 16, requires_grad=True)
    indices = selector(logits, boxes, [[4, 4], [2, 2], [1, 1]])
    gathered = memory.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
    gathered.sum().backward()
    if memory.grad is None or not torch.isfinite(memory.grad).all():
        raise AssertionError('SAQS gather broke the memory computation graph')

    class _Matcher(nn.Module):
        def forward(self, outputs, targets):
            return [(torch.arange(len(target['boxes'])),
                     torch.arange(len(target['boxes'])))
                    for target in targets]

    sgr_options = {
        'enabled': True, 'use_scale_weight': True, 'gamma': 0.25,
        'weight_max': 2.0, 'area_ref_mode': 'fixed', 'area_ref': 0.01,
        'eps': 1e-6, 'use_center_loss': True,
        'center_loss_coef': 0.25, 'min_box_size': 0.01,
        'center_loss_max': 10.0, 'apply_to_aux': True,
        'apply_to_dn': False,
    }
    criterion = SetCriterion(
        _Matcher(), {'loss_bbox': 5.0, 'loss_giou': 2.0},
        ['boxes'], num_classes=1, SGR=sgr_options)
    raw = torch.randn(2, 4, 4, requires_grad=True)
    pred = raw.sigmoid()
    outputs = {
        'pred_logits': torch.zeros(2, 4, 1), 'pred_boxes': pred,
        'aux_outputs': [
            {'pred_logits': torch.zeros(2, 4, 1), 'pred_boxes': pred},
            {'pred_logits': torch.zeros(2, 4, 1), 'pred_boxes': pred},
        ],
    }
    targets = [
        {'labels': torch.zeros(2, dtype=torch.long),
         'boxes': torch.tensor([[.3, .4, .02, .03], [.7, .6, .2, .1]])},
        {'labels': torch.zeros(1, dtype=torch.long),
         'boxes': torch.tensor([[.5, .5, .01, .01]])},
    ]
    losses = criterion(outputs, targets)
    expected_center = {'loss_center', 'loss_center_aux_0'}
    if not expected_center.issubset(losses) or 'loss_center_aux_1' in losses:
        raise AssertionError(f'SGR stage routing is wrong: {sorted(losses)}')
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise AssertionError('SGR produced NaN/Inf')
    total.backward()
    if raw.grad is None or not torch.isfinite(raw.grad).all():
        raise AssertionError('SGR prediction gradient is missing/non-finite')
    print('[PASS] HRDI alpha, SAQS gather and SGR prediction gradients')


def main(args):
    if args.allow_missing_pycocotools:
        _allow_import_without_pycocotools()
    test_all_yaml_builds()
    test_module_gradients()
    test_common_initialization_and_identity(args.identity_image_size)
    if args.forward:
        test_forward_shapes(args.batch_size, args.image_size)
    else:
        print('[SKIP] end-to-end forward; pass --forward for the full matrix')
    print('ALL HRDI/SAQS/SGR SMOKE TESTS PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--identity-image-size', type=int, default=160)
    parser.add_argument('--allow-missing-pycocotools', action='store_true')
    main(parser.parse_args())
