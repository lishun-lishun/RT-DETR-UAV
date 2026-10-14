"""Smoke/regression checks for the CRCG -> CGQS -> QCLR experiment line.

No dataset or checkpoint is required.  The default compact run is suitable
for CPU.  Use ``--full-640`` on the training server for the requested
B=2, 3x640x640 end-to-end shape run.  This script never starts training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import warnings

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILY = 'rtdetr_hrnetv2_w18_dut_anti_uav'
SUFFIXES = (
    '', '_crcg', '_cgqs', '_qclr', '_crcg_cgqs', '_crcg_qclr',
    '_cgqs_qclr', '_crcg_cgqs_qclr')


def _allow_import_only_stubs():
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('import-only dependency stub was invoked')

    if importlib.util.find_spec('pycocotools') is None:
        package = types.ModuleType('pycocotools')
        package.__path__ = []
        mask = types.ModuleType('pycocotools.mask')
        coco = types.ModuleType('pycocotools.coco')
        cocoeval = types.ModuleType('pycocotools.cocoeval')
        coco.COCO = _Unavailable
        cocoeval.COCOeval = _Unavailable
        for name in ('encode', 'decode', 'area', 'toBbox',
                     'frPyObjects', 'merge'):
            setattr(mask, name, lambda *args, **kwargs: None)
        package.mask = mask
        sys.modules.update({
            'pycocotools': package, 'pycocotools.mask': mask,
            'pycocotools.coco': coco, 'pycocotools.cocoeval': cocoeval})
    if importlib.util.find_spec('transformers') is None:
        transformers = types.ModuleType('transformers')
        transformers.RegNetModel = _Unavailable
        sys.modules['transformers'] = transformers


def _load(suffix='', seed=20261014):
    from src.core import YAMLConfig

    torch.manual_seed(seed)
    path = ROOT / 'configs' / 'rtdetr' / f'{FAMILY}{suffix}.yml'
    cfg = YAMLConfig(str(path))
    cfg.yaml_cfg['HRNetV2W18']['pretrained'] = False
    model, criterion = cfg.model, cfg.criterion
    model.multi_scale = None
    return model, criterion


def _dynamic_eval(model):
    model.eval()
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    return model


def _fingerprint(tensor):
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:20]


def test_yaml_builds():
    rows = []
    for suffix in SUFFIXES:
        model, _ = _load(suffix)
        tokens = set(suffix.removeprefix('_').split('_')) if suffix else set()
        actual = (model.backbone.crcg_enabled,
                  model.decoder.cgqs_enabled,
                  model.decoder.qclr_enabled)
        expected = tuple(name in tokens for name in ('crcg', 'cgqs', 'qclr'))
        assert actual == expected, (suffix, actual, expected)
        rows.append((suffix or 'baseline', sum(p.numel() for p in model.parameters())))
        del model
        gc.collect()
    print('[PASS] eight YAML combinations build with isolated switches')
    for name, parameters in rows:
        print(f'  {name:24s} parameters={parameters:,}')


def test_baseline_and_zero_scale_identity(image_size):
    baseline, _ = _load('', seed=20261014)
    candidate, _ = _load('_crcg_qclr', seed=20261014)
    _dynamic_eval(baseline)
    _dynamic_eval(candidate)
    baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    errors = [(baseline_state[key] - candidate_state[key]).abs().max().item()
              for key in baseline_state if key in candidate_state]
    assert len(errors) == len(baseline_state) and max(errors) == 0.0

    generator = torch.Generator().manual_seed(314159)
    inputs = torch.randn(1, 3, image_size, image_size, generator=generator)
    with torch.no_grad():
        base_backbone = baseline.backbone(inputs)
        base_encoder = baseline.encoder(base_backbone)
        base_output = baseline.decoder(base_encoder)
        new_backbone = candidate.backbone(inputs)
        new_encoder_packet = candidate.encoder(new_backbone)
        new_output = candidate.decoder(new_encoder_packet)
    encoder_errors = [
        (left - right).abs().max().item()
        for left, right in zip(base_encoder, new_encoder_packet['features'])]
    logits_error = (base_output['pred_logits']
                    - new_output['pred_logits']).abs().max().item()
    boxes_error = (base_output['pred_boxes']
                   - new_output['pred_boxes']).abs().max().item()
    assert max(encoder_errors + [logits_error, boxes_error]) == 0.0
    print('[PASS] zero-scale regression: '
          f'encoder={encoder_errors}, logits={logits_error}, boxes={boxes_error}')
    print('  baseline fingerprints: '
          f"logits={_fingerprint(base_output['pred_logits'])}, "
          f"boxes={_fingerprint(base_output['pred_boxes'])}")


def test_direct_b2_640_shapes():
    from src.nn.backbone.crcg import CrossResolutionConsensusGating
    from src.zoo.rtdetr.cgqs import ConsensusGuidedQuerySelector
    from src.zoo.rtdetr.qclr import QueryConditionedLocalRefinement

    crcg = CrossResolutionConsensusGating([18, 36, 72, 144], debug=True)
    branches = [torch.randn(2, channels, 640 // stride, 640 // stride)
                for channels, stride in zip((18, 36, 72, 144), (4, 8, 16, 32))]
    refined, reliability = crcg(branches)
    assert reliability.shape == (2, 1, 160, 160)
    assert refined[0].shape == branches[0].shape

    cgqs = ConsensusGuidedQuerySelector(num_queries=300, debug=True)
    indices = cgqs(
        torch.randn(2, 8400, 1), ((80, 80), (40, 40), (20, 20)),
        torch.ones(1, 8400, 1, dtype=torch.bool), reliability)
    assert indices.shape == (2, 300)

    qclr = QueryConditionedLocalRefinement(debug=True)
    queries = torch.randn(2, 300, 256)
    boxes = torch.rand(2, 300, 4)
    boxes[..., 2:].mul_(0.08)
    output = qclr(queries, boxes, torch.randn(2, 300, 1), refined[0])
    assert output.shape == (2, 300, 4)
    assert qclr.last_patch_shape == (2, 100, 25, 18)
    print('[PASS] direct B=2/640 shapes: '
          f'branches={crcg.last_branch_shapes}, R={tuple(reliability.shape)}, '
          f'queries={tuple(indices.shape)}, patches={qclr.last_patch_shape}, '
          f'boxes={tuple(output.shape)}')


def test_end_to_end_shapes(batch_size, image_size):
    tested = ('', '_crcg', '_cgqs', '_qclr',
              '_crcg_cgqs', '_crcg_cgqs_qclr')
    for suffix in tested:
        model, _ = _load(suffix, seed=31)
        _dynamic_eval(model)
        inputs = torch.randn(batch_size, 3, image_size, image_size)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with torch.no_grad():
                output = model(inputs)
        assert output['pred_logits'].shape == (batch_size, 300, 1)
        assert output['pred_boxes'].shape == (batch_size, 300, 4)
        assert torch.isfinite(output['pred_logits']).all()
        assert torch.isfinite(output['pred_boxes']).all()
        print(f'[PASS] forward {suffix or "baseline":24s} '
              f'logits={tuple(output["pred_logits"].shape)} '
              f'boxes={tuple(output["pred_boxes"].shape)}')
        del model, output, inputs
        gc.collect()


def test_backward():
    # Use nonzero residual scales for this gradient-connectivity check. At the
    # required zero initialization, projection gradients are mathematically
    # zero on step one while alpha/beta themselves receive the first update.
    model, criterion = _load('_crcg_cgqs_qclr', seed=91)
    model.train()
    with torch.no_grad():
        model.backbone.crcg.alpha.fill_(0.1)
        model.decoder.qclr.beta.fill_(0.1)
    inputs = torch.randn(2, 3, 128, 128)
    targets = [
        {'labels': torch.tensor([0, 0]),
         'boxes': torch.tensor([[.30, .30, .04, .05], [.65, .60, .20, .18]])},
        {'labels': torch.tensor([0, 0]),
         'boxes': torch.tensor([[.20, .72, .03, .04], [.72, .25, .18, .22]])},
    ]
    output = model(inputs, targets)
    losses = criterion(output, targets)
    loss = sum(losses[key] * criterion.weight_dict[key]
               for key in losses if key in criterion.weight_dict)
    assert torch.isfinite(loss)
    # Criterion matching need not select one of QCLR's top-N small queries at
    # random initialization. Add a prediction-only probe solely to exercise
    # that differentiable path; it is not a training loss or configuration.
    connectivity_probe = output['pred_boxes'][..., :2].mean()
    (loss + connectivity_probe).backward()
    watched = {
        'CRCG alpha': model.backbone.crcg.alpha,
        'CRCG projections': next(model.backbone.crcg.projections.parameters()),
        'QCLR beta': model.decoder.qclr.beta,
        'QCLR Wq': model.decoder.qclr.query_projection.weight,
        'QCLR Wf': model.decoder.qclr.feature_projection.weight,
    }
    for name, parameter in watched.items():
        assert parameter.grad is not None, f'{name} has no gradient'
        assert torch.isfinite(parameter.grad).all(), f'{name} gradient is not finite'
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    print('[PASS] criterion forward/backward: '
          f'loss={float(loss.detach()):.6f}, main={tuple(output["pred_boxes"].shape)}, '
          f'aux={len(output["aux_outputs"])}, dn={len(output["dn_aux_outputs"])}')
    for name, parameter in watched.items():
        print(f'  {name:18s} grad_abs_sum={float(parameter.grad.abs().sum()):.6g}')


def test_presnet_contracts():
    from src.core import YAMLConfig

    base_path = ROOT / 'configs' / 'rtdetr' / 'rtdetr_r18vd_dut_anti_uav.yml'
    for name, message in (
            ('CRCG', 'CRCG requires multi-resolution HRNet features'),
            ('QCLR', 'QCLR requires the HRNet highest-resolution feature')):
        cfg = YAMLConfig(str(base_path))
        cfg.yaml_cfg[name] = {'enabled': True}
        try:
            _ = cfg.model
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f'PResNet should reject {name}')

    cfg = YAMLConfig(str(base_path))
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    cfg.yaml_cfg['CGQS'] = {
        'enabled': True, 'use_consensus': True, 'consensus_weight': .5,
        'use_local_peak': True, 'peak_weight': .25, 'peak_kernel': 3,
        'score_mode': 'additive', 'eps': 1e-6, 'align_mode': 'bilinear',
        'align_corners': False, 'debug': False}
    model = cfg.model
    assert model.decoder.cgqs_enabled
    _dynamic_eval(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with torch.no_grad():
            output = model(torch.randn(1, 3, 128, 128))
    assert output['pred_boxes'].shape == (1, 300, 4)
    print('[PASS] PResNet contracts: CRCG/QCLR reject clearly; '
          'CGQS local-peak fallback builds and forwards')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--full-640', action='store_true',
                        help='run six end-to-end forwards at B=2, 640x640')
    parser.add_argument('--skip-backward', action='store_true')
    args = parser.parse_args()
    if args.full_640:
        args.batch_size, args.image_size = 2, 640

    _allow_import_only_stubs()
    test_yaml_builds()
    test_baseline_and_zero_scale_identity(min(args.image_size, 128))
    test_direct_b2_640_shapes()
    test_end_to_end_shapes(args.batch_size, args.image_size)
    if not args.skip_backward:
        test_backward()
    test_presnet_contracts()
    print('[PASS] CRCG/CGQS/QCLR validation completed')


if __name__ == '__main__':
    main()
