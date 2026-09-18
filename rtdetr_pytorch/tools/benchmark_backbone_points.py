"""Audit six residual-point YAMLs and profile real 640 detector/backbone code.

No dataset, training or pretrained download is needed. GFLOPs are explicitly
counted-operator lower bounds, NOT complete FLOPs: sampling/FFT/fused attention
and some normalization/elementwise math are uncounted. FADC's dense convolution
arithmetic is additionally reported because the native op is profiler-opaque.
Missing DCNv4 is reported as unavailable, NEVER substituted by another op.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'rtdetr_r18vd_dut_anti_uav'
METHODS = {'Baseline': '', 'P0-SRFD': '_p0_srfd', 'P1-DEConv': '_p1_deconv',
           'P2-DCNv4': '_p2_dcnv4', 'P3-SECD34': '_p3_secd34',
           'P4-FADC': '_p4_fadc', 'P4-WTConv': '_p4_wtconv'}
spec = importlib.util.spec_from_file_location('_plugin_points_audit', ROOT / 'tools/analyze_dut_models.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def config_path(method):
    return ROOT / 'configs/rtdetr' / (PREFIX + METHODS[method] + '.yml')


def resolved_audit():
    baseline = audit.fresh_config(config_path('Baseline'))
    methods, failures = {}, []
    for method in METHODS:
        cfg = audit.fresh_config(config_path(method))
        # __include__ is loader provenance, not an effective model setting.
        diff = audit.differences({k: v for k, v in baseline.items() if k != '__include__'},
                                 {k: v for k, v in cfg.items() if k != '__include__'})
        forbidden = [k for k in diff if k != 'output_dir'
                     and k != 'BackbonePlugins' and not k.startswith('BackbonePlugins.')]
        failures.extend(f'{method}: forbidden change {k}' for k in forbidden)
        enabled = [p for p, options in cfg.get('BackbonePlugins', {}).items()
                   if options.get('enabled', False)]
        expected = [] if method == 'Baseline' else [method.split('-')[0]]
        if enabled != expected:
            failures.append(f'{method}: expected exactly {expected}, got {enabled}')
        if cfg['RTDETR']['backbone'] != 'PResNet':
            failures.append(f'{method}: original PResNet must remain selected')
        if any(cfg.get(k, {}).get('enabled', False) for k in ('MERT', 'SECD')):
            failures.append(f'{method}: legacy SECD and MERT must be disabled')
        methods[method] = {'config': str(config_path(method).relative_to(ROOT)),
                           'baseline_differences': diff, 'enabled_points': enabled}
    return {'passed': not failures, 'failures': failures, 'methods': methods,
            'unchanged_protocol': {key: baseline.get(key) for key in
                ('PResNet', 'RTDETR', 'HybridEncoder', 'RTDETRTransformer', 'SetCriterion',
                 'HungarianMatcher', 'optimizer', 'lr_scheduler', 'epoches',
                 'checkpoint_step', 'train_dataloader', 'val_dataloader', 'MERT', 'SECD')}}


def build_model(method, seed=0, selective=False, override=None):
    import torch
    if 'src.core' not in sys.modules:
        core = audit.import_model_source(selective)
    else:
        core = sys.modules['src.core']
    torch.manual_seed(seed)
    kwargs = {'PResNet': {'pretrained': False}}  # TEST/PROFILE ONLY, YAML unmodified
    if override is not None:
        kwargs['BackbonePlugins'] = override
    cfg = core.YAMLConfig(str(config_path(method)), **kwargs)
    return cfg.model, cfg.yaml_cfg


def measure(method, args):
    import torch
    model, cfg = build_model(method, args.seed, args.model_only_import)
    model = model.to(args.device).eval()
    result = {'params': sum(p.numel() for p in model.parameters()),
              'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'backbone_params': sum(p.numel() for p in model.backbone.parameters()),
              'counted_operator_gflops_lower_bound': None,
              'backbone_counted_operator_gflops_lower_bound': None}
    image = torch.randn(1, 3, 640, 640, device=args.device)
    shapes = []
    hook = model.backbone.register_forward_hook(
        lambda module, inputs, outputs: shapes.extend(list(t.shape) for t in outputs))
    try:
        with torch.inference_mode(), torch.autocast('cuda', enabled=args.amp):
            output = model(image)
    except RuntimeError as error:
        if method == 'P2-DCNv4' and ('DCNv4 CUDA extension' in str(error)
                                      or 'CUDA-only' in str(error)):
            result.update(status='unavailable', error=str(error))
            return result
        raise
    finally:
        hook.remove()
    if shapes != [[1, 128, 80, 80], [1, 256, 40, 40], [1, 512, 20, 20]]:
        raise AssertionError(f'{method}: invalid feature shapes {shapes}')
    if any(not torch.isfinite(output[k]).all() for k in ('pred_boxes', 'pred_logits')):
        raise AssertionError(f'{method}: nonfinite detector output')
    result.update(status='passed', backbone_shapes=shapes,
                  output_shapes={key: list(value.shape) for key, value in output.items()})
    for scope, component in (('', model), ('backbone_', model.backbone)):
        with torch.inference_mode(), torch.autocast('cuda', enabled=args.amp), \
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                       with_flops=True, record_shapes=True) as profile:
            component(image)
        counted = {event.key: int(event.flops) for event in profile.key_averages() if event.flops}
        result[scope + 'counted_operator_gflops_lower_bound'] = sum(counted.values()) / 1e9
        result[scope + 'counted_operator_flops'] = counted
        result[scope + 'uncounted_operator_names'] = sorted(event.key for event in profile.key_averages()
                                                          if event.key.startswith('aten::') and not event.flops)
    # All candidate branches execute even at alpha=0 (no branch-pruning profile).
    result['fadc_dense_convolution_gflops_not_counted_by_profiler'] = (
        2 * 1 * 40 * 40 * 256 * 256 * 9 / 1e9 if method == 'P4-FADC' else 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolved-only', action='store_true')
    parser.add_argument('--model-only-import', action='store_true',
                        help='Load actual model code without dataset imports; not full training-env validation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'audit': resolved_audit()}
    if not report['audit']['passed']:
        raise AssertionError(report['audit']['failures'])
    if not args.resolved_only:
        import torch
        if args.amp and not args.device.startswith('cuda'):
            parser.error('--amp uses the training CUDA FP16 protocol')
        torch.set_num_threads(args.threads)
        report['measurement_protocol'] = {
            'input': [1, 3, 640, 640], 'scope': 'both full detector and backbone',
            'torch': torch.__version__, 'device': args.device, 'amp': args.amp,
            'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'seed': args.seed, 'model_only_import': args.model_only_import,
            'pretrained_download': False, 'checkpoint': None, 'deploy': False,
            'flops_scope': 'counted-operator LOWER BOUNDS, not complete GFLOPs',
            'not_counted': 'FFT, deformable sampling/convolution, transposed convolutions, '
                           'some fused attention, normalization and elementwise math',
        }
        results = {}
        for method in METHODS:
            results[method] = measure(method, args)
            print(method, json.dumps({k: v for k, v in results[method].items()
                                     if 'operator_names' not in k and 'operator_flops' not in k}), flush=True)
        baseline = results['Baseline']
        for result in results.values():
            result['added_params'] = result['params'] - baseline['params']
            for field in ('counted_operator_gflops_lower_bound',
                          'backbone_counted_operator_gflops_lower_bound'):
                result['added_' + field] = (result[field] - baseline[field]
                                            if result[field] is not None else None)
        report['results'] = results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
        print(f'Saved {args.output}', flush=True)
    elif args.resolved_only:
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
