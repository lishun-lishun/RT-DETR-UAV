"""Pure model-forward CUDA-event benchmark; no loader/evaluator/postprocess."""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.misc.amp import autocast_context
from src.misc.inference_audit import InferenceAudit


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def benchmark(args, config_path):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for CUDA-event timing')
    if args.batch_size < 1 or args.warmup < 0 or args.iters < 1:
        raise ValueError('Invalid batch-size/warmup/iters')
    torch.manual_seed(args.seed)
    cfg = YAMLConfig(config_path, use_amp=args.amp)
    # Skip redundant initialization downloads before strict checkpoint loading.
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint and 'module' in checkpoint['ema']:
        state, source = checkpoint['ema']['module'], 'ema.module'
    elif 'model' in checkpoint:
        state, source = checkpoint['model'], 'model'
    else:
        state, source = checkpoint, 'raw state_dict'
    model.load_state_dict(state, strict=True)
    size = cfg.yaml_cfg['RTDETRTransformer'].get('eval_spatial_size') or [800, 800]
    height, width = args.input_size if args.input_size else size
    if height < 1 or width < 1:
        raise ValueError('input-size dimensions must be positive')
    # This model caches anchors/position embeddings for its configured shape.
    for component in ('HybridEncoder', 'RTDETRTransformer'):
        cached_size = cfg.yaml_cfg.get(component, {}).get('eval_spatial_size')
        if cached_size is not None and list(cached_size) != [height, width]:
            raise ValueError('input-size must match {}.eval_spatial_size'.format(component))
    images = torch.randn(args.batch_size, 3, height, width, device='cuda')

    with torch.no_grad():
        # Audit separately so debug hooks do not contaminate measured latency.
        if args.debug_eval_amp:
            with InferenceAudit(model) as audit:
                with autocast_context(images.device, args.amp):
                    model(images)
            audit.report(expected_amp=args.amp)
        for _ in range(args.warmup):
            with autocast_context(images.device, args.amp):
                model(images)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        for start, end in zip(starts, ends):
            start.record()
            with autocast_context(images.device, args.amp):
                model(images)
            end.record()
        torch.cuda.synchronize()

    latencies = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    mean = statistics.mean(latencies)
    result = {
        'config': config_path,
        'checkpoint': args.resume,
        'weights': source,
        'input_size': [height, width],
        'batch_size': args.batch_size,
        'amp': args.amp,
        'params': sum(parameter.numel() for parameter in model.parameters()),
        'mean_latency_ms': mean,
        'median_latency_ms': statistics.median(latencies),
        'p95_latency_ms': percentile(latencies, 95),
        'fps': args.batch_size * 1000.0 / mean,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'warmup': args.warmup,
        'iterations': args.iters,
    }
    return result


def main(args):
    configs = [args.config] + (args.compare_configs or [])
    results = [benchmark(args, config_path) for config_path in configs]
    if len(results) > 1:
        # Reuse exactly the same checkpoint in every configuration: compare
        # inference structure, not training quality or dataset wall-clock time.
        if len({result['params'] for result in results}) != 1:
            raise RuntimeError('Comparison configurations have different parameter counts')
        if len({tuple(result['input_size']) for result in results}) != 1:
            raise RuntimeError('Comparison configurations have different input sizes')
        print('Variant | Params | Mean ms | Median ms | P95 ms | FPS')
        for result in results:
            print('{} | {} | {:.3f} | {:.3f} | {:.3f} | {:.2f}'.format(
                os.path.basename(result['config']), result['params'],
                result['mean_latency_ms'], result['median_latency_ms'],
                result['p95_latency_ms'], result['fps'],
            ))
        result = {'same_params': True, 'variants': results}
    else:
        result = results[0]
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--resume', '-r', required=True)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--debug-eval-amp', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--input-size', nargs=2, type=int, metavar=('H', 'W'))
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--output', help='optional benchmark JSON file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compare-configs', nargs='+',
                        help='additional configs to benchmark using the same checkpoint')
    main(parser.parse_args())
