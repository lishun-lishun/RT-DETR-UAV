"""Evaluate a DUT checkpoint on the real test (or val) split, in original FP32.

The original tools/train.py --test-only evaluates val_dataloader. This optional
entry point changes ONLY its dataset paths, keeping original evaluate intact.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def select_split(cfg, split):
    if split == 'test':
        paths = cfg.yaml_cfg.get('test_dataset')
        if not paths or not all(key in paths for key in ('img_folder', 'ann_file')):
            raise ValueError('The config must define test_dataset.img_folder/ann_file.')
        cfg.yaml_cfg['val_dataloader']['dataset'].update(paths)
    dataset = cfg.yaml_cfg['val_dataloader']['dataset']
    return {key: dataset[key] for key in ('img_folder', 'ann_file')}


def validate_ema_checkpoint(path):
    """Fail before evaluation when the requested checkpoint has no usable EMA."""
    state = torch.load(path, map_location='cpu')
    ema = state.get('ema') if isinstance(state, dict) else None
    module = ema.get('module') if isinstance(ema, dict) else None
    if not isinstance(module, dict) or not module:
        raise ValueError(
            f'Checkpoint does not contain a usable ema.module state: {path}')
    print(f'EMA checkpoint preflight PASS: {path} ({len(module)} tensors)')
    del state


def main(args):
    from src.core import YAMLConfig
    from src.misc import dist
    from src.solver import TASKS

    if getattr(args, 'require_ema', False):
        validate_ema_checkpoint(args.resume)
    dist.init_distributed()
    overrides = dict(resume=args.resume, use_amp=False)
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    cfg = YAMLConfig(args.config, **overrides)
    if args.num_workers is not None:
        loader = cfg.yaml_cfg['val_dataloader']
        loader['num_workers'] = args.num_workers
        if args.num_workers > 0:
            loader['prefetch_factor'] = 1
            loader['persistent_workers'] = False
        else:
            loader.pop('prefetch_factor', None)
            loader.pop('persistent_workers', None)
    paths = select_split(cfg, args.split)
    print(f'DUT evaluation split={args.split}; dataset={paths}; '
          f'num_workers={cfg.yaml_cfg["val_dataloader"]["num_workers"]}; '
          'original FP32 evaluate (no AMP).')
    TASKS[cfg.yaml_cfg['task']](cfg).val()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--output-dir', default=None,
                        help='write eval.pth away from the training directory')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='override evaluation workers; 0 disables workers')
    parser.add_argument(
        '--require-ema', action='store_true',
        help='fail instead of silently evaluating an unloaded EMA model')
    main(parser.parse_args())
