"""Evaluate a DUT checkpoint on the real test (or val) split, in original FP32.

The original tools/train.py --test-only evaluates val_dataloader. This optional
entry point changes ONLY its dataset paths, keeping original evaluate intact.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def select_split(cfg, split):
    if split == 'test':
        paths = cfg.yaml_cfg.get('test_dataset')
        if not paths or not all(key in paths for key in ('img_folder', 'ann_file')):
            raise ValueError('The config must define test_dataset.img_folder/ann_file.')
        cfg.yaml_cfg['val_dataloader']['dataset'].update(paths)
    dataset = cfg.yaml_cfg['val_dataloader']['dataset']
    return {key: dataset[key] for key in ('img_folder', 'ann_file')}


def main(args):
    from src.core import YAMLConfig
    from src.misc import dist
    from src.solver import TASKS

    dist.init_distributed()
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=False)
    paths = select_split(cfg, args.split)
    print(f'DUT evaluation split={args.split}; dataset={paths}; original FP32 evaluate (no AMP).')
    TASKS[cfg.yaml_cfg['task']](cfg).val()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    main(parser.parse_args())
