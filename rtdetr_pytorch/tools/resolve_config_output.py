"""Resolve a YAML's final output directory with RT-DETR's config loader.

Only yaml_utils is loaded: planning never constructs a model, downloads a
weight, creates a directory, or launches a DataLoader. A command-line output
override has the same precedence as tools/train.py.
"""

import argparse
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def resolve_output_dir(config_path, output_override=None):
    loader = runpy.run_path(str(ROOT / 'src' / 'core' / 'yaml_utils.py'))
    config = loader['load_config'](str(config_path), {})
    output_dir = output_override if output_override else config.get('output_dir')
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f'No valid output_dir resolved for {config_path}')
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output-dir', default=None,
                        help='final CLI override, equivalent to tools/train.py')
    args = parser.parse_args()
    print(resolve_output_dir(args.config, args.output_dir))


if __name__ == '__main__':
    main()
