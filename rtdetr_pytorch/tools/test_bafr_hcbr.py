"""Run BAFR/HCBR synthetic and integration checks without training.

Examples:
    python tools/test_bafr_hcbr.py
    python tools/test_bafr_hcbr.py --test-bafr --test-hcbr
    python tools/test_bafr_hcbr.py --test-baseline-equivalence --test-grad --test-amp
"""

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _iter_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-bafr', action='store_true')
    parser.add_argument('--test-hcbr', action='store_true')
    parser.add_argument('--test-baseline-equivalence', action='store_true')
    parser.add_argument('--test-grad', action='store_true')
    parser.add_argument('--test-amp', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    all_tests = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT / 'tests'), pattern='test_bafr_hcbr.py',
        top_level_dir=str(ROOT))
    selected = unittest.TestSuite()
    flags = any(vars(args).values())
    for case in _iter_cases(all_tests):
        name = case.id().lower()
        if not flags or any((
            args.test_bafr and 'bafrtests' in name,
            args.test_hcbr and 'hcbrtests' in name,
            args.test_baseline_equivalence and ('baseline_equivalence' in name
                                                or 'zero_init_equivalence' in name),
            args.test_grad and 'gradient' in name,
            args.test_amp and 'amp' in name,
        )):
            selected.addTest(case)
    if selected.countTestCases() == 0:
        parser.error('No tests match the selected flags')
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
