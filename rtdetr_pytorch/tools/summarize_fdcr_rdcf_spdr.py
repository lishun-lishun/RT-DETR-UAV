"""Summarize the fixed FDCR/RDCF/SPDR comparison from test_all_best.py.

The input is the ``summary.json`` emitted by ``tools/test_all_best.py``.  This
script deliberately expects the final seven-experiment order so an accidental
comparison against the wrong backbone baseline cannot silently produce a
plausible-looking gain table.

Metric values in JSON stay in COCO's native 0--1 units.  Markdown renders them
as percentages and renders differences as percentage points.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any


EXPERIMENTS = (
    'rtdetr_r18vd_dut_anti_uav',
    'rtdetr_r18vd_dut_anti_uav_fdcr',
    'rtdetr_r18vd_dut_anti_uav_rdcf',
    'rtdetr_hrnetv2_w18_dut_anti_uav',
    'rtdetr_hrnetv2_w18_dut_anti_uav_fdcr',
    'rtdetr_hrnetv2_w18_dut_anti_uav_rdcf',
    'rtdetr_hrnetv2_w18_dut_anti_uav_spdr',
)

PRES_BASELINE = EXPERIMENTS[0]
HR_BASELINE = EXPERIMENTS[3]

BASELINES = {
    EXPERIMENTS[0]: PRES_BASELINE,
    EXPERIMENTS[1]: PRES_BASELINE,
    EXPERIMENTS[2]: PRES_BASELINE,
    EXPERIMENTS[3]: HR_BASELINE,
    EXPERIMENTS[4]: HR_BASELINE,
    EXPERIMENTS[5]: HR_BASELINE,
    EXPERIMENTS[6]: HR_BASELINE,
}

FAMILIES = {
    EXPERIMENTS[0]: 'PResNet18',
    EXPERIMENTS[1]: 'PResNet18',
    EXPERIMENTS[2]: 'PResNet18',
    EXPERIMENTS[3]: 'HRNetV2-W18',
    EXPERIMENTS[4]: 'HRNetV2-W18',
    EXPERIMENTS[5]: 'HRNetV2-W18',
    EXPERIMENTS[6]: 'HRNetV2-W18',
}

VARIANTS = {
    EXPERIMENTS[0]: 'Baseline',
    EXPERIMENTS[1]: 'FDCR',
    EXPERIMENTS[2]: 'RDCF',
    EXPERIMENTS[3]: 'Baseline',
    EXPERIMENTS[4]: 'FDCR',
    EXPERIMENTS[5]: 'RDCF',
    EXPERIMENTS[6]: 'SPDR',
}

# Public report names mapped to the actual test_all_best.py summary fields.
METRICS = (
    ('AP', 'map_50_95'),
    ('AP50', 'map50'),
    ('AP75', 'map75'),
    ('APS', 'ap_small'),
    ('APM', 'ap_medium'),
    ('APL', 'ap_large'),
    ('AR1', 'ar_1'),
    ('AR10', 'ar_10'),
    ('AR100', 'ar_100'),
    ('ARS', 'ar_small'),
    ('ARM', 'ar_medium'),
    ('ARL', 'ar_large'),
)
GAIN_METRICS = ('AP', 'AP75', 'APS', 'ARS')

DEFAULT_JSON_NAME = 'fdcr_rdcf_spdr_comparison.json'
DEFAULT_MARKDOWN_NAME = 'fdcr_rdcf_spdr_comparison.md'


def _metric(value: Any) -> float | None:
    """Return a finite numeric metric, otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_protocol(summary: dict[str, Any]) -> None:
    """Reject reports that do not follow the fixed evaluation protocol."""
    problems = []
    if summary.get('split') != 'test':
        problems.append(f"split={summary.get('split')!r}, expected 'test'")
    if str(summary.get('gpu')) != '1':
        problems.append(f"gpu={summary.get('gpu')!r}, expected '1'")
    if summary.get('num_workers') != 2:
        problems.append(
            f"num_workers={summary.get('num_workers')!r}, expected 2")
    if summary.get('require_ema') is not True:
        problems.append('require_ema is not true')
    if problems:
        raise ValueError(
            'summary.json does not match the fixed test protocol: '
            + '; '.join(problems))


def build_comparison(summary: dict[str, Any], source: Path) -> dict[str, Any]:
    """Build normalized metric and per-backbone gain records."""
    if not isinstance(summary, dict):
        raise ValueError('summary.json root must be an object')
    _validate_protocol(summary)
    source_records = summary.get('records')
    if not isinstance(source_records, list):
        raise ValueError('summary.json records must be a list')

    names = [row.get('experiment') for row in source_records
             if isinstance(row, dict)]
    if names != list(EXPERIMENTS):
        raise ValueError(
            'summary.json experiment order/set mismatch; expected exactly: '
            + ', '.join(EXPERIMENTS))

    normalized = []
    by_name = {}
    for source_row in source_records:
        name = source_row['experiment']
        metrics = {
            public: _metric(source_row.get(field))
            for public, field in METRICS
        }
        row = {
            'index': source_row.get('index'),
            'experiment': name,
            'family': FAMILIES[name],
            'variant': VARIANTS[name],
            'baseline_experiment': BASELINES[name],
            'status': source_row.get('status', 'UNKNOWN'),
            'metrics': metrics,
            'metrics_complete': all(value is not None
                                    for value in metrics.values()),
            'gains_vs_baseline': {},
            'gains_vs_baseline_percentage_points': {},
            'checkpoint': source_row.get('checkpoint'),
            'config': source_row.get('config'),
            'message': source_row.get('message', ''),
        }
        normalized.append(row)
        by_name[name] = row

    for row in normalized:
        baseline = by_name[row['baseline_experiment']]
        for metric_name in GAIN_METRICS:
            value = row['metrics'][metric_name]
            baseline_value = baseline['metrics'][metric_name]
            if (row['status'] == 'PASS' and baseline['status'] == 'PASS'
                    and value is not None and baseline_value is not None):
                gain = value - baseline_value
                # Avoid noisy -0.0 in JSON/Markdown baseline rows.
                if gain == 0:
                    gain = 0.0
                gain_pp = gain * 100.0
            else:
                gain = None
                gain_pp = None
            row['gains_vs_baseline'][metric_name] = gain
            row['gains_vs_baseline_percentage_points'][metric_name] = gain_pp

    all_pass = all(row['status'] == 'PASS' and row['metrics_complete']
                   for row in normalized)
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'source_summary': str(source.resolve()),
        'metric_unit': 'fraction',
        'gain_unit': 'fraction',
        'protocol': {
            'checkpoint': 'best.pth',
            'split': 'test',
            'gpu': '1',
            'num_workers': 2,
            'require_ema': True,
            'precision': 'FP32',
            'input_size': [640, 640],
        },
        'all_pass': all_pass,
        'records': normalized,
    }


def _percent(value: float | None) -> str:
    return '-' if value is None else f'{value * 100.0:.2f}'


def _points(value: float | None) -> str:
    return '-' if value is None else f'{value:+.2f}'


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact full-metric table plus the requested gain table."""
    lines = [
        '# FDCR / RDCF / SPDR unified DUT test comparison',
        '',
        '- Checkpoint: `best.pth` (EMA required)',
        '- Split: `test`',
        '- Input: `640x640`',
        '- Precision: `FP32`',
        '- GPU: physical GPU `1`',
        '- DataLoader workers: `2`',
        '- AP/AR columns are percentages; gain columns are percentage points.',
        '',
        '## Complete COCO metrics',
        '',
        '| # | Backbone | Variant | Status | AP | AP50 | AP75 | APS | APM | APL | AR1 | AR10 | AR100 | ARS | ARM | ARL |',
        '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in payload['records']:
        values = ' | '.join(_percent(row['metrics'][name])
                            for name, _ in METRICS)
        lines.append(
            f"| {row['index']} | {row['family']} | {row['variant']} | "
            f"{row['status']} | {values} |")

    lines.extend([
        '',
        '## Gains relative to the matching backbone baseline',
        '',
        '| Backbone | Variant | Baseline | Status | AP gain | AP75 gain | APS gain | ARS gain |',
        '|---|---|---|---|---:|---:|---:|---:|',
    ])
    for row in payload['records']:
        gains = row['gains_vs_baseline_percentage_points']
        lines.append(
            f"| {row['family']} | {row['variant']} | "
            f"`{row['baseline_experiment']}` | {row['status']} | "
            f"{_points(gains['AP'])} | {_points(gains['AP75'])} | "
            f"{_points(gains['APS'])} | {_points(gains['ARS'])} |")

    failed = [row for row in payload['records']
              if row['status'] != 'PASS' or not row['metrics_complete']]
    if failed:
        lines.extend(['', '## Incomplete evaluations', ''])
        for row in failed:
            missing = [name for name, value in row['metrics'].items()
                       if value is None]
            detail = row.get('message') or (
                'missing metrics: ' + ', '.join(missing)
                if missing else 'no additional message')
            lines.append(
                f"- `{row['experiment']}`: `{row['status']}` — {detail}")
    return '\n'.join(lines) + '\n'


def generate(summary_path: Path, output_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    """Read, normalize, and write both machine- and human-readable reports."""
    summary_path = summary_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f'test summary not found: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
    payload = build_comparison(summary, summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / DEFAULT_JSON_NAME
    markdown_path = output_dir / DEFAULT_MARKDOWN_NAME
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    markdown_path.write_text(render_markdown(payload), encoding='utf-8')
    return payload, json_path, markdown_path


def main(args: argparse.Namespace) -> int:
    summary = Path(args.summary)
    output_dir = Path(args.output_dir) if args.output_dir else summary.parent
    payload, json_path, markdown_path = generate(summary, output_dir)
    print(f'Comparison JSON: {json_path}')
    print(f'Comparison Markdown: {markdown_path}')
    passed = sum(row['status'] == 'PASS' and row['metrics_complete']
                 for row in payload['records'])
    print(f'Comparison completeness: PASS={passed}, other={len(EXPERIMENTS) - passed}')
    return 0 if payload['all_pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True,
                        help='summary.json written by tools/test_all_best.py')
    parser.add_argument('--output-dir', default=None,
                        help='default: the directory containing summary.json')
    sys.exit(main(parser.parse_args()))
