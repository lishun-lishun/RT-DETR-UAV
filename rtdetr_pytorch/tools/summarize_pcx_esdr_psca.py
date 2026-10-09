"""Validate and summarize the fixed PCX/ESDR/PSCA unified test."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys


EXPERIMENTS = (
    'rtdetr_r18vd_dut_anti_uav',
    'rtdetr_r18vd_dut_anti_uav_pcx',
    'rtdetr_r18vd_dut_anti_uav_esdr',
    'rtdetr_r18vd_dut_anti_uav_psca',
    'rtdetr_hrnetv2_w18_dut_anti_uav',
    'rtdetr_hrnetv2_w18_dut_anti_uav_pcx',
    'rtdetr_hrnetv2_w18_dut_anti_uav_esdr',
    'rtdetr_hrnetv2_w18_dut_anti_uav_psca',
)

PRES_BASELINE = EXPERIMENTS[0]
HR_BASELINE = EXPERIMENTS[4]
BASELINES = {
    name: (HR_BASELINE if name.startswith('rtdetr_hrnetv2') else PRES_BASELINE)
    for name in EXPERIMENTS
}
FAMILIES = {
    name: ('HRNetV2-W18' if name.startswith('rtdetr_hrnetv2') else 'PResNet18')
    for name in EXPERIMENTS
}


def _variant(name):
    for method in ('pcx', 'esdr', 'psca'):
        if name.endswith('_' + method):
            return method.upper()
    return 'Baseline'


METRICS = (
    ('AP', 'map_50_95'), ('AP50', 'map50'), ('AP75', 'map75'),
    ('APS', 'ap_small'), ('APM', 'ap_medium'), ('APL', 'ap_large'),
    ('AR1', 'ar_1'), ('AR10', 'ar_10'), ('AR100', 'ar_100'),
    ('ARS', 'ar_small'), ('ARM', 'ar_medium'), ('ARL', 'ar_large'),
)
GAIN_METRICS = ('AP', 'AP75', 'APS', 'ARS')
DEFAULT_JSON_NAME = 'pcx_esdr_psca_comparison.json'
DEFAULT_MARKDOWN_NAME = 'pcx_esdr_psca_comparison.md'


def _finite_metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _validate_protocol(summary):
    problems = []
    if summary.get('split') != 'test':
        problems.append("split must be 'test'")
    if str(summary.get('gpu')) != '1':
        problems.append("gpu must be '1'")
    if summary.get('num_workers') != 2:
        problems.append('num_workers must be 2')
    if summary.get('require_ema') is not True:
        problems.append('EMA checkpoint is required')
    if problems:
        raise ValueError('summary.json violates fixed test protocol: ' + '; '.join(problems))


def build_comparison(summary, source):
    if not isinstance(summary, dict):
        raise ValueError('summary.json root must be an object')
    _validate_protocol(summary)
    records = summary.get('records')
    if not isinstance(records, list):
        raise ValueError('summary.json records must be a list')
    names = [row.get('experiment') for row in records if isinstance(row, dict)]
    if names != list(EXPERIMENTS):
        raise ValueError('summary.json experiment order/set mismatch')

    normalized = []
    by_name = {}
    for source_row in records:
        name = source_row['experiment']
        metrics = {public: _finite_metric(source_row.get(field))
                   for public, field in METRICS}
        row = {
            'index': source_row.get('index'),
            'experiment': name,
            'family': FAMILIES[name],
            'variant': _variant(name),
            'baseline_experiment': BASELINES[name],
            'status': source_row.get('status', 'UNKNOWN'),
            'metrics': metrics,
            'metrics_complete': all(value is not None for value in metrics.values()),
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
        for metric in GAIN_METRICS:
            value, base = row['metrics'][metric], baseline['metrics'][metric]
            gain = (value - base if row['status'] == baseline['status'] == 'PASS'
                    and value is not None and base is not None else None)
            if gain == 0:
                gain = 0.0
            row['gains_vs_baseline'][metric] = gain
            row['gains_vs_baseline_percentage_points'][metric] = (
                None if gain is None else gain * 100.0)

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'source_summary': str(Path(source).resolve()),
        'metric_unit': 'fraction',
        'gain_unit': 'fraction',
        'protocol': {
            'checkpoint': 'best.pth', 'split': 'test', 'gpu': '1',
            'num_workers': 2, 'require_ema': True,
            'precision': 'FP32', 'input_size': [640, 640],
        },
        'all_pass': all(row['status'] == 'PASS' and row['metrics_complete']
                        for row in normalized),
        'records': normalized,
    }


def _percent(value):
    return '-' if value is None else f'{value * 100.0:.2f}'


def _points(value):
    return '-' if value is None else f'{value:+.2f}'


def render_markdown(payload):
    lines = [
        '# PCX / ESDR / PSCA unified DUT test comparison', '',
        '- Protocol: `best.pth`, EMA required, `test`, 640x640, FP32, GPU 1, workers 2.',
        '- AP/AR values are percentages; gains are percentage points.', '',
        '## Complete COCO metrics', '',
        '| # | Backbone | Variant | Status | AP | AP50 | AP75 | APS | APM | APL | AR1 | AR10 | AR100 | ARS | ARM | ARL |',
        '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in payload['records']:
        values = ' | '.join(_percent(row['metrics'][name]) for name, _ in METRICS)
        lines.append(f"| {row['index']} | {row['family']} | {row['variant']} | "
                     f"{row['status']} | {values} |")
    lines.extend([
        '', '## Gains relative to the matching backbone baseline', '',
        '| Backbone | Variant | Baseline | Status | AP gain | AP75 gain | APS gain | ARS gain |',
        '|---|---|---|---|---:|---:|---:|---:|',
    ])
    for row in payload['records']:
        gains = row['gains_vs_baseline_percentage_points']
        lines.append(f"| {row['family']} | {row['variant']} | `{row['baseline_experiment']}` | "
                     f"{row['status']} | {_points(gains['AP'])} | "
                     f"{_points(gains['AP75'])} | {_points(gains['APS'])} | "
                     f"{_points(gains['ARS'])} |")
    failed = [row for row in payload['records']
              if row['status'] != 'PASS' or not row['metrics_complete']]
    if failed:
        lines.extend(['', '## Incomplete evaluations', ''])
        for row in failed:
            missing = [name for name, value in row['metrics'].items() if value is None]
            detail = row['message'] or 'missing metrics: ' + ', '.join(missing)
            lines.append(f"- `{row['experiment']}`: `{row['status']}` - {detail}")
    return '\n'.join(lines) + '\n'


def generate(summary_path, output_dir):
    summary_path = Path(summary_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
    payload = build_comparison(summary, summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / DEFAULT_JSON_NAME
    markdown_path = output_dir / DEFAULT_MARKDOWN_NAME
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    markdown_path.write_text(render_markdown(payload), encoding='utf-8')
    return payload, json_path, markdown_path


def main(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.summary).parent
    payload, json_path, markdown_path = generate(args.summary, output_dir)
    print(f'Comparison JSON: {json_path}')
    print(f'Comparison Markdown: {markdown_path}')
    return 0 if payload['all_pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output-dir')
    sys.exit(main(parser.parse_args()))
