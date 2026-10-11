"""Validate and summarize the fixed HRDI/SAQS/SGR test matrix."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys


EXPERIMENTS = (
    'rtdetr_hrnetv2_w18_dut_anti_uav',
    'rtdetr_hrnetv2_w18_dut_anti_uav_sgr',
    'rtdetr_hrnetv2_w18_dut_anti_uav_hrdi',
    'rtdetr_hrnetv2_w18_dut_anti_uav_saqs',
    'rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_sgr',
    'rtdetr_hrnetv2_w18_dut_anti_uav_saqs_sgr',
    'rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_saqs',
    'rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_saqs_sgr',
)
BASELINE = EXPERIMENTS[0]
METRICS = (
    ('AP', 'map_50_95'), ('AP50', 'map50'), ('AP75', 'map75'),
    ('APS', 'ap_small'), ('APM', 'ap_medium'), ('APL', 'ap_large'),
    ('AR1', 'ar_1'), ('AR10', 'ar_10'), ('AR100', 'ar_100'),
    ('ARS', 'ar_small'), ('ARM', 'ar_medium'), ('ARL', 'ar_large'),
)
GAIN_METRICS = ('AP', 'AP75', 'APS', 'ARS')


def variant(name):
    suffix = name.removeprefix(BASELINE).strip('_')
    return 'Baseline' if not suffix else suffix.upper().replace('_', '+')


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def build(summary, source):
    if summary.get('split') != 'test':
        raise ValueError("summary split must be 'test'")
    if summary.get('require_ema') is not True:
        raise ValueError('summary must require EMA checkpoints')
    records = summary.get('records')
    names = [row.get('experiment') for row in records or []]
    if names != list(EXPERIMENTS):
        raise ValueError('summary experiment order/set mismatch')

    rows = []
    for raw in records:
        values = {label: finite(raw.get(field)) for label, field in METRICS}
        rows.append({
            'index': raw.get('index'),
            'experiment': raw['experiment'],
            'variant': variant(raw['experiment']),
            'status': raw.get('status', 'UNKNOWN'),
            'metrics': values,
            'metrics_complete': all(value is not None for value in values.values()),
            'gains_vs_baseline_percentage_points': {},
            'checkpoint': raw.get('checkpoint'),
            'config': raw.get('config'),
            'message': raw.get('message', ''),
        })
    base = rows[0]
    for row in rows:
        for metric in GAIN_METRICS:
            value, reference = row['metrics'][metric], base['metrics'][metric]
            gain = (100.0 * (value - reference)
                    if row['status'] == base['status'] == 'PASS'
                    and value is not None and reference is not None else None)
            row['gains_vs_baseline_percentage_points'][metric] = gain
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'source_summary': str(source),
        'protocol': {
            'backbone': 'HRNetV2-W18', 'checkpoint': 'best.pth',
            'split': 'test', 'precision': 'FP32', 'input_size': [640, 640],
        },
        'all_pass': all(row['status'] == 'PASS' and row['metrics_complete']
                        for row in rows),
        'records': rows,
    }


def percent(value):
    return '-' if value is None else f'{100.0 * value:.2f}'


def points(value):
    return '-' if value is None else f'{value:+.2f}'


def markdown(payload):
    lines = [
        '# HRDI / SAQS / SGR DUT test comparison', '',
        '- Protocol: `best.pth`, EMA required, true `test`, 640x640, FP32.',
        '- AP/AR values are percentages; gains are percentage points.', '',
        '| # | Variant | Status | AP | AP50 | AP75 | APS | APM | APL | AR1 | AR10 | AR100 | ARS | ARM | ARL |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in payload['records']:
        values = ' | '.join(percent(row['metrics'][name]) for name, _ in METRICS)
        lines.append(f"| {row['index']} | {row['variant']} | {row['status']} | {values} |")
    lines.extend([
        '', '## Gains versus HRNetV2-W18 baseline', '',
        '| Variant | AP | AP75 | APS | ARS |',
        '|---|---:|---:|---:|---:|',
    ])
    for row in payload['records']:
        gain = row['gains_vs_baseline_percentage_points']
        lines.append(f"| {row['variant']} | {points(gain['AP'])} | "
                     f"{points(gain['AP75'])} | {points(gain['APS'])} | "
                     f"{points(gain['ARS'])} |")
    return '\n'.join(lines) + '\n'


def main(args):
    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = (Path(args.output_dir).expanduser().resolve()
                  if args.output_dir else summary_path.parent)
    payload = build(json.loads(summary_path.read_text(encoding='utf-8-sig')),
                    summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'hrdi_saqs_sgr_comparison.json'
    md_path = output_dir / 'hrdi_saqs_sgr_comparison.md'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    md_path.write_text(markdown(payload), encoding='utf-8')
    print(f'Comparison JSON: {json_path}')
    print(f'Comparison Markdown: {md_path}')
    return 0 if payload['all_pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output-dir')
    sys.exit(main(parser.parse_args()))
