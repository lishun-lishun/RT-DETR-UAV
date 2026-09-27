"""Evaluate every immediate child experiment's best.pth and record metrics.

Each experiment directory is matched to a YAML with the same basename. Tests
run sequentially on one visible GPU to bound CUDA and shared-memory use. A
failure is recorded and does not prevent evaluation of the remaining models.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / 'output' / 'three_gpu_b16_warmup_cosine')
DEFAULT_CONFIG_DIR = PROJECT_ROOT / 'configs' / 'rtdetr'
METRIC_FIELDS = (
    'map_50_95', 'map50', 'map75', 'ap_small', 'ap_medium', 'ap_large',
    'ar_1', 'ar_10', 'ar_100', 'ar_small', 'ar_medium', 'ar_large')

COCO_LINE = re.compile(
    r'Average\s+(?:Precision|Recall)\s+\((AP|AR)\)\s+@\[\s*'
    r'IoU=\s*([^|]+?)\s*\|\s*area=\s*([^|]+?)\s*\|\s*'
    r'maxDets=\s*(\d+)\s*\]\s*=\s*([-+]?\d+(?:\.\d+)?)')


def _clean(value):
    return ''.join(value.split())


def parse_coco_metrics(text):
    metrics = {}
    for kind, iou, area, max_dets, value in COCO_LINE.findall(text):
        iou, area = _clean(iou), _clean(area)
        value = float(value)
        if kind == 'AP' and max_dets == '100':
            if area == 'all' and iou == '0.50:0.95':
                metrics['map_50_95'] = value
            elif area == 'all' and iou == '0.50':
                metrics['map50'] = value
            elif area == 'all' and iou == '0.75':
                metrics['map75'] = value
            elif iou == '0.50:0.95' and area in ('small', 'medium', 'large'):
                metrics[f'ap_{area}'] = value
        elif kind == 'AR' and iou == '0.50:0.95':
            if area == 'all' and max_dets in ('1', '10', '100'):
                metrics[f'ar_{max_dets}'] = value
            elif max_dets == '100' and area in ('small', 'medium', 'large'):
                metrics[f'ar_{area}'] = value
    return metrics


def find_config(config_dir, experiment_name):
    direct = config_dir / f'{experiment_name}.yml'
    if direct.is_file():
        return direct
    matches = sorted(config_dir.rglob(f'{experiment_name}.yml'))
    return matches[0] if len(matches) == 1 else None


def write_reports(report_dir, metadata, records):
    fields = [
        'index', 'experiment', 'status', 'split', *METRIC_FIELDS,
        'duration_sec', 'return_code', 'config', 'checkpoint', 'log', 'message']
    with (report_dir / 'summary.csv').open(
            'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    payload = {**metadata, 'records': records}
    (report_dir / 'summary.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# RT-DETR best.pth evaluation', '',
        f'- Generated: {metadata["generated_at"]}',
        f'- Split: `{metadata["split"]}`',
        f'- Output root: `{metadata["output_root"]}`',
        f'- Total folders: {metadata["total"]}',
        f'- PASS: {sum(row["status"] == "PASS" for row in records)}',
        f'- Failed/skipped: {sum(row["status"] != "PASS" for row in records)}',
        '',
        '| # | Experiment | Status | mAP50-95 | mAP50 | mAP75 | AP small | Seconds |',
        '|---:|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in records:
        value = lambda key: (f'{row[key]:.4f}'
                             if isinstance(row.get(key), float) else '-')
        lines.append(
            f'| {row["index"]} | {row["experiment"]} | {row["status"]} | '
            f'{value("map_50_95")} | {value("map50")} | {value("map75")} | '
            f'{value("ap_small")} | {row.get("duration_sec", 0):.1f} |')

    ranked = sorted(
        (row for row in records
         if row['status'] == 'PASS' and isinstance(row.get('map_50_95'), float)),
        key=lambda row: row['map_50_95'], reverse=True)
    lines.extend(['', '## Ranking by mAP50-95', '',
                  '| Rank | Experiment | mAP50-95 | mAP50 | mAP75 |',
                  '|---:|---|---:|---:|---:|'])
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f'| {rank} | {row["experiment"]} | {row["map_50_95"]:.4f} | '
            f'{row.get("map50", float("nan")):.4f} | '
            f'{row.get("map75", float("nan")):.4f} |')
    (report_dir / 'summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')


def run_and_tee(command, environment, log_path):
    lines = []
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
        return_code = process.wait()
    return return_code, ''.join(lines)


def main(args):
    output_root = Path(args.root).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f'Experiment output root not found: {output_root}')
    if not config_dir.is_dir():
        raise FileNotFoundError(f'Config directory not found: {config_dir}')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')

    experiments = sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower())
    plan = []
    for directory in experiments:
        checkpoint = directory / 'best.pth'
        config = find_config(config_dir, directory.name)
        status = ('RUN' if checkpoint.is_file() and config is not None else
                  'MISSING_BEST' if not checkpoint.is_file() else
                  'MISSING_CONFIG')
        plan.append((directory, checkpoint, config, status))

    print('=' * 72)
    print('All best.pth evaluation plan')
    print(f'Output root: {output_root}')
    print(f'Split: {args.split}; GPU: {args.gpu}; workers: {args.num_workers}')
    print(f'Total folders: {len(plan)}')
    print(f'Will run: {sum(item[3] == "RUN" for item in plan)}')
    print(f'Will record missing: {sum(item[3] != "RUN" for item in plan)}')
    print('=' * 72)
    for index, (directory, checkpoint, config, status) in enumerate(plan, 1):
        print(f'{index:02d}. [{status}] {directory.name}')
        print(f'    checkpoint={checkpoint}')
        print(f'    config={config or "NOT FOUND"}')
    print('=' * 72)
    if args.dry_run:
        print('DRY RUN complete: no test process or report directory was created.')
        return 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.report_dir:
        report_dir = Path(args.report_dir).expanduser().resolve()
    else:
        report_dir = output_root.parent / f'{output_root.name}_test_results' / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = report_dir / 'logs'
    eval_dir = report_dir / 'eval_artifacts'
    logs_dir.mkdir()
    eval_dir.mkdir()

    metadata = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'output_root': str(output_root), 'config_dir': str(config_dir),
        'report_dir': str(report_dir), 'split': args.split,
        'gpu': args.gpu, 'num_workers': args.num_workers,
        'total': len(plan),
    }
    records = []
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.gpu
    environment['OMP_NUM_THREADS'] = environment.get('OMP_NUM_THREADS', '1')
    environment['PYTHONUNBUFFERED'] = '1'

    for index, (directory, checkpoint, config, plan_status) in enumerate(plan, 1):
        base = {
            'index': index, 'experiment': directory.name,
            'split': args.split, 'checkpoint': str(checkpoint),
            'config': str(config) if config else '', 'return_code': '',
            'duration_sec': 0.0, 'log': '', 'message': '',
        }
        if plan_status != 'RUN':
            base['status'] = plan_status
            base['message'] = ('best.pth not found' if plan_status == 'MISSING_BEST'
                               else 'matching YAML not found')
            records.append(base)
            write_reports(report_dir, metadata, records)
            continue

        log_path = logs_dir / f'{index:02d}_{directory.name}.log'
        artifact_path = eval_dir / directory.name
        command = [
            sys.executable, str(PROJECT_ROOT / 'tools' / 'test_dut.py'),
            '-c', str(config), '-r', str(checkpoint),
            '--split', args.split, '--num-workers', str(args.num_workers),
            '--output-dir', str(artifact_path),
        ]
        print('\n' + '=' * 72)
        print(f'[TEST {index}/{len(plan)}] {directory.name}')
        print('Command:', ' '.join(command))
        print('=' * 72)
        started = time.monotonic()
        return_code, output = run_and_tee(command, environment, log_path)
        duration = time.monotonic() - started
        metrics = parse_coco_metrics(output)
        base.update(metrics)
        base.update(return_code=return_code, duration_sec=round(duration, 3),
                    log=str(log_path))
        if return_code != 0:
            base['status'] = 'FAILED'
            base['message'] = f'evaluation exited with code {return_code}'
        elif 'map_50_95' not in metrics:
            base['status'] = 'METRICS_NOT_PARSED'
            base['message'] = 'COCO summary was not found in stdout'
        else:
            base['status'] = 'PASS'
        records.append(base)
        write_reports(report_dir, metadata, records)
        print(f'[{base["status"]}] {directory.name}: '
              f'mAP50-95={base.get("map_50_95", "-")} '
              f'mAP50={base.get("map50", "-")} '
              f'mAP75={base.get("map75", "-")}')

    passed = sum(record['status'] == 'PASS' for record in records)
    print('\n' + '=' * 72)
    print(f'Evaluation complete: PASS={passed}, other={len(records) - passed}')
    print(f'CSV:  {report_dir / "summary.csv"}')
    print(f'JSON: {report_dir / "summary.json"}')
    print(f'Markdown: {report_dir / "summary.md"}')
    print('=' * 72)
    return 0 if passed == len(records) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(DEFAULT_OUTPUT_ROOT),
                        help='directory whose immediate children are experiments')
    parser.add_argument('--config-dir', default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    parser.add_argument('--gpu', default='1',
                        help='one physical GPU ID, for example 1')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--report-dir', default=None)
    parser.add_argument('--dry-run', action='store_true')
    sys.exit(main(parser.parse_args()))
