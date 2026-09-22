"""Summarize RT-DETR JSON-lines training logs and validation trends.

Usage: python tools/analyze_training_log.py output/experiment/log.txt

Epochs are reported as both the repository's zero-based index and a
human-readable one-based completed-epoch number. Trend labels are descriptive
heuristics, not proof of under/overfitting.
"""

import argparse
import json
import math
from pathlib import Path


METRICS = {'AP': 0, 'AP50': 1, 'AP75': 2, 'AP_small': 3,
           'AP_medium': 4, 'AP_large': 5, 'AR1': 6, 'AR10': 7,
           'AR100': 8, 'AR_small': 9, 'AR_medium': 10, 'AR_large': 11}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_stats(entry):
    values = entry.get('test_coco_eval_bbox', entry.get('coco_eval_bbox', []))
    if not isinstance(values, list):
        values = []
    result = {}
    for name, index in METRICS.items():
        value = finite(values[index]) if len(values) > index else None
        result[name] = value if value is not None and value >= 0 else None
    return result


def parse_log(path):
    rows = []
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {error}') from error
            epoch = entry.get('epoch')
            if not isinstance(epoch, int):
                raise ValueError(f'{path}:{line_number}: epoch must be an integer')
            row = {'epoch_index': epoch, 'completed_epoch': epoch + 1,
                   'lr': finite(entry.get('train_lr')),
                   'train_loss': finite(entry.get('train_loss')),
                   'loss_bbox': finite(entry.get('train_loss_bbox')),
                   'loss_giou': finite(entry.get('train_loss_giou')),
                   'loss_vfl': finite(entry.get('train_loss_vfl'))}
            row.update(extract_stats(entry))
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no epoch records')
    # A resumed run may append the same epoch twice; keep its final record.
    latest = {row['epoch_index']: row for row in rows}
    return [latest[key] for key in sorted(latest)]


def best_record(rows, metric):
    valid = [row for row in rows if row[metric] is not None]
    if not valid:
        return None
    winner = max(valid, key=lambda row: row[metric])
    return {'epoch_index': winner['epoch_index'],
            'completed_epoch': winner['completed_epoch'],
            'value': winner[metric]}


def trend(rows):
    scored = [row for row in rows if row['AP'] is not None]
    if len(scored) < 6:
        return {'assessment': 'insufficient_validation_history'}
    best = best_record(scored, 'AP')
    tail = scored[-min(10, len(scored)):]
    first, last = tail[0], tail[-1]
    loss_fell = (first['train_loss'] is not None and last['train_loss'] is not None
                 and last['train_loss'] < first['train_loss'])
    ap_fell = last['AP'] < first['AP'] - 0.005
    after_best = last['epoch_index'] - best['epoch_index']
    if loss_fell and ap_fell and after_best >= 5:
        assessment = 'possible_overfitting_or_late_training_instability'
    elif best['epoch_index'] >= scored[-1]['epoch_index'] - 2 and loss_fell:
        assessment = 'still_improving_possible_undertraining'
    else:
        assessment = 'no_clear_under_or_overfitting_signal'
    return {'assessment': assessment, 'window_epochs': len(tail),
            'tail_AP_change': last['AP'] - first['AP'],
            'tail_train_loss_change': (last['train_loss'] - first['train_loss']
                                       if first['train_loss'] is not None and
                                       last['train_loss'] is not None else None),
            'epochs_since_best_AP': after_best,
            'train_loss_fell_while_val_AP_fell': bool(loss_fell and ap_fell)}


def analyze(path):
    rows = parse_log(path)
    return {'path': str(path.resolve()), 'recorded_epochs': len(rows),
            'first_epoch': rows[0]['completed_epoch'],
            'last_epoch': rows[-1]['completed_epoch'],
            'best': {metric: best_record(rows, metric)
                     for metric in ('AP', 'AP50', 'AP75', 'AP_small')},
            'trend': trend(rows), 'epochs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='RT-DETR output/.../log.txt')
    parser.add_argument('--output', type=Path, help='optional full JSON report')
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f'log not found: {args.log}')
    report = analyze(args.log)
    summary = {key: value for key, value in report.items() if key != 'epochs'}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    columns = ('completed_epoch', 'lr', 'train_loss', 'loss_bbox',
               'loss_giou', 'loss_vfl', *METRICS)
    print(','.join(('epoch', *columns[1:])))
    for row in report['epochs']:
        print(','.join('' if row[key] is None else str(row[key]) for key in columns))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8')


if __name__ == '__main__':
    main()
