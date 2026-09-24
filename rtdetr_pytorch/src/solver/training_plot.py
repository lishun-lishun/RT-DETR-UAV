"""Generate one convergence figure from the RT-DETR JSON-lines log."""

import json
import math
from pathlib import Path


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _score(scores, index):
    value = _finite(scores[index]) if len(scores) > index else math.nan
    return value if math.isfinite(value) and value >= 0 else math.nan


def _read_rows(log_path):
    latest = {}
    with Path(log_path).open('r', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted final write must not destroy an existing plot.
                continue
            epoch = entry.get('epoch')
            scores = entry.get('test_coco_eval_bbox', [])
            if not isinstance(epoch, int) or not isinstance(scores, list):
                continue
            latest[epoch] = {
                'epoch': epoch + 1,
                'lr': _finite(entry.get('train_lr')),
                'loss': _finite(entry.get('train_loss')),
                'map_50_95': _score(scores, 0),
                'map50': _score(scores, 1),
                'map75': _score(scores, 2),
                'ap_small': _score(scores, 3),
            }
    return [latest[index] for index in sorted(latest)]


def require_plot_backend():
    """Fail before a long run when the requested plotting dependency is absent."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            'plot_training_curves=true requires matplotlib. Install it with: '
            'python -m pip install matplotlib') from error


def plot_training_curves(log_path, output_path):
    """Plot six separate convergence panels in one atomically replaced PNG."""
    require_plot_backend()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _read_rows(log_path)
    if not rows:
        return False

    epochs = [row['epoch'] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=140, sharex=True)
    panels = (
        ('loss', 'Train loss', '#d62728', False),
        ('lr', 'Backbone learning rate (logged group 0)', '#ff7f0e', False),
        ('map_50_95', 'Validation mAP50-95', '#1f77b4', True),
        ('map50', 'Validation mAP50', '#2ca02c', True),
        ('map75', 'Validation mAP75', '#9467bd', True),
        ('ap_small', 'Validation AP small', '#17becf', True),
    )
    for axis, (key, title, color, percentage) in zip(axes.flat, panels):
        multiplier = 100.0 if percentage else 1.0
        values = [multiplier * row[key] for row in rows]
        axis.plot(epochs, values, color=color, linewidth=1.9)
        axis.set_title(title)
        axis.set_xlabel('Completed epoch')
        axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
        if percentage:
            axis.set_ylabel('AP (%)')
            axis.set_ylim(0, 100)
            valid = [(multiplier * row[key], row['epoch']) for row in rows
                     if math.isfinite(row[key])]
            if valid:
                best_value, best_epoch = max(valid)
                axis.scatter([best_epoch], [best_value], color=color,
                             edgecolors='white', linewidths=0.8, s=45, zorder=5)
                axis.annotate(
                    f'best {best_value:.2f} @ {best_epoch}',
                    xy=(best_epoch, best_value), xytext=(6, 7),
                    textcoords='offset points', color=color, fontsize=8)
        elif key == 'lr':
            axis.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        else:
            axis.set_ylabel('Loss')
    fig.suptitle('RT-DETR training convergence', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + '.tmp' + output_path.suffix)
    fig.savefig(temporary, format=output_path.suffix.lstrip('.') or 'png')
    plt.close(fig)
    temporary.replace(output_path)
    return True
