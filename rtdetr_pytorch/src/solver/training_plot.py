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
                'loss': _finite(entry.get('train_loss')),
                'map_50_95': _finite(scores[0]) if len(scores) > 0 else math.nan,
                'map50': _finite(scores[1]) if len(scores) > 1 else math.nan,
                'map75': _finite(scores[2]) if len(scores) > 2 else math.nan,
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
    """Plot loss and COCO bbox AP metrics in one atomically replaced PNG."""
    require_plot_backend()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _read_rows(log_path)
    if not rows:
        return False

    epochs = [row['epoch'] for row in rows]
    fig, loss_axis = plt.subplots(figsize=(11, 6.5), dpi=140)
    metric_axis = loss_axis.twinx()

    loss_line, = loss_axis.plot(
        epochs, [row['loss'] for row in rows], color='#d62728', linewidth=1.8,
        label='Train loss')
    map_line, = metric_axis.plot(
        epochs, [100 * row['map_50_95'] for row in rows], color='#1f77b4',
        linewidth=2.2, label='mAP50-95')
    map50_line, = metric_axis.plot(
        epochs, [100 * row['map50'] for row in rows], color='#2ca02c',
        linewidth=1.7, label='mAP50')
    map75_line, = metric_axis.plot(
        epochs, [100 * row['map75'] for row in rows], color='#9467bd',
        linewidth=1.7, label='mAP75')

    valid_ap = [(row['map_50_95'], row['epoch']) for row in rows
                if math.isfinite(row['map_50_95'])]
    if valid_ap:
        best_ap, best_epoch = max(valid_ap)
        metric_axis.scatter([best_epoch], [100 * best_ap], color='#1f77b4',
                            edgecolors='white', linewidths=0.8, s=55, zorder=5)
        metric_axis.annotate(
            f'best {best_ap * 100:.2f} @ {best_epoch}',
            xy=(best_epoch, best_ap * 100), xytext=(7, 8),
            textcoords='offset points', color='#1f77b4', fontsize=9)

    loss_axis.set_xlabel('Completed epoch')
    loss_axis.set_ylabel('Train loss', color='#d62728')
    metric_axis.set_ylabel('Validation AP (%)')
    metric_axis.set_ylim(0, 100)
    loss_axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    loss_axis.set_title('RT-DETR training convergence')
    loss_axis.legend(handles=[loss_line, map_line, map50_line, map75_line],
                     loc='best', frameon=True)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + '.tmp' + output_path.suffix)
    fig.savefig(temporary, format=output_path.suffix.lstrip('.') or 'png')
    plt.close(fig)
    temporary.replace(output_path)
    return True

