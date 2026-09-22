"""Analyze COCO UAV annotations without loading images or changing training.

From rtdetr_pytorch::

    python tools/analyze_uav_dataset.py
    python tools/analyze_uav_dataset.py --train /path/train.json --val /path/val.json --output dataset_stats.json

The simulated resize is the repository's direct Resize([S, S]): x and y are
scaled independently. ``both_below_N`` means width AND height are below N.
COCO AP_small, in contrast, uses original annotation area, not resized area.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median


PROJECT = Path(__file__).resolve().parents[1]
DATA_RELATIVE = Path('DUT-Anti-UAV/DUT-Anti-UAV/labels')
SIZES = (640, 800, 960)
LENGTH_THRESHOLDS = (4, 8, 16, 32)


def default_labels():
    # First location is exactly what the DUT YAML resolves from the training
    # working directory. The second supports this local checkout's layout;
    # using it is reported explicitly, never silently changes training paths.
    candidates = (PROJECT.parent / DATA_RELATIVE,
                  PROJECT.parent.parent / DATA_RELATIVE)
    for directory in candidates:
        if (directory / 'train.json').is_file() and (directory / 'val.json').is_file():
            return directory
    return candidates[0]


def describe(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p / 100
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        fraction = position - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    return {'min': ordered[0], 'max': ordered[-1], 'mean': fmean(ordered),
            'median': median(ordered),
            **{f'p{p}': percentile(p) for p in (10, 25, 50, 75, 90)}}


def load_split(path):
    with path.open('r', encoding='utf-8') as stream:
        data = json.load(stream)
    images = {}
    for image in data.get('images', []):
        width, height = image.get('width'), image.get('height')
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) \
                or width <= 0 or height <= 0:
            raise ValueError(f'{path}: invalid image size for image {image.get("id")}')
        image_id = image['id']
        if image_id in images:
            raise ValueError(f'{path}: duplicate image id {image_id}')
        images[image_id] = image
    categories = {item['id']: item.get('name', str(item['id']))
                  for item in data.get('categories', [])}
    if not images or not categories:
        raise ValueError(f'{path}: COCO images/categories are empty')
    boxes, counts, categories_seen = [], Counter(), Counter()
    rejected = Counter()
    for annotation in data.get('annotations', []):
        image = images.get(annotation.get('image_id'))
        if image is None:
            rejected['unknown_image'] += 1
            continue
        if annotation.get('iscrowd', 0):
            rejected['crowd'] += 1
            continue
        box = annotation.get('bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            rejected['invalid_bbox'] += 1
            continue
        x, y, width, height = box
        if not all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in (x, y, width, height)) or width <= 0 or height <= 0:
            rejected['invalid_bbox'] += 1
            continue
        category = annotation.get('category_id')
        if category not in categories:
            rejected['unknown_category'] += 1
            continue
        counts[annotation['image_id']] += 1
        categories_seen[category] += 1
        boxes.append((float(width), float(height), float(image['width']),
                      float(image['height'])))
    per_image = [counts[image_id] for image_id in images]
    return {
        'path': str(path.resolve()),
        'images': len(images), 'annotations_raw': len(data.get('annotations', [])),
        'annotations_valid': len(boxes), 'annotations_rejected': dict(rejected),
        'categories': {str(category): {'name': name, 'instances': categories_seen[category]}
                       for category, name in categories.items()},
        'objects_per_image': {'mean': fmean(per_image), 'median': median(per_image),
                              'max': max(per_image), 'zero_images': per_image.count(0)},
        'image_width': describe([float(image['width']) for image in images.values()]),
        'image_height': describe([float(image['height']) for image in images.values()]),
    }, boxes


def box_statistics(boxes):
    fields = {
        'width': [width for width, _, _, _ in boxes],
        'height': [height for _, height, _, _ in boxes],
        'area': [width * height for width, height, _, _ in boxes],
        'width_ratio': [width / image_w for width, _, image_w, _ in boxes],
        'height_ratio': [height / image_h for _, height, _, image_h in boxes],
        'area_ratio': [width * height / (image_w * image_h)
                       for width, height, image_w, image_h in boxes],
    }
    return {name: describe(values) for name, values in fields.items()}


def small_statistics(boxes, size=None):
    if size is None:
        dimensions = [(width, height) for width, height, _, _ in boxes]
    else:
        dimensions = [(width * size / image_w, height * size / image_h)
                      for width, height, image_w, image_h in boxes]
    total = len(dimensions)

    def measure(predicate):
        count = sum(bool(predicate(width, height)) for width, height in dimensions)
        return {'count': count, 'fraction': count / total if total else None}

    return {
        'n_boxes': total,
        'width_below_px': {str(n): measure(lambda w, h, n=n: w < n)
                           for n in LENGTH_THRESHOLDS},
        'height_below_px': {str(n): measure(lambda w, h, n=n: h < n)
                            for n in LENGTH_THRESHOLDS},
        'both_below_px': {str(n): measure(lambda w, h, n=n: w < n and h < n)
                          for n in LENGTH_THRESHOLDS},
        'area_below_px2': {str(n * n): measure(lambda w, h, n=n: w * h < n * n)
                            for n in (8, 16, 32)},
        'area_at_least_1024': measure(lambda w, h: w * h >= 32 * 32),
    }


def analyze(train, val):
    train_info, train_boxes = load_split(train)
    val_info, val_boxes = load_split(val)
    return {'train': train_info, 'val': val_info,
            'bbox_train': box_statistics(train_boxes),
            'small_original_train': small_statistics(train_boxes),
            'resized_train': {str(size): small_statistics(train_boxes, size)
                              for size in SIZES},
            'note': ('Fractions are over valid non-crowd train boxes. Resize is '
                     'direct square scaling, not letterbox. both_below_px requires '
                     'both width and height below threshold; COCO AP_small uses '
                     'original annotation area.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = default_labels()
    parser.add_argument('--train', type=Path, default=default_dir / 'train.json')
    parser.add_argument('--val', type=Path, default=default_dir / 'val.json')
    parser.add_argument('--output', type=Path, help='optional JSON report path')
    args = parser.parse_args()
    for path in (args.train, args.val):
        if not path.is_file():
            parser.error(f'annotation not found: {path}; pass --train and --val explicitly')
    report = analyze(args.train, args.val)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
