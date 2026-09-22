"""Read-only, GT-region diagnostics for trained BAFR/HCBR DUT models.

Example (run from rtdetr_pytorch on the server)::

    python tools/analyze_bafr_hcbr_examples.py --config configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bafr.yml --checkpoint output/rtdetr_r18vd_dut_anti_uav_bafr/best.pth --annotations ../DUT-Anti-UAV/DUT-Anti-UAV/labels/val.json --image-root ../DUT-Anti-UAV/DUT-Anti-UAV/images/val --image-ids 10 11 12 --sharp-ids 10 11 --blurred-ids 12 --amp

``--sharp-ids`` and ``--blurred-ids`` are manual *analysis-only* labels. The
dataset does not provide blur labels. This tool never supplies GT masks or blur
labels to the model and never changes training files or checkpoints.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotations', type=Path, required=True,
                        help='COCO-format JSON with images and bbox annotations')
    parser.add_argument('--image-root', type=Path, required=True,
                        help='directory containing COCO image file_name paths')
    parser.add_argument('--config', type=Path, required=True,
                        help='BAFR, HCBR or combined YAML config')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='trained checkpoint; EMA weights are used when present')
    parser.add_argument('--image-ids', type=int, nargs='+', required=True,
                        help='COCO image IDs to inspect')
    parser.add_argument('--sharp-ids', type=int, nargs='*', default=[],
                        help='manually selected clearly visible target image IDs')
    parser.add_argument('--blurred-ids', type=int, nargs='*', default=[],
                        help='manually selected blurred target image IDs')
    parser.add_argument('--device', default=None,
                        help='device such as cuda:0; defaults to CUDA when available')
    parser.add_argument('--amp', action='store_true',
                        help='use CUDA autocast for feature extraction')
    return parser.parse_args()


def validate_val_preprocess(config):
    """Fail closed if this script would not match the baseline val transform."""
    ops = config['val_dataloader']['dataset']['transforms']['ops']
    expected = ['Resize', 'ToImageTensor', 'ConvertDtype']
    names = [op.get('type') for op in ops]
    if names != expected or list(ops[0].get('size', [])) != [640, 640]:
        raise ValueError('Expected validation Resize([640,640]), ToImageTensor, '
                         'ConvertDtype; found %r. Update this analysis tool '
                         'to match the actual validation preprocessing.' % ops)
    return 640, 640


def load_coco(path):
    with path.open('r', encoding='utf-8') as handle:
        coco = json.load(handle)
    images = {int(image['id']): image for image in coco['images']}
    annotations = defaultdict(list)
    for annotation in coco.get('annotations', []):
        if 'bbox' in annotation and not annotation.get('iscrowd', 0):
            annotations[int(annotation['image_id'])].append(annotation['bbox'])
    return images, annotations


def bbox_mask(boxes, image_size, feature_size, device):
    """Map original-image xywh boxes to resized feature cells, no letterbox.

    Validation directly resizes W×H to 640×640. A feature grid of fw×fh
    therefore maps an original box edge x to x*fw/W (and y to y*fh/H).
    Start cells use floor, end cells use ceil. At least one cell is retained for
    each valid tiny bbox. The background is the complement of the box union.
    """
    import torch

    image_w, image_h = image_size
    feature_h, feature_w = feature_size
    mask = torch.zeros((feature_h, feature_w), dtype=torch.bool, device=device)
    accepted = 0
    for box in boxes:
        if len(box) != 4:
            continue
        x, y, width, height = map(float, box)
        if not all(map(math.isfinite, (x, y, width, height))) or width <= 0 or height <= 0:
            continue
        left = min(feature_w - 1, max(0, math.floor(x * feature_w / image_w)))
        top = min(feature_h - 1, max(0, math.floor(y * feature_h / image_h)))
        right = max(left + 1, min(feature_w, math.ceil((x + width) * feature_w / image_w)))
        bottom = max(top + 1, min(feature_h, math.ceil((y + height) * feature_h / image_h)))
        if x >= image_w or y >= image_h or x + width <= 0 or y + height <= 0:
            continue
        mask[top:bottom, left:right] = True
        accepted += 1
    return mask, accepted


def region_stats(field, boxes, image_size):
    """Return target/background means and target max from a 1×1×H×W map."""
    if field.ndim != 4 or field.shape[:2] != (1, 1):
        raise ValueError('Expected a single-image single-channel feature map')
    mask, box_count = bbox_mask(boxes, image_size, field.shape[-2:], field.device)
    values = field[0, 0]
    target = values[mask]
    background = values[~mask]
    return {
        'feature_size_hw': [int(values.shape[0]), int(values.shape[1])],
        'valid_boxes': box_count,
        'target_mean': float(target.mean()) if target.numel() else None,
        'target_max': float(target.max()) if target.numel() else None,
        'background_mean': float(background.mean()) if background.numel() else None,
    }


def load_backbone(config_path, checkpoint_path, device):
    import torch
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    validate_val_preprocess(cfg.yaml_cfg)
    # Analysis requires a checkpoint; avoid a redundant network download of
    # the original PResNet pretrained state during construction.
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    model = cfg.model
    state = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(state, dict) and isinstance(state.get('ema'), dict) and 'module' in state['ema']:
        weights, source = state['ema']['module'], 'ema.module (validation weights)'
    elif isinstance(state, dict) and 'model' in state:
        weights, source = state['model'], 'model'
    else:
        weights, source = state, 'direct state_dict'
    if not isinstance(weights, dict):
        raise ValueError('Checkpoint does not contain a model state_dict')
    if weights and all(key.startswith('module.') for key in weights):
        weights = {key[len('module.'):]: value for key, value in weights.items()}
    # Strict loading catches a config/checkpoint mismatch instead of reporting
    # meaningless region statistics from partially initialized weights.
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    return model.backbone, source


def record_inputs(backbone):
    captured = {}
    handles = []

    def remember(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return hook

    bafr = None
    if getattr(backbone, 'bafr_enabled', False):
        bafr = backbone.res_layers[1].blocks[-1]
        handles.append(bafr.register_forward_pre_hook(remember('bafr')))
    hcbr_modules = {}
    for name in ('hcbr_p3', 'hcbr_p4'):
        module = getattr(backbone, name, None)
        if module is not None:
            hcbr_modules[name] = module
            handles.append(module.register_forward_pre_hook(remember(name)))
    if bafr is None and not hcbr_modules:
        raise ValueError('The selected config has neither BAFR nor HCBR enabled')
    return captured, handles, bafr, hcbr_modules


def mean_present(records, path):
    values = []
    for record in records:
        node = record
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            values.append(node)
    return sum(values) / len(values) if values else None


def main():
    args = parse_args()
    # --help remains usable on a machine without project dependencies.
    import torch
    from PIL import Image
    from torchvision.transforms import v2 as T

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    for path in (args.annotations, args.config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.image_root.is_dir():
        raise NotADirectoryError(args.image_root)
    selected = list(dict.fromkeys(args.image_ids))
    sharp, blurred = set(args.sharp_ids), set(args.blurred_ids)
    if sharp & blurred:
        raise ValueError('sharp-ids and blurred-ids must not overlap')
    if not (sharp | blurred).issubset(selected):
        raise ValueError('Every sharp/blurred ID must also be listed in --image-ids')

    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    images, annotations = load_coco(args.annotations)
    missing = [image_id for image_id in selected if image_id not in images]
    if missing:
        raise ValueError('Image IDs missing from COCO annotations: %s' % missing)
    backbone, weight_source = load_backbone(args.config, args.checkpoint, device)
    print(json.dumps({'checkpoint_weights': weight_source,
                      'device': str(device), 'analysis_only': True}, ensure_ascii=False))
    captured, handles, bafr, hcbr_modules = record_inputs(backbone)
    preprocess = T.Compose([T.Resize((640, 640)), T.ToImageTensor(), T.ConvertDtype()])
    root = args.image_root.resolve()
    records = []
    try:
        for image_id in selected:
            image_meta = images[image_id]
            image_path = (root / image_meta['file_name']).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError('COCO file_name points outside --image-root: %s' % image_path)
            captured.clear()
            try:
                with Image.open(image_path) as image:
                    image = image.convert('RGB')
                    original_size = image.size
                    tensor = preprocess(image).unsqueeze(0).to(device)
            except (OSError, Image.DecompressionBombError) as exc:
                print(json.dumps({'image_id': image_id, 'error': str(exc)}, ensure_ascii=False))
                continue
            if (image_meta.get('width'), image_meta.get('height')) != original_size:
                raise ValueError('COCO image dimensions disagree with file for image ID %d' % image_id)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
                _ = backbone(tensor)
                boxes = annotations[image_id]
                record = {
                    'image_id': image_id,
                    'file_name': image_meta['file_name'],
                    'manual_blur_class': ('sharp' if image_id in sharp else
                                          'blurred' if image_id in blurred else None),
                    'original_size_wh': list(original_size),
                    'bbox_count': len(boxes),
                }
                if bafr is not None:
                    if 'bafr' not in captured:
                        raise RuntimeError('BAFR hook did not observe a feature map')
                    route = bafr.route_evidence(captured['bafr'])[2]
                    record['bafr_route_b'] = region_stats(route, boxes, original_size)
                for name, module in hcbr_modules.items():
                    if name not in captured:
                        raise RuntimeError('%s hook did not observe a feature map' % name)
                    gate = module.compute_components(captured[name])['gate']
                    record[name + '_gate'] = region_stats(gate, boxes, original_size)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
    finally:
        for handle in handles:
            handle.remove()

    sharp_records = [r for r in records if r['image_id'] in sharp]
    blurred_records = [r for r in records if r['image_id'] in blurred]
    sharp_mean = mean_present(sharp_records, ('bafr_route_b', 'target_mean'))
    blurred_mean = mean_present(blurred_records, ('bafr_route_b', 'target_mean'))
    summary = {
        'analyzed_images': len(records),
        'sharp_target_route_b_mean': sharp_mean,
        'blurred_target_route_b_mean': blurred_mean,
        'blurred_gt_sharp': (blurred_mean > sharp_mean
                             if sharp_mean is not None and blurred_mean is not None else None),
    }
    for name in hcbr_modules:
        summary[name + '_target_gate_mean'] = mean_present(records, (name + '_gate', 'target_mean'))
        summary[name + '_background_gate_mean'] = mean_present(records, (name + '_gate', 'background_mean'))
    print(json.dumps({'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
