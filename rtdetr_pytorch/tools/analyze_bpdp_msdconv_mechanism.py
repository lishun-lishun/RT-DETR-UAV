"""Offline real-image mechanism statistics for manually selected DUT samples.

The manifest is a JSON list. Each item has ``image``, ``category`` (for
example ``clear``, ``blur`` or ``background``), and optional COCO-format
``boxes`` in original-image pixels. Empty boxes analyze the whole feature map.

Example item:
  {"image": "images/val/0001.jpg", "category": "clear",
   "boxes": [[120, 80, 12, 9]]}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from src.core import YAMLConfig


def _load_model(config_path, checkpoint_path, device):
    overrides = {'PResNet': {'pretrained': False}}
    config_name = Path(config_path).stem.lower()
    if 'bpdp' in config_name:
        overrides['BDPD'] = {'debug': True}
    if 'msdconv' in config_name:
        overrides['MSDConv'] = {'debug': True}
    cfg = YAMLConfig(str(config_path), **overrides)
    model = cfg.model.to(device).eval()
    model.multi_scale = None
    state = torch.load(checkpoint_path, map_location='cpu')
    weights = state['ema']['module'] if 'ema' in state else state['model']
    model.load_state_dict(weights, strict=True)
    return model


def _regions(boxes, image_size, map_size):
    image_w, image_h = image_size
    map_h, map_w = map_size
    if not boxes:
        return [(0, map_h, 0, map_w)]
    regions = []
    for x, y, width, height in boxes:
        x0 = max(0, min(map_w - 1, int(x / image_w * map_w)))
        y0 = max(0, min(map_h - 1, int(y / image_h * map_h)))
        x1 = max(x0 + 1, min(map_w, int((x + width) / image_w * map_w + 0.999)))
        y1 = max(y0 + 1, min(map_h, int((y + height) / image_h * map_h + 0.999)))
        regions.append((y0, y1, x0, x1))
    return regions


def _region_means(tensors, boxes, image_size):
    first = next(iter(tensors.values()))
    regions = _regions(boxes, image_size, first.shape[-2:])
    output = {}
    for name, tensor in tensors.items():
        if tensor.ndim == 4:
            tensor = tensor.mean(dim=1)
        values = [tensor[0, y0:y1, x0:x1].mean().item()
                  for y0, y1, x0, x1 in regions]
        output[name] = sum(values) / len(values)
    return output


def analyze(config, checkpoint, manifest, device):
    model = _load_model(config, checkpoint, device)
    backbone = model.backbone
    manifest = Path(manifest).resolve()
    samples = json.loads(manifest.read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    for sample in samples:
        path = Path(sample['image'])
        if not path.is_absolute():
            path = manifest.parent / path
        with Image.open(path) as image:
            image = image.convert('RGB')
            original_size = image.size
            tensor = pil_to_tensor(image).float().div_(255).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(640, 640), mode='bilinear',
                               align_corners=False).to(device)
        with torch.no_grad():
            backbone(tensor)
        tensors = {}
        branch = backbone.res_layers[1].blocks[0].branch2a
        if hasattr(branch, 'last_debug_tensors') and branch.last_debug_tensors:
            tensors.update(branch.last_debug_tensors)
        if backbone.msdconv_p3 is not None and backbone.msdconv_p3.last_debug_tensors:
            tensors.update(backbone.msdconv_p3.last_debug_tensors)
        if not tensors:
            raise RuntimeError('Selected config did not produce BDPD/MSDConv debug tensors')
        values = _region_means(tensors, sample.get('boxes', []), original_size)
        grouped[sample['category']].append(values)

    summary = {}
    for category, records in grouped.items():
        summary[category] = {
            name: sum(record[name] for record in records) / len(records)
            for name in records[0]
        }
        summary[category]['samples'] = len(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, type=Path)
    parser.add_argument('-r', '--checkpoint', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.config, args.checkpoint, args.manifest, args.device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
