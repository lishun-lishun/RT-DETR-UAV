"""Compare CURE responses inside GT UAV regions and background regions."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig  # noqa: E402


GROUPS = ("0-8", "8-16", "16-32", ">32", "background")
METRICS = ("discrepancy", "gate", "residual_magnitude")


def load_model_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    return checkpoint["model"]


def target_boxes_xyxy(target, image_height, image_width):
    source = target["boxes"]
    format_name = str(getattr(source, "format", "xyxy")).lower()
    boxes = source.detach().float().cpu()
    if "cxcywh" in format_name:
        center_x, center_y, width, height = boxes.unbind(-1)
        boxes = torch.stack([
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ], dim=-1)
    if boxes.numel() and float(boxes.abs().max()) <= 1.5:
        scale = boxes.new_tensor([
            image_width, image_height, image_width, image_height
        ])
        boxes = boxes * scale
    return boxes


def size_group(box):
    width = max(float(box[2] - box[0]), 0.0)
    height = max(float(box[3] - box[1]), 0.0)
    size = max(width, height)
    if size <= 8:
        return "0-8"
    if size <= 16:
        return "8-16"
    if size <= 32:
        return "16-32"
    return ">32"


def feature_region(box, image_height, image_width, feature_height, feature_width):
    x1 = max(0, min(feature_width - 1, math.floor(float(box[0]) / image_width * feature_width)))
    y1 = max(0, min(feature_height - 1, math.floor(float(box[1]) / image_height * feature_height)))
    x2 = max(x1 + 1, min(feature_width, math.ceil(float(box[2]) / image_width * feature_width)))
    y2 = max(y1 + 1, min(feature_height, math.ceil(float(box[3]) / image_height * feature_height)))
    return slice(y1, y2), slice(x1, x2)


def update_stats(stat, maps, region):
    for metric in METRICS:
        values = maps[metric][region].float()
        stat[metric + "_sum"] += float(values.sum())
        stat[metric + "_count"] += values.numel()


def finalize(stats):
    result = {}
    for group in GROUPS:
        result[group] = {}
        for metric in METRICS:
            count = stats[group][metric + "_count"]
            result[group][metric] = (
                stats[group][metric + "_sum"] / count if count else None
            )
        result[group]["feature_cells"] = stats[group]["discrepancy_count"]
    return result


def main(args):
    config = YAMLConfig(args.config, resume=args.checkpoint)
    cure_config = config.yaml_cfg.get("PResNet", {}).get("cure_config")
    if not cure_config or not cure_config.get("enabled", False):
        raise ValueError("analyze_cure.py requires a CURE-enabled config")
    if 8 not in cure_config.get("target_strides", [8]):
        raise ValueError("analyze_cure.py currently analyzes stride-8 CURE")
    cure_config["debug"] = True
    # A complete detector checkpoint is loaded below; avoid an unnecessary
    # backbone-weight download while constructing the model.
    config.yaml_cfg["PResNet"]["pretrained"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.model
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    model.to(device).eval()
    cure = model.backbone.cure_s3
    data_loader = config.val_dataloader

    stats = {group: defaultdict(float) for group in GROUPS}
    processed = 0
    with torch.no_grad():
        for images, targets in data_loader:
            if args.max_images > 0:
                remaining = args.max_images - processed
                if remaining <= 0:
                    break
                images = images[:remaining]
                targets = targets[:remaining]
            images = images.to(device)
            model(images)
            debug_maps = cure.last_debug_maps
            image_height, image_width = images.shape[-2:]
            feature_height, feature_width = debug_maps["gate"].shape[-2:]

            for batch_index, target in enumerate(targets):
                maps = {
                    metric: debug_maps[metric][batch_index, 0]
                    for metric in METRICS
                }
                foreground = torch.zeros(
                    feature_height, feature_width, dtype=torch.bool
                )
                boxes = target_boxes_xyxy(target, image_height, image_width)
                for box in boxes:
                    region = feature_region(
                        box,
                        image_height,
                        image_width,
                        feature_height,
                        feature_width,
                    )
                    foreground[region] = True
                    update_stats(stats[size_group(box)], maps, region)
                update_stats(stats["background"], maps, ~foreground)

            processed += len(targets)
            print("Analyzed {} images".format(processed))

    groups = finalize(stats)
    warnings = []
    tiny_gate = groups["0-8"]["gate"]
    background_gate = groups["background"]["gate"]
    tiny_discrepancy = groups["0-8"]["discrepancy"]
    background_discrepancy = groups["background"]["discrepancy"]
    if tiny_gate is not None and background_gate is not None and tiny_gate <= background_gate:
        warnings.append("Tiny-target mean gate is not above background mean gate.")
    if (
        tiny_discrepancy is not None
        and background_discrepancy is not None
        and tiny_discrepancy <= background_discrepancy
    ):
        warnings.append(
            "Tiny-target mean discrepancy is not above background; edges/clutter may dominate."
        )

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_images": processed,
        "feature_stride": 8,
        "size_definition": "max(box_width, box_height) in resized input pixels",
        "region_reduction": "mean over stride-8 feature cells",
        "groups": groups,
        "warnings": warnings,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved CURE analysis to {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", default="cure_analysis.json")
    main(parser.parse_args())
