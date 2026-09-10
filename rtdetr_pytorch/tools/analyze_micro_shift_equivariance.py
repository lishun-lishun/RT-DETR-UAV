"""Analyze detection sensitivity to fixed 0--1 pixel validation shifts.

This script is diagnostic only. It does not train or modify the model.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.solver.mert import MicroShiftPairGenerator
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou


DEFAULT_SHIFTS = ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1))


def _target_boxes_as_normalized_cxcywh(target, height, width):
    boxes = target["boxes"].as_subclass(torch.Tensor).float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    box_format = getattr(getattr(target["boxes"], "format", None), "value", "XYXY")
    if str(box_format).lower() == "cxcywh":
        result = boxes.clone()
        if float(result.detach().max()) > 1.5:
            result = result / result.new_tensor([width, height, width, height])
        return result

    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([
        (x1 + x2) * 0.5 / width,
        (y1 + y2) * 0.5 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ], dim=-1)


def _greedy_detection_match(outputs, batch_index, gt_boxes, gt_labels, score_thr, iou_thr):
    logits = outputs["pred_logits"][batch_index]
    pred_boxes = outputs["pred_boxes"][batch_index]
    scores, labels = logits.sigmoid().max(dim=-1)
    keep = scores >= score_thr
    query_ids = torch.nonzero(keep).flatten()
    if query_ids.numel() == 0 or gt_boxes.numel() == 0:
        return {}

    candidate_boxes = pred_boxes[query_ids]
    ious, _ = box_iou(
        box_cxcywh_to_xyxy(candidate_boxes), box_cxcywh_to_xyxy(gt_boxes)
    )
    candidates = []
    for prediction_index in range(len(query_ids)):
        for gt_index in range(len(gt_boxes)):
            if labels[query_ids[prediction_index]] != gt_labels[gt_index]:
                continue
            iou = float(ious[prediction_index, gt_index])
            if iou >= iou_thr:
                candidates.append((
                    float(scores[query_ids[prediction_index]]),
                    prediction_index,
                    gt_index,
                    iou,
                ))
    candidates.sort(reverse=True)
    used_predictions, used_targets, matches = set(), set(), {}
    for score, prediction_index, gt_index, iou in candidates:
        if prediction_index in used_predictions or gt_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(gt_index)
        matches[gt_index] = {
            "score": score,
            "iou": iou,
            "box": candidate_boxes[prediction_index].detach(),
        }
    return matches


def _size_group(box, height, width):
    maximum_side = max(float(box[2]) * width, float(box[3]) * height)
    if maximum_side < 8:
        return "0-8px"
    if maximum_side < 16:
        return "8-16px"
    if maximum_side < 32:
        return "16-32px"
    return ">=32px"


def _fully_visible(box, dx, dy, height, width):
    xyxy = box_cxcywh_to_xyxy(box[None])[0]
    return bool(
        xyxy[0] * width + dx >= 0
        and xyxy[1] * height + dy >= 0
        and xyxy[2] * width + dx <= width
        and xyxy[3] * height + dy <= height
    )


def _empty_stat():
    return {
        "eligible_gt": 0,
        "baseline_detected": 0,
        "detection_flips": 0,
        "both_detected": 0,
        "confidence_change_sum": 0.0,
        "iou_change_sum": 0.0,
        "center_drift_sum": 0.0,
    }


def _finalize(stat):
    both = max(stat["both_detected"], 1)
    baseline = max(stat["baseline_detected"], 1)
    return {
        "eligible_gt": stat["eligible_gt"],
        "baseline_detected": stat["baseline_detected"],
        "detection_flips": stat["detection_flips"],
        "detection_flip_rate": stat["detection_flips"] / baseline,
        "both_detected": stat["both_detected"],
        "mean_abs_confidence_change": stat["confidence_change_sum"] / both,
        "mean_abs_iou_change": stat["iou_change_sum"] / both,
        "mean_bbox_center_drift_pixels": stat["center_drift_sum"] / both,
    }


def main(args):
    config = YAMLConfig(args.config)
    config.yaml_cfg["PResNet"]["pretrained"] = False
    model = config.model.to(config.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"]
    else:
        state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    stats = {
        "{},{}".format(dx, dy): defaultdict(_empty_stat)
        for dx, dy in DEFAULT_SHIFTS
    }
    processed = 0
    with torch.no_grad():
        for images, targets in config.val_dataloader:
            images = images.to(config.device)
            targets = [
                {key: value.to(config.device) for key, value in target.items()}
                for target in targets
            ]
            height, width = images.shape[-2:]
            baseline_outputs = model(images)

            normalized_targets = [
                _target_boxes_as_normalized_cxcywh(target, height, width)
                for target in targets
            ]
            baseline_matches = [
                _greedy_detection_match(
                    baseline_outputs, index, boxes, targets[index]["labels"],
                    args.score_threshold, args.iou_threshold,
                )
                for index, boxes in enumerate(normalized_targets)
            ]

            for dx, dy in DEFAULT_SHIFTS:
                shift_name = "{},{}".format(dx, dy)
                if (dx, dy) == (0, 0):
                    shifted_outputs = baseline_outputs
                else:
                    shifts = torch.tensor(
                        [[dx, dy]] * len(images), dtype=torch.int64
                    )
                    shifted_images = MicroShiftPairGenerator.shift_images(images, shifts)
                    shifted_outputs = model(shifted_images)
                    shifted_outputs = dict(shifted_outputs)
                    shifted_outputs["pred_boxes"] = shifted_outputs["pred_boxes"].clone()
                    shifted_outputs["pred_boxes"][..., 0] -= dx / width
                    shifted_outputs["pred_boxes"][..., 1] -= dy / height

                shifted_matches = [
                    _greedy_detection_match(
                        shifted_outputs, index, boxes, targets[index]["labels"],
                        args.score_threshold, args.iou_threshold,
                    )
                    for index, boxes in enumerate(normalized_targets)
                ]
                for batch_index, boxes in enumerate(normalized_targets):
                    for gt_index, box in enumerate(boxes):
                        if not _fully_visible(box, dx, dy, height, width):
                            continue
                        group = _size_group(box, height, width)
                        stat = stats[shift_name][group]
                        stat["eligible_gt"] += 1
                        baseline_match = baseline_matches[batch_index].get(gt_index)
                        shifted_match = shifted_matches[batch_index].get(gt_index)
                        if baseline_match is None:
                            continue
                        stat["baseline_detected"] += 1
                        if shifted_match is None:
                            stat["detection_flips"] += 1
                            continue
                        stat["both_detected"] += 1
                        stat["confidence_change_sum"] += abs(
                            shifted_match["score"] - baseline_match["score"]
                        )
                        stat["iou_change_sum"] += abs(
                            shifted_match["iou"] - baseline_match["iou"]
                        )
                        center_delta = shifted_match["box"][:2] - baseline_match["box"][:2]
                        center_delta = center_delta * center_delta.new_tensor([width, height])
                        stat["center_drift_sum"] += float(center_delta.norm())

            processed += len(images)
            print("Analyzed {} images".format(processed))
            if args.max_images > 0 and processed >= args.max_images:
                break

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "size_definition": "maximum GT box side at model input resolution",
        "num_images": processed,
        "shifts": {
            shift: {group: _finalize(stat) for group, stat in groups.items()}
            for shift, groups in stats.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved analysis to {}".format(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", default="micro_shift_equivariance.json")
    main(parser.parse_args())
