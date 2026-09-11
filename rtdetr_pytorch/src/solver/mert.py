"""Micro-Shift Equivariant Refinement Training (MERT).

MERT is a training-only regularizer.  It owns no learnable parameters and does
not alter RT-DETR's model, decoder, criterion, matcher, or inference path.
"""

import contextlib
import copy
from collections import Counter

import numpy as np
import torch
import torch.distributed as tdist
import torch.nn as nn
import torch.nn.functional as F

from src.misc import dist


__all__ = [
    "get_mert_weight",
    "MicroShiftPairGenerator",
    "RefinementTrajectoryEquivarianceLoss",
    "MERTTrainingPlugin",
]


def _validate_mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("{} must be a mapping".format(name))
    return value


def _clone_target(target):
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in target.items()
    }


def get_mert_weight(current_epoch, base_weight, schedule_cfg=None):
    """Return the MERT loss weight for the trainer's real zero-based epoch."""
    schedule_cfg = _validate_mapping(schedule_cfg, "MERT.schedule")
    base_weight = float(base_weight)
    current_epoch = int(current_epoch)
    if base_weight < 0:
        raise ValueError("MERT trajectory weight must be non-negative")
    if not schedule_cfg.get("enabled", False):
        return base_weight

    start = int(schedule_cfg.get("start_epoch", 20))
    warmup_end = int(schedule_cfg.get("warmup_end_epoch", 40))
    hold_end = int(schedule_cfg.get("hold_end_epoch", 100))
    end = int(schedule_cfg.get("end_epoch", 160))
    if not (0 <= start < warmup_end <= hold_end < end):
        raise ValueError(
            "MERT schedule must satisfy 0 <= start_epoch < warmup_end_epoch "
            "<= hold_end_epoch < end_epoch"
        )

    if current_epoch < start:
        return 0.0
    if current_epoch < warmup_end:
        return base_weight * (current_epoch - start) / (warmup_end - start)
    if current_epoch < hold_end:
        return base_weight
    if current_epoch < end:
        return base_weight * (1.0 - (current_epoch - hold_end) / (end - hold_end))
    return 0.0


class MicroShiftPairGenerator(object):
    """Create a non-circular translated image/target pair after augmentation."""

    _INSTANCE_FIELDS = ("boxes", "labels", "area", "iscrowd", "masks", "keypoints")

    def __init__(self, config=None, debug=False):
        config = _validate_mapping(config, "MERT.micro_shift")
        self.max_pixels = int(config.get("max_pixels", 1))
        self.choices = tuple(int(value) for value in config.get(
            "choices", [-1, 0, 1]
        ))
        self.forbid_zero_zero = bool(config.get("forbid_zero_zero", True))
        self.per_image = bool(config.get("per_image", True))
        self.fill_mode = config.get("fill_mode", "baseline")
        self.debug = bool(debug)

        if self.max_pixels < 0:
            raise ValueError("MERT.micro_shift.max_pixels must be non-negative")
        if not self.choices:
            raise ValueError("MERT.micro_shift.choices must not be empty")
        if any(abs(value) > self.max_pixels for value in self.choices):
            raise ValueError("Every micro-shift choice must be within max_pixels")
        if self.fill_mode != "baseline":
            raise ValueError(
                "This pipeline supports only fill_mode='baseline' (zero after ToImageTensor)"
            )

        self.shift_pairs = [
            (dx, dy) for dx in self.choices for dy in self.choices
            if not self.forbid_zero_zero or (dx, dy) != (0, 0)
        ]
        if not self.shift_pairs:
            raise ValueError("No valid micro-shift pair can be sampled")
        self.last_debug_info = {}

    def _sample_shifts(self, batch_size):
        count = batch_size if self.per_image else 1
        selected = torch.randint(len(self.shift_pairs), (count,)).tolist()
        shifts = [self.shift_pairs[index] for index in selected]
        if not self.per_image:
            shifts = shifts * batch_size
        return torch.tensor(shifts, dtype=torch.int64)

    @staticmethod
    def shift_images(images, shifts, fill_value=0.0):
        """Translate with a blank canvas; content never wraps across borders."""
        if images.ndim != 4:
            raise ValueError("MERT images must be a 4-D NCHW tensor")
        if shifts.shape != (images.shape[0], 2):
            raise ValueError("shifts must have shape [batch_size, 2]")

        batch_size, _, height, width = images.shape
        shifted = torch.full_like(images, fill_value)
        for index in range(batch_size):
            dx, dy = (int(value) for value in shifts[index].tolist())
            if abs(dx) >= width or abs(dy) >= height:
                continue

            src_x1, src_x2 = max(-dx, 0), width - max(dx, 0)
            src_y1, src_y2 = max(-dy, 0), height - max(dy, 0)
            dst_x1, dst_x2 = max(dx, 0), width - max(-dx, 0)
            dst_y1, dst_y2 = max(dy, 0), height - max(-dy, 0)
            shifted[index, :, dst_y1:dst_y2, dst_x1:dst_x2] = \
                images[index, :, src_y1:src_y2, src_x1:src_x2]
        return shifted

    @classmethod
    def shift_target(cls, target, dx, dy, height, width):
        """Shift normalized cxcywh boxes via pixel xyxy, clip, and convert back."""
        original = _clone_target(target)
        boxes = original.get("boxes")
        if boxes is None:
            raise KeyError("Every MERT target must contain 'boxes'")
        num_gt = boxes.shape[0]
        origin_gt_id = torch.arange(num_gt, dtype=torch.int64, device=boxes.device)
        original["origin_gt_id"] = origin_gt_id
        original["mert_fully_visible"] = torch.ones(
            num_gt, dtype=torch.bool, device=boxes.device
        )

        shifted = _clone_target(target)
        if num_gt == 0:
            shifted["origin_gt_id"] = origin_gt_id
            shifted["mert_fully_visible"] = torch.zeros(
                0, dtype=torch.bool, device=boxes.device
            )
            return original, shifted, 0

        cx, cy, box_w, box_h = boxes.unbind(-1)
        x1 = (cx - 0.5 * box_w) * width + dx
        y1 = (cy - 0.5 * box_h) * height + dy
        x2 = (cx + 0.5 * box_w) * width + dx
        y2 = (cy + 0.5 * box_h) * height + dy
        translated_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        fully_visible = (
            (translated_xyxy[:, 0] >= 0)
            & (translated_xyxy[:, 1] >= 0)
            & (translated_xyxy[:, 2] <= width)
            & (translated_xyxy[:, 3] <= height)
        )
        clipped = translated_xyxy.clone()
        clipped[:, 0::2].clamp_(0, width)
        clipped[:, 1::2].clamp_(0, height)
        valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])

        clipped_w = clipped[:, 2] - clipped[:, 0]
        clipped_h = clipped[:, 3] - clipped[:, 1]
        shifted_boxes = torch.stack([
            (clipped[:, 0] + clipped[:, 2]) * 0.5 / width,
            (clipped[:, 1] + clipped[:, 3]) * 0.5 / height,
            clipped_w / width,
            clipped_h / height,
        ], dim=-1)

        for key in cls._INSTANCE_FIELDS:
            value = shifted.get(key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_gt:
                shifted[key] = value[valid]
        shifted["boxes"] = shifted_boxes[valid]
        if "area" in shifted:
            shifted["area"] = (clipped_w * clipped_h)[valid]
        shifted["origin_gt_id"] = origin_gt_id[valid]
        shifted["mert_fully_visible"] = fully_visible[valid]
        border_excluded = int((~fully_visible).sum().detach().cpu())
        return original, shifted, border_excluded

    def __call__(self, images, targets):
        batch_size, _, height, width = images.shape
        shifts = self._sample_shifts(batch_size)
        shifted_images = self.shift_images(images, shifts, fill_value=0.0)

        original_targets = []
        shifted_targets = []
        border_excluded = 0
        for target, shift in zip(targets, shifts):
            dx, dy = (int(value) for value in shift.tolist())
            original, shifted, excluded = self.shift_target(
                target, dx, dy, height, width
            )
            original_targets.append(original)
            shifted_targets.append(shifted)
            border_excluded += excluded

        if self.debug:
            distribution = Counter(
                "{},{}".format(int(dx), int(dy))
                for dx, dy in shifts.detach().cpu().tolist()
            )
            self.last_debug_info = {
                "shift_distribution": dict(distribution),
                "num_gt": sum(len(target["boxes"]) for target in original_targets),
                "num_total_gt": sum(
                    len(target["boxes"]) for target in original_targets
                ),
                "num_border_excluded": border_excluded,
                "num_gt_filtered_by_border": border_excluded,
            }
        return shifted_images, original_targets, shifted_targets, shifts


class RefinementTrajectoryEquivarianceLoss(nn.Module):
    """GT-identity-aligned consistency of decoded box refinement increments."""

    def __init__(
        self, matcher, trajectory=None, size_weight=None, matching=None, pair=None,
        debug=False
    ):
        super().__init__()
        trajectory = _validate_mapping(trajectory, "MERT.trajectory")
        size_weight = _validate_mapping(size_weight, "MERT.size_weight")
        matching = _validate_mapping(matching, "MERT.matching")
        pair = _validate_mapping(pair, "MERT.pair")

        if trajectory.get("loss_type", "smooth_l1") != "smooth_l1":
            raise ValueError("MERT currently supports only trajectory.loss_type='smooth_l1'")
        if not matching.get("use_final_layer_matching", True):
            raise ValueError("MERT v1 requires final-layer Hungarian matching")
        if not matching.get("exclude_dn_queries", True):
            raise ValueError("MERT v1 requires DN queries to be excluded")

        self.matcher = matcher
        self.enabled = bool(trajectory.get("enabled", True))
        self.weight = float(trajectory.get("weight", 0.10))
        self.beta = float(trajectory.get("beta", 0.01))
        self.layers = trajectory.get("layers", "last_2")
        # Old MERT YAMLs did not have component weights and averaged cxcywh.
        # A coefficient of 0.25 for each two-component sum reproduces that
        # exact four-dimensional mean.  MERT V2 YAMLs explicitly select 1/.25.
        legacy_component_weighting = (
            "xy_weight" not in trajectory
            and "wh_weight" not in trajectory
            and float(trajectory.get("weight", -1.0)) == 0.20
            and float(trajectory.get("beta", -1.0)) == 0.10
            and trajectory.get("layers") == "all"
        )
        self.xy_weight = float(trajectory.get(
            "xy_weight", 0.25 if legacy_component_weighting else 1.0
        ))
        self.wh_weight = float(trajectory.get(
            "wh_weight", 0.25 if legacy_component_weighting else 0.25
        ))
        self.size_weight_enabled = bool(size_weight.get("enabled", True))
        self.reference_area = float(size_weight.get("reference_area", 256.0))
        self.gamma = float(size_weight.get("gamma", 0.25))
        max_object_area = size_weight.get("max_object_area", None)
        self.max_object_area = (
            None if max_object_area is None else float(max_object_area)
        )
        self.min_weight = float(size_weight.get("min_weight", 1.0))
        self.max_weight = float(size_weight.get("max_weight", 2.5))
        self.require_fully_visible = bool(pair.get("require_fully_visible", True))
        self.debug = bool(debug)
        self.eps = 1.0e-6
        self.last_debug_info = {}

        if self.weight < 0 or self.beta <= 0:
            raise ValueError("MERT trajectory weight must be non-negative and beta positive")
        if self.xy_weight < 0 or self.wh_weight < 0:
            raise ValueError("MERT xy_weight and wh_weight must be non-negative")
        if self.reference_area <= 0 or self.gamma < 0:
            raise ValueError("Invalid MERT size-weight configuration")
        if self.max_object_area is not None and self.max_object_area <= 0:
            raise ValueError("MERT max_object_area must be positive or null")
        if self.min_weight <= 0 or self.max_weight < self.min_weight:
            raise ValueError("Invalid MERT size-weight clamp")

    @staticmethod
    def decoder_box_trajectory(outputs):
        """Return decoder layers only; RT-DETR appends encoder top-k as last aux."""
        if "pred_boxes" not in outputs:
            raise KeyError("MERT requires outputs['pred_boxes']")
        auxiliary = outputs.get("aux_outputs", [])
        if not auxiliary:
            raise RuntimeError("MERT requires aux_outputs with intermediate decoder boxes")
        decoder_auxiliary = auxiliary[:-1]
        trajectory = [item["pred_boxes"] for item in decoder_auxiliary]
        trajectory.append(outputs["pred_boxes"])
        if len(trajectory) < 2:
            raise RuntimeError("MERT requires at least two decoder layers")
        return trajectory

    @staticmethod
    def _matcher_view(outputs):
        return {
            "pred_logits": outputs["pred_logits"].detach().float(),
            "pred_boxes": outputs["pred_boxes"].detach().float(),
        }

    def _transition_indices(self, num_layers):
        """Select zero-based B_i -> B_(i+1) refinement transition indices."""
        num_transitions = num_layers - 1
        if self.layers == "all":
            return list(range(num_transitions))
        if isinstance(self.layers, str) and self.layers.startswith("last_"):
            try:
                count = int(self.layers.split("_", 1)[1])
            except (TypeError, ValueError):
                raise ValueError(
                    "MERT.trajectory.layers must be all, last_N, or a list"
                )
            if count <= 0:
                raise ValueError("MERT last_N must select at least one transition")
            count = min(count, num_transitions)
            return list(range(num_transitions - count, num_transitions))
        if not isinstance(self.layers, (list, tuple)):
            raise ValueError(
                "MERT.trajectory.layers must be all, last_N, or a list"
            )
        transitions = []
        for transition_index in self.layers:
            transition_index = int(transition_index)
            if transition_index < 0 or transition_index >= num_transitions:
                raise ValueError(
                    "MERT transition index {} is outside [0, {}]".format(
                        transition_index, num_transitions - 1
                    )
                )
            if transition_index not in transitions:
                transitions.append(transition_index)
        if not transitions:
            raise ValueError("MERT.trajectory.layers selects no transition")
        return transitions

    def _size_weight(self, box, height, width):
        if not self.size_weight_enabled:
            return box.new_tensor(1.0)
        area = (box[2] * width) * (box[3] * height)
        weight = (self.reference_area / (area + self.eps)).pow(self.gamma)
        weight = weight.clamp(self.min_weight, self.max_weight)
        if self.max_object_area is not None:
            weight = torch.where(
                area > self.max_object_area, torch.zeros_like(weight), weight
            )
        return weight

    @staticmethod
    def _query_map(target, indices, require_fully_visible=False):
        src_idx, target_idx = indices
        target_idx_device = target_idx.to(target["origin_gt_id"].device)
        origin_ids = target["origin_gt_id"][target_idx_device].detach().cpu().tolist()
        source_ids = src_idx.detach().cpu().tolist()
        if require_fully_visible:
            visible = target["mert_fully_visible"][target_idx_device].detach().cpu().tolist()
        else:
            visible = [True] * len(origin_ids)
        return {
            int(origin_id): int(source_id)
            for origin_id, source_id, keep in zip(origin_ids, source_ids, visible)
            if keep
        }

    def _distributed_scale(self, denominator):
        world_size = 1
        global_denominator = denominator.detach().clone()
        if tdist.is_available() and tdist.is_initialized():
            tdist.all_reduce(global_denominator)
            world_size = tdist.get_world_size()
        return world_size / (global_denominator + self.eps)

    def forward(
        self,
        outputs_original,
        outputs_shifted,
        original_targets,
        shifted_targets,
        shifts,
        input_size,
        current_weight=None,
    ):
        current_weight = self.weight if current_weight is None else float(current_weight)
        original_trajectory = self.decoder_box_trajectory(outputs_original)
        shifted_trajectory = self.decoder_box_trajectory(outputs_shifted)
        if len(original_trajectory) != len(shifted_trajectory):
            raise RuntimeError("Original and shifted decoder trajectories have different lengths")

        height, width = input_size
        transitions = self._transition_indices(len(original_trajectory))
        original_indices = self.matcher(
            self._matcher_view(outputs_original), original_targets
        )
        shifted_indices = self.matcher(
            self._matcher_view(outputs_shifted), shifted_targets
        )

        original_tracks = []
        shifted_tracks = []
        weights = []
        object_areas = []
        num_original_matched = 0
        num_shifted_matched = 0
        for batch_index, (indices_o, indices_s) in enumerate(
            zip(original_indices, shifted_indices)
        ):
            num_original_matched += len(indices_o[0])
            num_shifted_matched += len(indices_s[0])
            original_map = self._query_map(original_targets[batch_index], indices_o)
            shifted_map = self._query_map(
                shifted_targets[batch_index],
                indices_s,
                require_fully_visible=self.require_fully_visible,
            )
            common_ids = sorted(set(original_map) & set(shifted_map))
            dx, dy = shifts[batch_index].to(
                original_trajectory[0].device,
                original_trajectory[0].dtype,
            )
            zero = dx.new_zeros(())
            inverse_shift = torch.stack([
                dx / width, dy / height, zero, zero
            ])
            for origin_id in common_ids:
                query_o = original_map[origin_id]
                query_s = shifted_map[origin_id]
                track_o = torch.stack([
                    boxes[batch_index, query_o].float()
                    for boxes in original_trajectory
                ])
                track_s = torch.stack([
                    boxes[batch_index, query_s].float()
                    for boxes in shifted_trajectory
                ]) - inverse_shift.float()
                original_tracks.append(track_o)
                shifted_tracks.append(track_s)
                box = original_targets[batch_index]["boxes"][origin_id].float()
                weights.append(self._size_weight(box, height, width))
                object_areas.append((box[2] * width) * (box[3] * height))

        if original_tracks:
            tracks_o = torch.stack(original_tracks)
            tracks_s = torch.stack(shifted_tracks)
            delta_o = tracks_o[:, 1:] - tracks_o[:, :-1]
            delta_s = tracks_s[:, 1:] - tracks_s[:, :-1]
            delta_o = delta_o[:, transitions]
            delta_s = delta_s[:, transitions]
            weight_tensor = torch.stack(weights).to(delta_o)
            area_tensor = torch.stack(object_areas).to(delta_o)

            component_loss = F.smooth_l1_loss(
                delta_o, delta_s, beta=self.beta, reduction="none"
            )
            xy_element_loss = component_loss[..., :2].sum(dim=-1)
            wh_element_loss = component_loss[..., 2:].sum(dim=-1)
            xy_numerator = (xy_element_loss * weight_tensor[:, None]).sum()
            wh_numerator = (wh_element_loss * weight_tensor[:, None]).sum()
            denominator = weight_tensor.sum() * len(transitions)
        else:
            # Every rank must enter the denominator all-reduce, even when this
            # rank has no matched MERT pair, otherwise distributed training can
            # deadlock on batches with uneven target counts.
            zero = outputs_original["pred_boxes"].sum() * 0.0
            xy_numerator = zero
            wh_numerator = zero
            denominator = outputs_original["pred_boxes"].new_zeros(())
            weight_tensor = outputs_original["pred_boxes"].new_empty((0,))
            area_tensor = outputs_original["pred_boxes"].new_empty((0,))
            delta_o = outputs_original["pred_boxes"].new_empty(
                (0, len(transitions), 4)
            )
            delta_s = delta_o
            xy_element_loss = delta_o.new_empty((0, len(transitions)))
            wh_element_loss = delta_o.new_empty((0, len(transitions)))

        normalization_scale = self._distributed_scale(denominator)
        xy_loss = xy_numerator * normalization_scale
        wh_loss = wh_numerator * normalization_scale
        raw_loss = self.xy_weight * xy_loss + self.wh_weight * wh_loss

        if self.debug:
            valid_size = weight_tensor > 0
            num_mert_gt = int(valid_size.sum().detach().cpu())
            num_total_gt = sum(len(target["boxes"]) for target in original_targets)
            num_area_filtered = 0
            if self.size_weight_enabled and self.max_object_area is not None:
                for target in original_targets:
                    boxes = target["boxes"]
                    areas = (boxes[:, 2] * width) * (boxes[:, 3] * height)
                    num_area_filtered += int(
                        (areas > self.max_object_area).sum().detach().cpu()
                    )

            combined_element_loss = (
                self.xy_weight * xy_element_loss
                + self.wh_weight * wh_element_loss
            )
            layer_numerators = (
                combined_element_loss * weight_tensor[:, None]
            ).sum(dim=0)
            layer_losses = layer_numerators * normalization_scale
            layer_loss_dict = {
                "B{}->B{}".format(index, index + 1): float(value.detach())
                for index, value in zip(transitions, layer_losses)
            }
            transition_stats = {
                "trajectory_loss_transition_{}".format(position): float(value.detach())
                for position, value in enumerate(layer_losses, start=1)
            }

            if num_mert_gt:
                valid_areas = area_tensor[valid_size].detach()
                valid_weights = weight_tensor[valid_size].detach()
                trajectory_error = (
                    delta_o[valid_size].detach() - delta_s[valid_size].detach()
                ).abs()
                mean_area = float(valid_areas.mean())
                mean_weight = float(valid_weights.mean())
                max_weight = float(valid_weights.max())
                mean_xy_error = float(trajectory_error[..., :2].mean())
                mean_wh_error = float(trajectory_error[..., 2:].mean())
                magnitude_o = float(delta_o[valid_size].detach().norm(dim=-1).mean())
                magnitude_s = float(delta_s[valid_size].detach().norm(dim=-1).mean())
                trajectory_difference = float(
                    (delta_o[valid_size].detach() - delta_s[valid_size].detach())
                    .norm(dim=-1).mean()
                )
            else:
                mean_area = mean_weight = max_weight = 0.0
                mean_xy_error = mean_wh_error = 0.0
                magnitude_o = magnitude_s = trajectory_difference = 0.0

            self.last_debug_info = {
                "trajectory_beta": self.beta,
                "active_refinement_layers": [
                    "B{}->B{}".format(index, index + 1)
                    for index in transitions
                ],
                "xy_loss": float(xy_loss.detach()),
                "wh_loss": float(wh_loss.detach()),
                "trajectory_loss": float(raw_loss.detach()),
                "weighted_trajectory_loss": float(
                    (current_weight * raw_loss).detach()
                ),
                "trajectory_loss_per_layer": layer_loss_dict,
                "num_total_gt": num_total_gt,
                "num_mert_gt": num_mert_gt,
                "num_gt_filtered_by_area": num_area_filtered,
                "num_valid_pairs": num_mert_gt,
                "num_original_matched_queries": num_original_matched,
                "num_shifted_matched_queries": num_shifted_matched,
                "mean_object_area": mean_area,
                "mean_size_weight": mean_weight,
                "max_size_weight": max_weight,
                "mean_xy_trajectory_error": mean_xy_error,
                "mean_wh_trajectory_error": mean_wh_error,
                "mean_original_refinement_magnitude": magnitude_o,
                "mean_shifted_refinement_magnitude": magnitude_s,
                "mean_trajectory_difference": trajectory_difference,
            }
            self.last_debug_info.update(transition_stats)
        return current_weight * raw_loss


class MERTTrainingPlugin(object):
    """Orchestrate MERT without becoming part of the detector state_dict."""

    def __init__(self, config, matcher, current_epoch=0):
        config = _validate_mapping(config, "MERT")
        self.enabled = bool(config.get("enabled", False))
        self.forward_mode = config.get("forward_mode", "concat")
        self.debug = bool(config.get("debug", False))
        self.current_epoch = int(current_epoch)
        self._debug_printed = False
        self.last_debug_info = {}

        if self.forward_mode not in ("concat", "sequential"):
            raise ValueError("MERT.forward_mode must be 'concat' or 'sequential'")
        self.pair_generator = MicroShiftPairGenerator(
            config.get("micro_shift"), debug=self.debug
        ) \
            if self.enabled else None
        self.trajectory_loss = RefinementTrajectoryEquivarianceLoss(
            matcher,
            trajectory=config.get("trajectory"),
            size_weight=config.get("size_weight"),
            matching=config.get("matching"),
            pair=config.get("pair"),
            debug=self.debug,
        ) if self.enabled else None
        schedule = _validate_mapping(config.get("schedule"), "MERT.schedule")
        supervision = _validate_mapping(
            config.get("supervision"), "MERT.supervision"
        )
        if self.enabled:
            self.current_mert_weight = get_mert_weight(
                self.current_epoch, self.trajectory_loss.weight, schedule
            )
        else:
            self.current_mert_weight = 0.0
        # Missing supervision keeps legacy MERT behavior.  The formal V2 YAML
        # explicitly sets this to false for one-view detection supervision.
        self.shifted_detection_loss = bool(
            supervision.get("shifted_detection_loss", True)
        )

    @staticmethod
    def _model_module(model):
        return dist.de_parallel(model)

    def prepare(self, model, samples, targets):
        """Apply RT-DETR's train-time scale first, then create the exact shift."""
        module = self._model_module(model)
        multi_scale = getattr(module, "multi_scale", None)
        if multi_scale:
            size = int(np.random.choice(multi_scale))
            samples = F.interpolate(samples, size=[size, size])

        shifted_images, original_targets, shifted_targets, shifts = \
            self.pair_generator(samples, targets)
        return {
            "original_images": samples,
            "shifted_images": shifted_images,
            "original_targets": original_targets,
            "shifted_targets": shifted_targets,
            "shifts": shifts,
            "input_size": samples.shape[-2:],
        }

    @contextlib.contextmanager
    def disable_internal_multiscale(self, model):
        """Avoid a second resize after MERT has sampled the exact input pixels."""
        module = self._model_module(model)
        original_multi_scale = getattr(module, "multi_scale", None)
        if hasattr(module, "multi_scale"):
            module.multi_scale = None
        try:
            yield
        finally:
            if hasattr(module, "multi_scale"):
                module.multi_scale = original_multi_scale

    @staticmethod
    def split_concatenated_outputs(outputs, batch_size):
        def select(start, end):
            selected = {
                "pred_logits": outputs["pred_logits"][start:end],
                "pred_boxes": outputs["pred_boxes"][start:end],
            }
            if "aux_outputs" in outputs:
                selected["aux_outputs"] = [
                    {
                        "pred_logits": item["pred_logits"][start:end],
                        "pred_boxes": item["pred_boxes"][start:end],
                    }
                    for item in outputs["aux_outputs"]
                ]
            if "dn_aux_outputs" in outputs:
                selected["dn_aux_outputs"] = [
                    {
                        "pred_logits": item["pred_logits"][start:end],
                        "pred_boxes": item["pred_boxes"][start:end],
                    }
                    for item in outputs["dn_aux_outputs"]
                ]
            if "dn_meta" in outputs:
                selected["dn_meta"] = dict(outputs["dn_meta"])
                selected["dn_meta"]["dn_positive_idx"] = \
                    outputs["dn_meta"]["dn_positive_idx"][start:end]
            return selected
        return select(0, batch_size), select(batch_size, 2 * batch_size)

    def calculate_loss(self, outputs_original, outputs_shifted, pair):
        if not self.trajectory_loss.enabled:
            loss = outputs_original["pred_boxes"].sum() * 0.0
        else:
            loss = self.trajectory_loss(
                outputs_original,
                outputs_shifted,
                pair["original_targets"],
                pair["shifted_targets"],
                pair["shifts"],
                pair["input_size"],
                current_weight=self.current_mert_weight,
            )
        self.last_debug_info = dict(self.pair_generator.last_debug_info)
        self.last_debug_info.update(self.trajectory_loss.last_debug_info)
        self.last_debug_info.update({
            "current_epoch": self.current_epoch,
            "current_mert_weight": self.current_mert_weight,
            "trajectory_beta": self.trajectory_loss.beta,
            "shifted_detection_loss_enabled": self.shifted_detection_loss,
        })
        if self.debug and not self._debug_printed and dist.is_main_process():
            print("[MERT debug] {}".format(self.last_debug_info))
            self._debug_printed = True
        return loss


def average_loss_dicts(first, second):
    if set(first) != set(second):
        raise RuntimeError("Sequential MERT views returned different detection loss keys")
    return {key: 0.5 * (first[key] + second[key]) for key in first}
