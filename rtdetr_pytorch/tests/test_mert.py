"""Tests for the training-only MERT plug-in.

Run from ``rtdetr_pytorch``:

    python -m unittest tests.test_mert -v
"""

import sys
import types
import copy
import unittest
from unittest import mock
from pathlib import Path

import torch
import torch.nn as nn


# MERT itself does not use these optional packages, but importing the project's
# src package loads all registered dataset/backbone modules.
try:
    import pycocotools  # noqa: F401
except ImportError:
    pycocotools = types.ModuleType("pycocotools")
    pycocotools.mask = types.ModuleType("pycocotools.mask")
    pycocotools.coco = types.ModuleType("pycocotools.coco")
    pycocotools.coco.COCO = object
    pycocotools.cocoeval = types.ModuleType("pycocotools.cocoeval")
    pycocotools.cocoeval.COCOeval = object
    sys.modules.update({
        "pycocotools": pycocotools,
        "pycocotools.mask": pycocotools.mask,
        "pycocotools.coco": pycocotools.coco,
        "pycocotools.cocoeval": pycocotools.cocoeval,
    })

try:
    import transformers  # noqa: F401
except ImportError:
    transformers = types.ModuleType("transformers")
    transformers.RegNetModel = object
    sys.modules["transformers"] = transformers


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.core import YAMLConfig  # noqa: E402
from src.core.yaml_utils import load_config  # noqa: E402
from src.solver.mert import (  # noqa: E402
    MERTTrainingPlugin,
    MicroShiftPairGenerator,
    RefinementTrajectoryEquivarianceLoss,
    get_mert_weight,
)
from src.solver.det_engine import train_one_epoch  # noqa: E402


MERT_CONFIG = {
    "enabled": True,
    "forward_mode": "concat",
    "micro_shift": {
        "max_pixels": 2,
        "choices": [-2, -1, 0, 1, 2],
        "forbid_zero_zero": True,
        "per_image": True,
        "fill_mode": "baseline",
    },
    "trajectory": {
        "enabled": True,
        "loss_type": "smooth_l1",
        "weight": 0.2,
        "beta": 0.1,
        "layers": "all",
    },
    "size_weight": {
        "enabled": True,
        "reference_area": 256.0,
        "gamma": 0.5,
        "min_weight": 1.0,
        "max_weight": 4.0,
    },
    "matching": {
        "use_final_layer_matching": True,
        "exclude_dn_queries": True,
    },
    "pair": {"require_fully_visible": True},
    "debug": False,
}


class DesiredQueryMatcher(nn.Module):
    """A deterministic matcher that permits different query IDs per view."""

    @torch.no_grad()
    def forward(self, outputs, targets):
        result = []
        for target in targets:
            num_targets = len(target["boxes"])
            if num_targets == 0:
                empty = torch.zeros(0, dtype=torch.int64)
                result.append((empty, empty))
                continue
            first_query = int(target.get("desired_query", 0))
            result.append((
                torch.arange(first_query, first_query + num_targets, dtype=torch.int64),
                torch.arange(num_targets, dtype=torch.int64),
            ))
        return result


def make_target(box=(0.5, 0.5, 0.1, 0.1), desired_query=0):
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([box[2] * box[3] * 640 * 640]),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
        "image_id": torch.tensor([1]),
        "orig_size": torch.tensor([640, 640]),
        "size": torch.tensor([640, 640]),
        "desired_query": desired_query,
    }


def make_outputs(track, query_index, num_queries=3, perturb_layer=None):
    layers = []
    for layer_index, box in enumerate(track):
        tensor = torch.full((1, num_queries, 4), 0.25, dtype=torch.float32)
        tensor[0, query_index] = torch.tensor(box)
        if perturb_layer is not None and layer_index == perturb_layer:
            tensor[0, query_index, 0] += 0.02
        layers.append(tensor.requires_grad_())
    logits = torch.zeros(1, num_queries, 1, requires_grad=True)
    encoder_boxes = torch.full((1, num_queries, 4), 99.0, requires_grad=True)
    encoder_logits = torch.zeros(1, num_queries, 1, requires_grad=True)
    return {
        "pred_logits": logits,
        "pred_boxes": layers[-1],
        "aux_outputs": [
            {"pred_logits": logits, "pred_boxes": layer}
            for layer in layers[:-1]
        ] + [{"pred_logits": encoder_logits, "pred_boxes": encoder_boxes}],
        # Deliberately invalid DN values: MERT must never inspect this field.
        "dn_aux_outputs": [{
            "pred_logits": torch.full_like(logits, float("nan")),
            "pred_boxes": torch.full_like(layers[-1], float("nan")),
        }],
    }


class TestMicroShiftPairGenerator(unittest.TestCase):
    def test_image_shift_has_no_circular_wrapping(self):
        image = torch.arange(1, 8, dtype=torch.float32).reshape(1, 1, 1, 7)
        shifted = MicroShiftPairGenerator.shift_images(
            image, torch.tensor([[2, 0]])
        )
        self.assertTrue(torch.equal(
            shifted, torch.tensor([[[[0, 0, 1, 2, 3, 4, 5]]]], dtype=torch.float32)
        ))

    def test_exact_normalized_box_shift_and_inverse(self):
        target = make_target()
        _, shifted, excluded = MicroShiftPairGenerator.shift_target(
            target, dx=2, dy=-1, height=640, width=640
        )
        expected = torch.tensor([[0.5 + 2 / 640, 0.5 - 1 / 640, 0.1, 0.1]])
        self.assertTrue(torch.allclose(shifted["boxes"], expected, atol=1e-7))
        restored = shifted["boxes"] - torch.tensor([[2 / 640, -1 / 640, 0, 0]])
        self.assertTrue(torch.allclose(restored, target["boxes"], atol=1e-7))
        self.assertEqual(excluded, 0)

    def test_origin_identity_and_border_mask_are_preserved(self):
        target = make_target(box=(0.01, 0.5, 0.02, 0.1))
        original, shifted, _ = MicroShiftPairGenerator(
            {"choices": [-2], "max_pixels": 2}, debug=True
        ).shift_target(target, dx=-2, dy=-2, height=640, width=640)
        self.assertEqual(original["origin_gt_id"].tolist(), [0])
        self.assertEqual(shifted["origin_gt_id"].tolist(), [0])
        self.assertFalse(bool(shifted["mert_fully_visible"][0]))
        self.assertGreater(float(shifted["boxes"][0, 2]), 0.0)

    def test_sampling_forbids_zero_zero_and_is_per_image(self):
        generator = MicroShiftPairGenerator({
            "choices": [-1, 0, 1], "max_pixels": 1,
            "forbid_zero_zero": True, "per_image": True,
        })
        images = torch.randn(16, 3, 8, 8)
        targets = [make_target() for _ in range(16)]
        _, _, _, shifts = generator(images, targets)
        self.assertEqual(shifts.shape, (16, 2))
        self.assertTrue(all(tuple(shift) != (0, 0) for shift in shifts.tolist()))


class TestTrajectoryLoss(unittest.TestCase):
    def setUp(self):
        self.matcher = DesiredQueryMatcher()
        self.loss = RefinementTrajectoryEquivarianceLoss(
            self.matcher,
            trajectory={"weight": 0.2, "beta": 0.1, "layers": "all"},
            size_weight={"enabled": False},
            matching={
                "use_final_layer_matching": True,
                "exclude_dn_queries": True,
            },
            pair={"require_fully_visible": True},
            debug=True,
        )
        self.track = [
            [0.50, 0.50, 0.10, 0.10],
            [0.51, 0.49, 0.11, 0.09],
            [0.515, 0.495, 0.105, 0.095],
        ]

    def _targets(self):
        original = make_target(desired_query=0)
        shifted = make_target(desired_query=2)
        original["origin_gt_id"] = torch.tensor([0])
        shifted["origin_gt_id"] = torch.tensor([0])
        original["mert_fully_visible"] = torch.tensor([True])
        shifted["mert_fully_visible"] = torch.tensor([True])
        return [original], [shifted]

    def test_equivariant_trajectory_is_zero_with_different_queries(self):
        dx, dy = 2, -1
        shifted_track = [
            [box[0] + dx / 640, box[1] + dy / 640, box[2], box[3]]
            for box in self.track
        ]
        original_targets, shifted_targets = self._targets()
        value = self.loss(
            make_outputs(self.track, 0),
            make_outputs(shifted_track, 2),
            original_targets,
            shifted_targets,
            torch.tensor([[dx, dy]]),
            (640, 640),
        )
        self.assertLess(abs(float(value)), 1e-10)
        self.assertEqual(self.loss.last_debug_info["num_valid_pairs"], 1)
        self.assertEqual(len(self.loss.last_debug_info["trajectory_loss_per_layer"]), 2)

    def test_trajectory_perturbation_is_positive_and_backward_works(self):
        dx, dy = 2, -1
        shifted_track = [
            [box[0] + dx / 640, box[1] + dy / 640, box[2], box[3]]
            for box in self.track
        ]
        outputs_o = make_outputs(self.track, 0)
        outputs_s = make_outputs(shifted_track, 2, perturb_layer=1)
        original_targets, shifted_targets = self._targets()
        value = self.loss(
            outputs_o, outputs_s, original_targets, shifted_targets,
            torch.tensor([[dx, dy]]), (640, 640)
        )
        self.assertGreater(float(value), 0.0)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNotNone(outputs_s["aux_outputs"][1]["pred_boxes"].grad)
        self.assertGreater(
            float(outputs_s["aux_outputs"][1]["pred_boxes"].grad.abs().sum()), 0.0
        )

    def test_encoder_proposal_and_dn_outputs_are_excluded(self):
        outputs = make_outputs(self.track, 0)
        trajectory = self.loss.decoder_box_trajectory(outputs)
        self.assertEqual(len(trajectory), 3)
        self.assertTrue(all(float(layer.max()) < 1.0 for layer in trajectory))

    def test_no_valid_pair_returns_finite_graph_zero(self):
        outputs_o = make_outputs(self.track, 0)
        outputs_s = make_outputs(self.track, 2)
        original_targets, shifted_targets = self._targets()
        shifted_targets[0]["mert_fully_visible"][:] = False
        value = self.loss(
            outputs_o, outputs_s, original_targets, shifted_targets,
            torch.tensor([[1, 0]]), (640, 640)
        )
        self.assertEqual(float(value), 0.0)
        self.assertTrue(torch.isfinite(value))
        value.backward()

    def test_tiny_object_size_weights(self):
        weighted = RefinementTrajectoryEquivarianceLoss(
            self.matcher,
            size_weight={
                "enabled": True, "reference_area": 256.0, "gamma": 0.5,
                "min_weight": 1.0, "max_weight": 4.0,
            },
        )
        cases = ((16, 1.0), (8, 2.0), (4, 4.0), (32, 1.0))
        for side, expected in cases:
            box = torch.tensor([0.5, 0.5, side / 640, side / 640])
            self.assertAlmostEqual(
                float(weighted._size_weight(box, 640, 640)), expected, places=5
            )


class TestMERTV2Loss(unittest.TestCase):
    def _loss(self, trajectory=None, size_weight=None, debug=True):
        trajectory_cfg = {
            "weight": 0.10,
            "beta": 0.01,
            "layers": "last_2",
            "xy_weight": 1.0,
            "wh_weight": 0.25,
        }
        if trajectory:
            trajectory_cfg.update(trajectory)
        return RefinementTrajectoryEquivarianceLoss(
            DesiredQueryMatcher(),
            trajectory=trajectory_cfg,
            size_weight=size_weight or {"enabled": False},
            debug=debug,
        )

    @staticmethod
    def _targets():
        original = make_target(desired_query=0)
        shifted = make_target(desired_query=2)
        original["origin_gt_id"] = torch.tensor([0])
        shifted["origin_gt_id"] = torch.tensor([0])
        original["mert_fully_visible"] = torch.tensor([True])
        shifted["mert_fully_visible"] = torch.tensor([True])
        return [original], [shifted]

    def test_beta_001_is_passed_to_smooth_l1(self):
        loss = self._loss(trajectory={"layers": "all"})
        original_targets, shifted_targets = self._targets()
        track = [[0.5, 0.5, 0.1, 0.1], [0.51, 0.5, 0.1, 0.1]]
        with mock.patch(
            "src.solver.mert.F.smooth_l1_loss",
            wraps=torch.nn.functional.smooth_l1_loss,
        ) as smooth_l1:
            loss(
                make_outputs(track, 0),
                make_outputs(track, 2),
                original_targets,
                shifted_targets,
                torch.tensor([[1, 0]]),
                (640, 640),
            )
        self.assertEqual(smooth_l1.call_args.kwargs["beta"], 0.01)

    def test_v2_loss_defaults(self):
        loss = RefinementTrajectoryEquivarianceLoss(DesiredQueryMatcher())
        self.assertEqual(loss.weight, 0.10)
        self.assertEqual(loss.beta, 0.01)
        self.assertEqual(loss.layers, "last_2")
        self.assertEqual(loss.xy_weight, 1.0)
        self.assertEqual(loss.wh_weight, 0.25)
        pair_generator = MicroShiftPairGenerator()
        self.assertEqual(pair_generator.max_pixels, 1)
        self.assertEqual(pair_generator.choices, (-1, 0, 1))

    def test_last_n_and_explicit_transition_selection(self):
        loss = self._loss()
        self.assertEqual(loss._transition_indices(6), [3, 4])
        loss.layers = "last_1"
        self.assertEqual(loss._transition_indices(6), [4])
        loss.layers = "last_3"
        self.assertEqual(loss._transition_indices(6), [2, 3, 4])
        loss.layers = [3, 4]
        self.assertEqual(loss._transition_indices(6), [3, 4])

    def test_xy_and_wh_losses_are_separate_and_weighted(self):
        original_targets, shifted_targets = self._targets()
        base = [[0.50, 0.50, 0.10, 0.10], [0.50, 0.50, 0.10, 0.10]]
        shifted_xy = [
            [0.50 + 1 / 640, 0.50, 0.10, 0.10],
            [0.52 + 1 / 640, 0.50, 0.10, 0.10],
        ]
        xy_info = self._loss(trajectory={"layers": "all"})
        xy_value = xy_info(
            make_outputs(base, 0), make_outputs(shifted_xy, 2),
            original_targets, shifted_targets, torch.tensor([[1, 0]]),
            (640, 640),
        )
        self.assertGreater(xy_info.last_debug_info["xy_loss"], 0.0)
        self.assertEqual(xy_info.last_debug_info["wh_loss"], 0.0)
        self.assertAlmostEqual(
            float(xy_value),
            0.10 * xy_info.last_debug_info["xy_loss"],
            places=7,
        )

        shifted_wh = [
            [0.50 + 1 / 640, 0.50, 0.10, 0.10],
            [0.50 + 1 / 640, 0.50, 0.12, 0.10],
        ]
        wh_info = self._loss(trajectory={"layers": "all"})
        wh_value = wh_info(
            make_outputs(base, 0), make_outputs(shifted_wh, 2),
            original_targets, shifted_targets, torch.tensor([[1, 0]]),
            (640, 640),
        )
        self.assertEqual(wh_info.last_debug_info["xy_loss"], 0.0)
        self.assertGreater(wh_info.last_debug_info["wh_loss"], 0.0)
        self.assertAlmostEqual(
            float(wh_value),
            0.10 * 0.25 * wh_info.last_debug_info["wh_loss"],
            places=7,
        )

    def test_warmup_hold_decay_schedule(self):
        schedule = {
            "enabled": True,
            "start_epoch": 20,
            "warmup_end_epoch": 40,
            "hold_end_epoch": 100,
            "end_epoch": 160,
        }
        expected = {
            0: 0.0, 19: 0.0, 20: 0.0, 30: 0.05, 40: 0.10,
            80: 0.10, 99: 0.10, 100: 0.10, 130: 0.05,
            160: 0.0, 180: 0.0, 200: 0.0,
        }
        for epoch, value in expected.items():
            self.assertAlmostEqual(
                get_mert_weight(epoch, 0.10, schedule), value, places=8
            )

    def test_small_object_weight_and_area_filter(self):
        loss = self._loss(size_weight={
            "enabled": True,
            "reference_area": 256.0,
            "gamma": 0.25,
            "max_object_area": 1024.0,
            "min_weight": 1.0,
            "max_weight": 2.5,
        })
        expected = {8: 4 ** 0.25, 16: 1.0, 32: 1.0, 64: 0.0}
        for side, value in expected.items():
            box = torch.tensor([0.5, 0.5, side / 640, side / 640])
            self.assertAlmostEqual(
                float(loss._size_weight(box, 640, 640)), value, places=5
            )

        sides = torch.tensor([8.0, 16.0, 32.0, 64.0])
        boxes = torch.stack([
            torch.full_like(sides, 0.5),
            torch.full_like(sides, 0.5),
            sides / 640,
            sides / 640,
        ], dim=-1)
        original_target = {
            "boxes": boxes,
            "labels": torch.zeros(4, dtype=torch.int64),
            "area": sides.square(),
            "origin_gt_id": torch.arange(4),
            "mert_fully_visible": torch.ones(4, dtype=torch.bool),
            "desired_query": 0,
        }
        shifted_target = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in original_target.items()
        }
        shifted_target["boxes"][:, 0] += 1 / 640

        def outputs(layer_boxes):
            logits = torch.zeros(1, 4, 1, requires_grad=True)
            layers = [layer_boxes[None].clone().requires_grad_() for _ in range(3)]
            return {
                "pred_logits": logits,
                "pred_boxes": layers[-1],
                "aux_outputs": [
                    {"pred_logits": logits, "pred_boxes": layers[0]},
                    {"pred_logits": logits, "pred_boxes": layers[1]},
                    {"pred_logits": logits, "pred_boxes": torch.zeros_like(layers[0])},
                ],
            }

        loss(
            outputs(boxes), outputs(shifted_target["boxes"]),
            [original_target], [shifted_target], torch.tensor([[1, 0]]),
            (640, 640),
        )
        self.assertEqual(loss.last_debug_info["num_total_gt"], 4)
        self.assertEqual(loss.last_debug_info["num_gt_filtered_by_area"], 1)
        self.assertEqual(loss.last_debug_info["num_mert_gt"], 3)

    def test_legacy_yaml_defaults_remain_compatible(self):
        plugin = MERTTrainingPlugin(
            MERT_CONFIG, DesiredQueryMatcher(), current_epoch=80
        )
        self.assertEqual(plugin.current_mert_weight, 0.20)
        self.assertTrue(plugin.shifted_detection_loss)
        self.assertEqual(plugin.trajectory_loss.xy_weight, 0.25)
        self.assertEqual(plugin.trajectory_loss.wh_weight, 0.25)
        self.assertIsNone(plugin.trajectory_loss.max_object_area)


class TestMERTIntegration(unittest.TestCase):
    def test_disabled_plugin_creates_nothing(self):
        plugin = MERTTrainingPlugin({"enabled": False}, DesiredQueryMatcher())
        self.assertFalse(plugin.enabled)
        self.assertIsNone(plugin.pair_generator)
        self.assertIsNone(plugin.trajectory_loss)

    def test_concat_split_keeps_all_decoder_layers(self):
        first = make_outputs([[0.1] * 4, [0.2] * 4, [0.3] * 4], 0)
        second = make_outputs([[0.4] * 4, [0.5] * 4, [0.6] * 4], 0)
        combined = {
            "pred_logits": torch.cat([first["pred_logits"], second["pred_logits"]]),
            "pred_boxes": torch.cat([first["pred_boxes"], second["pred_boxes"]]),
            "aux_outputs": [
                {
                    "pred_logits": torch.cat([a["pred_logits"], b["pred_logits"]]),
                    "pred_boxes": torch.cat([a["pred_boxes"], b["pred_boxes"]]),
                }
                for a, b in zip(first["aux_outputs"], second["aux_outputs"])
            ],
            "dn_aux_outputs": [
                {
                    "pred_logits": torch.cat([
                        first["dn_aux_outputs"][0]["pred_logits"],
                        second["dn_aux_outputs"][0]["pred_logits"],
                    ]),
                    "pred_boxes": torch.cat([
                        first["dn_aux_outputs"][0]["pred_boxes"],
                        second["dn_aux_outputs"][0]["pred_boxes"],
                    ]),
                }
            ],
            "dn_meta": {
                "dn_positive_idx": (
                    torch.tensor([0]), torch.tensor([1])
                ),
                "dn_num_group": 1,
                "dn_num_split": [3, 3],
            },
        }
        split_first, split_second = MERTTrainingPlugin.split_concatenated_outputs(
            combined, 1
        )
        self.assertTrue(torch.equal(split_first["pred_boxes"], first["pred_boxes"]))
        self.assertTrue(torch.equal(split_second["pred_boxes"], second["pred_boxes"]))
        self.assertEqual(len(split_first["aux_outputs"]), len(first["aux_outputs"]))
        self.assertEqual(split_first["dn_aux_outputs"][0]["pred_boxes"].shape[0], 1)
        self.assertEqual(len(split_first["dn_meta"]["dn_positive_idx"]), 1)
        self.assertEqual(
            split_second["dn_meta"]["dn_positive_idx"][0].tolist(), [1]
        )

    def test_yaml_configs_load_and_do_not_change_model_schema(self):
        legacy_names = (
            "rtdetr_r18vd_200e_dut_anti_uav_mert.yml",
            "rtdetr_r34vd_200e_dut_anti_uav_mert.yml",
            "rtdetr_r50vd_200e_dut_anti_uav_mert.yml",
            "rtdetr_r101vd_200e_dut_anti_uav_mert.yml",
        )
        for name in legacy_names:
            config = load_config(str(PROJECT_DIR / "configs" / "rtdetr" / name), {})
            self.assertTrue(config["MERT"]["enabled"])
            self.assertEqual(config["MERT"]["forward_mode"], "concat")
            self.assertNotIn("MERT", config["RTDETR"])

        v2_names = (
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp0_baseline.yml",
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp1_legacy.yml",
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp2_beta.yml",
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp3_late.yml",
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml",
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp5_full.yml",
        )
        output_dirs = []
        for name in v2_names:
            config = load_config(str(PROJECT_DIR / "configs" / "rtdetr" / name), {})
            self.assertNotIn("MERT", config["RTDETR"])
            output_dirs.append(config["output_dir"])
        self.assertEqual(len(output_dirs), len(set(output_dirs)))
        full = load_config(
            str(PROJECT_DIR / "configs" / "rtdetr" / v2_names[-1]), {}
        )["MERT"]
        self.assertEqual(full["trajectory"]["beta"], 0.01)
        self.assertEqual(full["trajectory"]["layers"], "last_2")
        self.assertFalse(full["supervision"]["shifted_detection_loss"])

    def test_real_r18_model_forward_backward_and_parameter_identity(self):
        config_path = PROJECT_DIR / "configs" / "rtdetr" / \
            "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp5_full.yml"
        config = YAMLConfig(str(config_path))
        config.yaml_cfg["PResNet"]["pretrained"] = False
        config.yaml_cfg["RTDETR"]["multi_scale"] = None
        config.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = None
        config.yaml_cfg["RTDETRTransformer"]["eval_spatial_size"] = None
        model = config.model
        criterion = config.criterion
        parameter_names_before = list(model.state_dict())
        parameter_count_before = sum(parameter.numel() for parameter in model.parameters())

        plugin = MERTTrainingPlugin(
            config.yaml_cfg["MERT"], criterion.matcher, current_epoch=80
        )
        # Three feature levels contain 336 locations at 128x128, enough for
        # the unchanged 300-query RT-DETR top-k selection.
        images = torch.randn(1, 3, 128, 128)
        targets = [make_target(box=(0.5, 0.5, 0.02, 0.02))]
        pair = plugin.prepare(model, images, targets)
        paired_images = torch.cat([
            pair["original_images"], pair["shifted_images"]
        ])
        paired_targets = pair["original_targets"] + pair["shifted_targets"]

        model.train()
        outputs = model(paired_images, paired_targets)
        outputs_o, outputs_s = plugin.split_concatenated_outputs(outputs, 1)
        detection_losses = criterion(outputs_o, pair["original_targets"])
        mert_loss = plugin.calculate_loss(outputs_o, outputs_s, pair)
        total_loss = sum(detection_losses.values()) + mert_loss
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        original_gradient = model.backbone.res_layers[1].blocks[0].branch2a.conv.weight.grad
        self.assertIsNotNone(original_gradient)
        self.assertTrue(torch.isfinite(original_gradient).all())
        self.assertEqual(parameter_names_before, list(model.state_dict()))
        self.assertEqual(
            parameter_count_before,
            sum(parameter.numel() for parameter in model.parameters()),
        )
        clone = YAMLConfig(str(config_path))
        clone.yaml_cfg["PResNet"]["pretrained"] = False
        clone.yaml_cfg["RTDETR"]["multi_scale"] = None
        clone.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = None
        clone.yaml_cfg["RTDETRTransformer"]["eval_spatial_size"] = None
        clone_model = clone.model
        clone_model.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        clone_model.eval()
        with torch.no_grad():
            inference_output = model(images)
            clone_output = clone_model(images)
        self.assertEqual(inference_output["pred_logits"].shape, (1, 300, 1))
        self.assertEqual(inference_output["pred_boxes"].shape, (1, 300, 4))
        self.assertTrue(torch.equal(
            inference_output["pred_logits"], clone_output["pred_logits"]
        ))
        self.assertTrue(torch.equal(
            inference_output["pred_boxes"], clone_output["pred_boxes"]
        ))

    def test_train_loop_disabled_concat_and_sequential_paths(self):
        class TinyDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.1))
                self.multi_scale = [8]
                self.batch_sizes = []

            def forward(self, images, targets=None):
                self.batch_sizes.append(images.shape[0])
                batch_size = images.shape[0]
                signal = images.mean(dim=(1, 2, 3)) * self.scale * 0.01
                base = images.new_tensor([0.5, 0.5, 0.1, 0.1])[None, None]
                base = base.expand(batch_size, 3, 4).clone()
                base[:, :, 0] = base[:, :, 0] + signal[:, None]
                layers = [base + self.scale * value for value in (0.0, 0.001, 0.002)]
                logits = self.scale.expand(batch_size, 3, 1)
                encoder = base + self.scale * 0.003
                return {
                    "pred_logits": logits,
                    "pred_boxes": layers[-1],
                    "aux_outputs": [
                        {"pred_logits": logits, "pred_boxes": layers[0]},
                        {"pred_logits": logits, "pred_boxes": layers[1]},
                        {"pred_logits": logits, "pred_boxes": encoder},
                    ],
                }

        class TinyCriterion(nn.Module):
            def __init__(self):
                super().__init__()
                self.matcher = DesiredQueryMatcher()
                self.batch_sizes = []

            def forward(self, outputs, targets):
                self.batch_sizes.append(len(targets))
                return {
                    "loss_det": outputs["pred_boxes"].square().mean()
                    + outputs["pred_logits"].square().mean()
                }

        def run(config):
            model = TinyDetector()
            criterion = TinyCriterion()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            target = make_target()
            target.pop("desired_query")
            train_one_epoch(
                model,
                criterion,
                [(torch.randn(1, 3, 8, 8), [target])],
                optimizer,
                torch.device("cpu"),
                epoch=0,
                print_freq=10,
                mert_config=config,
            )
            self.assertTrue(torch.isfinite(model.scale.grad))
            self.assertEqual(model.multi_scale, [8])
            model.criterion_batch_sizes = criterion.batch_sizes
            return model

        disabled = run({"enabled": False})
        self.assertEqual(disabled.batch_sizes, [1])
        self.assertEqual(disabled.criterion_batch_sizes, [1])

        concat_config = copy.deepcopy(MERT_CONFIG)
        concat = run(concat_config)
        self.assertEqual(concat.batch_sizes, [2])
        self.assertEqual(concat.criterion_batch_sizes, [2])

        sequential_config = copy.deepcopy(MERT_CONFIG)
        sequential_config["forward_mode"] = "sequential"
        sequential = run(sequential_config)
        self.assertEqual(sequential.batch_sizes, [1, 1])
        self.assertEqual(sequential.criterion_batch_sizes, [1, 1])

        fair_config = copy.deepcopy(MERT_CONFIG)
        fair_config["supervision"] = {"shifted_detection_loss": False}
        fair = run(fair_config)
        # The shifted image is still forwarded for its trajectory, but only
        # the original view enters the unchanged detection criterion.
        self.assertEqual(fair.batch_sizes, [2])
        self.assertEqual(fair.criterion_batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
