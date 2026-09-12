"""Unit and integration tests for the CURE feature plug-in.

Run from ``rtdetr_pytorch``:

    python -m unittest tests.test_mert tests.test_cure -v
"""

import copy
import sys
import types
import unittest
from pathlib import Path

import torch


# Importing the project's src package also imports optional dataset/backbone
# integrations which are unrelated to CURE.  Keep this test self-contained.
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
CONFIG_DIR = PROJECT_DIR / "configs" / "rtdetr"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.core import YAMLConfig  # noqa: E402
from src.core.yaml_utils import load_config  # noqa: E402
from src.nn.backbone.cure import (  # noqa: E402
    ContextUnexplainedResidualEnhancement,
    MaskedRingConv2d,
)
from src.nn.backbone.presnet import PResNet  # noqa: E402


BASELINE_CONFIG = "rtdetr_r18vd_200e_dut_anti_uav_baseline.yml"
CURE_CONFIG = "rtdetr_r18vd_200e_dut_anti_uav_cure.yml"
MERT_CONFIG = "rtdetr_r18vd_200e_dut_anti_uav_mert_v2_exp4_late_xywh.yml"
CURE_MERT_CONFIG = "rtdetr_r18vd_200e_dut_anti_uav_cure_mert.yml"


def build_model(config_name):
    config = YAMLConfig(str(CONFIG_DIR / config_name))
    config.yaml_cfg["PResNet"]["pretrained"] = False
    config.yaml_cfg["RTDETR"]["multi_scale"] = None
    config.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = None
    config.yaml_cfg["RTDETRTransformer"]["eval_spatial_size"] = None
    return config, config.model


class CUREUnitTests(unittest.TestCase):
    def test_5x5_ring_mask_exactly_excludes_center_3x3(self):
        convolution = MaskedRingConv2d(
            2, kernel_size=5, exclude_center_size=3, depthwise=True
        )
        expected = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(convolution.kernel_mask[0, 0], expected))
        self.assertEqual(int(convolution.kernel_mask.sum()), 16)
        self.assertTrue(torch.equal(
            convolution.masked_weight[:, :, 1:4, 1:4],
            torch.zeros_like(convolution.masked_weight[:, :, 1:4, 1:4]),
        ))

    def test_mask_remains_effective_after_optimizer_step(self):
        torch.manual_seed(0)
        convolution = MaskedRingConv2d(
            3, kernel_size=5, exclude_center_size=3, depthwise=True
        )
        optimizer = torch.optim.AdamW(convolution.parameters(), lr=0.1)
        loss = convolution(torch.randn(2, 3, 9, 9)).square().mean()
        loss.backward()
        optimizer.step()

        center = convolution.masked_weight[:, :, 1:4, 1:4]
        center_gradient = convolution.weight.grad[:, :, 1:4, 1:4]
        self.assertEqual(torch.count_nonzero(center).item(), 0)
        self.assertEqual(torch.count_nonzero(center_gradient).item(), 0)

    def test_ring_size_validation(self):
        invalid_pairs = [(4, 1), (5, 2), (3, 3), (3, 5), (0, 1)]
        for kernel_size, exclude_size in invalid_pairs:
            with self.subTest(kernel=kernel_size, exclude=exclude_size):
                with self.assertRaises(ValueError):
                    MaskedRingConv2d(
                        4,
                        kernel_size=kernel_size,
                        exclude_center_size=exclude_size,
                    )

        three = MaskedRingConv2d(4, 3, 1)
        seven = MaskedRingConv2d(4, 7, 3)
        self.assertEqual(int(three.kernel_mask.sum()), 8)
        self.assertEqual(int(seven.kernel_mask.sum()), 40)

    def test_shape_discrepancy_gate_and_debug_statistics(self):
        module = ContextUnexplainedResidualEnhancement(
            16, alpha_init=0.0, debug=True
        ).eval()
        x = torch.randn(2, 16, 11, 13)
        with torch.no_grad():
            output = module(x)
        self.assertTrue(torch.equal(output, x))
        expected_shapes = {
            "input_shape": tuple(x.shape),
            "context_pred_shape": tuple(x.shape),
            "residual_shape": tuple(x.shape),
            "discrepancy_shape": (2, 1, 11, 13),
            "gate_shape": (2, 1, 11, 13),
            "output_shape": tuple(x.shape),
        }
        for key, value in expected_shapes.items():
            self.assertEqual(module.last_debug_info[key], value)
        for key in (
            "context_abs_mean", "residual_abs_mean", "residual_abs_std",
            "discrepancy_mean", "discrepancy_std", "discrepancy_min",
            "discrepancy_max", "gate_mean", "gate_std", "gate_min",
            "gate_max", "enhancement_abs_mean",
            "enhancement_to_feature_ratio",
        ):
            self.assertIn(key, module.last_debug_info)
        self.assertGreaterEqual(module.last_debug_info["discrepancy_min"], 0.0)
        self.assertLessEqual(module.last_debug_info["discrepancy_max"], 2.0)
        self.assertGreaterEqual(module.last_debug_info["gate_min"], 0.0)
        self.assertLessEqual(module.last_debug_info["gate_max"], 1.0)
        self.assertEqual(
            tuple(module.last_debug_maps["residual_magnitude"].shape),
            (2, 1, 11, 13),
        )

    def test_alpha_zero_identity_then_internal_gradients_flow(self):
        torch.manual_seed(1)
        module = ContextUnexplainedResidualEnhancement(16, alpha_init=0.0)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        output = module(x)
        self.assertTrue(torch.equal(output, x))

        module.zero_grad(set_to_none=True)
        module.alpha.data.fill_(0.1)
        module(x).square().mean().backward()
        parameters = {
            "context": module.context_predictor.ring_conv.weight,
            "residual": module.residual_proj.conv.weight,
            "gate": module.gate.reduce.conv.weight,
            "alpha": module.alpha,
        }
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_residual_only_does_not_multiply_by_discrepancy(self):
        module = ContextUnexplainedResidualEnhancement(
            8, gate_enabled=False, alpha_init=0.2
        ).eval()
        x = torch.randn(1, 8, 7, 7)
        with torch.no_grad():
            context = module.context_predictor(x)
            embedding = module.residual_proj(x - context)
            expected = x + module.alpha * embedding
            actual = module(x)
        self.assertTrue(torch.allclose(actual, expected, atol=0.0, rtol=0.0))

    def test_parameter_count_for_formal_s3_module(self):
        module = ContextUnexplainedResidualEnhancement(128)
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 44930)


class CUREIntegrationTests(unittest.TestCase):
    def test_target_stages_use_actual_r18_channels_and_preserve_shapes(self):
        backbone = PResNet(
            depth=18,
            variant="d",
            return_idx=[1, 2, 3],
            freeze_norm=False,
            pretrained=False,
            cure_config={
                "enabled": True,
                "target_strides": [8, 16],
                "fusion": {"alpha_init": 0.0},
            },
        ).eval()
        self.assertEqual(backbone.cure_s3.channels, 128)
        self.assertEqual(backbone.cure_s4.channels, 256)
        self.assertFalse(hasattr(backbone, "cure_s5"))
        with torch.no_grad():
            outputs = backbone(torch.randn(1, 3, 65, 67))
        self.assertEqual([item.shape[1] for item in outputs], [128, 256, 512])
        self.assertEqual(backbone.out_strides, [8, 16, 32])

    def test_disabled_cure_does_not_instantiate_module_or_change_forward(self):
        plain = PResNet(
            depth=18, return_idx=[1, 2, 3], freeze_norm=False, pretrained=False
        ).eval()
        disabled = PResNet(
            depth=18,
            return_idx=[1, 2, 3],
            freeze_norm=False,
            pretrained=False,
            cure_config={"enabled": False},
        ).eval()
        disabled.load_state_dict(plain.state_dict(), strict=True)
        self.assertFalse(any("cure_" in name for name, _ in disabled.named_modules()))
        x = torch.randn(1, 3, 97, 99)
        with torch.no_grad():
            expected = plain(x)
            actual = disabled(x)
        for expected_feature, actual_feature in zip(expected, actual):
            self.assertTrue(torch.equal(expected_feature, actual_feature))

    def test_baseline_checkpoint_only_misses_new_cure_keys(self):
        plain = PResNet(
            depth=18, return_idx=[1, 2, 3], freeze_norm=False, pretrained=False
        )
        cure = PResNet(
            depth=18,
            return_idx=[1, 2, 3],
            freeze_norm=False,
            pretrained=False,
            cure_config={"enabled": True, "target_strides": [8]},
        )
        incompatible = cure.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith("cure_s3.") for key in incompatible.missing_keys
        ))
        original_keys = set(plain.state_dict())
        self.assertTrue(original_keys.issubset(set(cure.state_dict())))

    def test_four_modes_and_late_xywh_mert_are_exactly_preserved(self):
        modes = {
            BASELINE_CONFIG: (False, False),
            CURE_CONFIG: (True, False),
            MERT_CONFIG: (False, True),
            CURE_MERT_CONFIG: (True, True),
        }
        for config_name, (cure_enabled, mert_enabled) in modes.items():
            with self.subTest(config=config_name):
                config = load_config(str(CONFIG_DIR / config_name), {})
                self.assertEqual(config["CURE"]["enabled"], cure_enabled)
                self.assertEqual(config.get("MERT", {}).get("enabled", False), mert_enabled)
                nested = config["PResNet"].get("cure_config")
                self.assertEqual(bool(nested and nested.get("enabled")), cure_enabled)

        late = load_config(str(CONFIG_DIR / MERT_CONFIG), {})
        combined = load_config(str(CONFIG_DIR / CURE_MERT_CONFIG), {})
        standalone = load_config(str(CONFIG_DIR / CURE_CONFIG), {})
        self.assertEqual(late["MERT"], combined["MERT"])
        self.assertEqual(standalone["CURE"], combined["CURE"])
        self.assertEqual(standalone["optimizer"], combined["optimizer"])

    def test_full_model_alpha_zero_preserves_s3_and_predictions(self):
        _, baseline = build_model(BASELINE_CONFIG)
        _, cure = build_model(CURE_CONFIG)
        incompatible = cure.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(all(
            key.startswith("backbone.cure_s3.") for key in incompatible.missing_keys
        ))

        captured = {"baseline": None, "cure": None}
        baseline_handle = baseline.backbone.res_layers[1].register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("baseline", output)
        )
        cure_handle = cure.backbone.cure_s3.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("cure", output)
        )
        baseline.eval()
        cure.eval()
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            baseline_output = baseline(image)
            cure_output = cure(image)
        baseline_handle.remove()
        cure_handle.remove()

        self.assertTrue(torch.equal(captured["baseline"], captured["cure"]))
        self.assertTrue(torch.equal(
            baseline_output["pred_boxes"], cure_output["pred_boxes"]
        ))
        self.assertTrue(torch.equal(
            baseline_output["pred_logits"], cure_output["pred_logits"]
        ))

    def test_cure_mert_has_one_shared_cure_and_same_inference_model_as_cure(self):
        _, cure = build_model(CURE_CONFIG)
        combined_config, combined = build_model(CURE_MERT_CONFIG)
        cure_names = [
            name for name, module in combined.named_modules()
            if isinstance(module, ContextUnexplainedResidualEnhancement)
        ]
        self.assertEqual(cure_names, ["backbone.cure_s3"])
        self.assertFalse(any("mert" in name.lower() for name, _ in combined.named_modules()))
        self.assertEqual(
            sum(parameter.numel() for parameter in cure.parameters()),
            sum(parameter.numel() for parameter in combined.parameters()),
        )
        self.assertTrue(combined_config.yaml_cfg["MERT"]["enabled"])

        calls = []
        handle = combined.backbone.cure_s3.register_forward_hook(
            lambda _module, inputs, _output: calls.append(inputs[0].shape[0])
        )
        combined.eval()
        with torch.no_grad():
            combined(torch.randn(2, 3, 128, 128))
        handle.remove()
        self.assertEqual(calls, [2])

    def test_model_parameter_counts_match_expected_pairing(self):
        _, baseline = build_model(BASELINE_CONFIG)
        _, cure = build_model(CURE_CONFIG)
        _, mert = build_model(MERT_CONFIG)
        _, combined = build_model(CURE_MERT_CONFIG)
        counts = [
            sum(parameter.numel() for parameter in model.parameters())
            for model in (baseline, cure, mert, combined)
        ]
        self.assertEqual(counts[0], counts[2])
        self.assertEqual(counts[1], counts[3])
        self.assertEqual(counts[1] - counts[0], 44930)

    def test_cure_optimizer_lr_is_independent_from_pretrained_backbone(self):
        config, model = build_model(CURE_CONFIG)
        optimizer = config.optimizer
        learning_rates = {
            id(parameter): group["lr"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for name, parameter in model.named_parameters():
            if name.startswith("backbone.cure_"):
                self.assertEqual(learning_rates[id(parameter)], 1.0e-4)
            elif name.startswith("backbone."):
                self.assertEqual(learning_rates[id(parameter)], 1.0e-5)


if __name__ == "__main__":
    unittest.main()
