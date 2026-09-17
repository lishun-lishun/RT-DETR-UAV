"""Standard-library tests of fairness guards and inference metric pairing."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools/analyze_dut_models.py"
SPEC = importlib.util.spec_from_file_location("dut_analyze", SCRIPT)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


class FairnessGuardTests(unittest.TestCase):
    def test_nested_differences_include_absent_fields(self):
        diff = analyze.differences({"optimizer": {"lr": 0.0001}},
                                   {"optimizer": {"lr": 0.01}, "MERT": {"enabled": True}})
        self.assertEqual(set(diff), {"optimizer.lr", "MERT.enabled"})
        self.assertEqual(diff["MERT.enabled"]["reference"], "<absent>")

    def test_hyperparameters_cannot_be_exempted(self):
        for key in ("optimizer.lr", "optimizer.params", "epoches", "use_amp",
                    "train_dataloader.batch_size", "RTDETR.multi_scale",
                    "HybridEncoder.eval_spatial_size", "RTDETRTransformer.num_queries",
                    "train_dataloader.dataset.transforms.ops"):
            self.assertFalse(analyze.permitted_difference(key, original=True), key)
            self.assertFalse(analyze.permitted_difference(key), key)

    def test_dataset_adaptations_only_exempted_against_official(self):
        for key in ("num_classes", "remap_mscoco_category", "test_dataset.ann_file",
                    "train_dataloader.dataset.img_folder", "val_dataloader.dataset.ann_file"):
            self.assertTrue(analyze.permitted_difference(key, original=True), key)
            self.assertFalse(analyze.permitted_difference(key), key)

    def test_method_switches_exempted(self):
        for key in ("MERT.enabled", "MERT.beta", "SECD.transitions", "SECD.enabled"):
            self.assertTrue(analyze.permitted_difference(key))

    def test_loader_receives_explicit_fresh_accumulator(self):
        seen = []

        def loader(path, cfg):
            seen.append(cfg)
            cfg["path"] = path
            return cfg

        with patch.object(analyze.runpy, "run_path", return_value={"load_config": loader}):
            analyze.fresh_config(Path("first.yml"))
            analyze.fresh_config(Path("second.yml"))
        self.assertIsNot(seen[0], seen[1])


class PairingTests(unittest.TestCase):
    def metrics(self):
        sample = {
            "params": 10, "trainable_params": 10, "structure_sha256": "structure",
            "state_sha256": "weights", "output_shapes": {"pred_boxes": [1, 300, 4]},
            "output_sha256": "output", "forward_calls": 1,
            "module_execution_sha256": "execution",
            "conv_linear_macs_lower_bound": 20, "torch_profiler_flops_lower_bound": 40,
            "counted_profiler_operators": {"aten::mm": {"flops": 40, "events": 1}},
            "uncounted_operator_events_including_metadata": {"aten::softmax": 1},
        }
        return {method: copy.deepcopy(sample) for method in analyze.METHODS}

    def test_all_three_pairs_compare_real_values(self):
        checks = analyze.validate_pairings(self.metrics())
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(check["passed"] for check in checks.values()))

    def test_equal_flops_but_changed_outputs_fails(self):
        metrics = self.metrics()
        metrics["MERT"]["output_sha256"] = "different"
        with self.assertRaisesRegex(AssertionError, "output_sha256"):
            analyze.validate_pairings(metrics)

    def test_equal_outputs_but_extra_model_execution_fails(self):
        metrics = self.metrics()
        metrics["SECD34+MERT"]["module_execution_sha256"] = "second-forward"
        with self.assertRaisesRegex(AssertionError, "module_execution_sha256"):
            analyze.validate_pairings(metrics)

    def test_uncounted_profiler_events_are_diagnostic_only(self):
        metrics = self.metrics()
        metrics["SECD34+MERT"]["uncounted_operator_events_including_metadata"]["aten::div_"] = 1
        checks = analyze.validate_pairings(metrics)
        self.assertTrue(checks["SECD34 == SECD34+MERT"]["passed"])
        self.assertFalse(checks["SECD34 == SECD34+MERT"]["uncounted_operator_event_counts_equal"])


if __name__ == "__main__":
    unittest.main()
