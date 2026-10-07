from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "inspect_resume_checkpoint.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("inspect_resume_checkpoint", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _complete_state(epoch: int, *, aliases: bool = False):
    if aliases:
        return {
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {0: {}}, "param_groups": [{}]},
            "scheduler_state_dict": {"last_epoch": epoch},
            "epoch": epoch,
            "model_ema": {"module": {"weight": torch.tensor([1.0])}},
            "grad_scaler": {"scale": 65536.0},
            "best_stat": {"epoch": epoch - 1, "coco_eval_bbox": 0.5},
        }
    return {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {}}, "param_groups": [{}]},
        "lr_scheduler": {"last_epoch": epoch},
        "last_epoch": epoch,
        "ema": {"module": {"weight": torch.tensor([1.0])}},
        "scaler": {"scale": 65536.0},
        "best_stat": {"epoch": epoch - 1, "coco_eval_bbox": 0.5},
    }


class InspectResumeCheckpointTest(unittest.TestCase):
    def test_complete_last_has_absolute_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(_complete_state(12), root / "last.pth")
            torch.save(_complete_state(70), root / "checkpoint0070.pth")
            torch.save(_complete_state(99), root / "best.pth")
            result = TOOL.inspect_output_dir(root)
        self.assertTrue(result["strict_resume_possible"])
        self.assertEqual(result["checkpoint_kind"], "last")
        self.assertEqual(Path(result["checkpoint"]).name, "last.pth")
        self.assertEqual(result["saved_epoch"], 12)
        self.assertEqual(result["next_epoch"], 13)

    def test_checkpoint_group_uses_saved_epoch_and_includes_rolling_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Filename order is deliberately misleading; saved epoch is authoritative.
            torch.save(_complete_state(5), root / "checkpoint9999.pth")
            torch.save(_complete_state(83), root / "checkpoint0001.pth")
            torch.save(_complete_state(87), root / "checkpoint.pth")
            torch.save(_complete_state(150), root / "best.pth")
            result = TOOL.inspect_output_dir(root)
        self.assertEqual(Path(result["checkpoint"]).name, "checkpoint.pth")
        self.assertEqual(result["saved_epoch"], 87)
        self.assertEqual(result["checkpoint_kind"], "checkpoint")

    def test_invalid_last_falls_through_and_aliases_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = _complete_state(40)
            incomplete.pop("optimizer")
            torch.save(incomplete, root / "last.pth")
            torch.save(_complete_state(41, aliases=True), root / "checkpoint0041.pth")
            result = TOOL.inspect_output_dir(root)
        self.assertEqual(Path(result["checkpoint"]).name, "checkpoint0041.pth")
        self.assertEqual(result["saved_epoch"], 41)
        self.assertTrue(all(result["states"].values()))
        self.assertEqual(result["keys"]["epoch"], "epoch")
        self.assertEqual(result["keys"]["scaler"], "grad_scaler")

    def test_best_is_only_last_resort_and_must_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(_complete_state(55), root / "best.pth")
            result = TOOL.inspect_output_dir(root)
        self.assertEqual(Path(result["checkpoint"]).name, "best.pth")
        self.assertEqual(result["checkpoint_kind"], "best")
        self.assertEqual(result["saved_epoch"], 55)

    def test_model_only_weights_are_rejected_with_required_phrase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"model": {"weight": torch.tensor([1.0])}}, root / "best.pth")
            json_path = root / "audit.json"
            output = io.StringIO()
            with redirect_stdout(output):
                rc = TOOL.main([str(root), "--json-out", str(json_path)])
            report = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(rc, 2)
        self.assertFalse(report["strict_resume_possible"])
        self.assertIsNone(report["checkpoint"])
        self.assertIn("STRICT RESUME NOT POSSIBLE", output.getvalue())
        self.assertIn("OPTIMIZER_STATE_LOADED=NO", output.getvalue())
        self.assertIn("SCHEDULER_STATE_LOADED=NO", output.getvalue())
        self.assertIn("EMA_STATE_LOADED=NO", output.getvalue())
        self.assertIn("AMP_SCALER_STATE_LOADED=NO", output.getvalue())

    def test_epoch_199_is_completed_for_200_epoch_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(_complete_state(199), root / "checkpoint.pth")
            result = TOOL.inspect_output_dir(root, target_epochs=200)
        self.assertTrue(result["completed"])
        self.assertEqual(result["saved_epoch"], 199)
        self.assertEqual(result["next_epoch"], 200)

    def test_corrupt_candidate_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pth").write_bytes(b"not a torch checkpoint")
            result = TOOL.inspect_output_dir(root)
        self.assertFalse(result["strict_resume_possible"])
        self.assertFalse(result["candidates"][0]["load_ok"])
        self.assertTrue(result["candidates"][0]["error"])

    def test_target_epochs_must_be_positive(self):
        with self.assertRaisesRegex(SystemExit, "positive"):
            TOOL.main(["unused", "--target-epochs", "0"])


if __name__ == "__main__":
    unittest.main()
