from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "train_fdcr_rdcf_spdr_3gpu.sh"


def _script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


class TrainQueueContractTest(unittest.TestCase):
    def test_queue_has_exact_fixed_order(self):
        text = _script()
        ordered_markers = [
            'emit "1. HRNetV2-W18+SPDR [RESUME]"',
            'emit "2. PResNet18+FDCR [FRESH]"',
            'emit "3. HRNetV2-W18+FDCR [FRESH]"',
            'emit "4. PResNet18+RDCF [FRESH]"',
            'emit "5. HRNetV2-W18+RDCF [FRESH]"',
        ]
        positions = [text.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Total tasks = 5", text)

    def test_spdr_is_strict_resume_and_new_experiments_are_fresh(self):
        text = _script()
        self.assertIn('SPDR_NAME="rtdetr_hrnetv2_w18_dut_anti_uav_spdr"', text)
        self.assertIn('run_training "HRNetV2-W18+SPDR" "RESUME"', text)
        self.assertIn('cmd+=( -r "$resume_path" )', text)
        self.assertIn('if [[ "$mode" == "RESUME" ]]', text)
        self.assertIn("STRICT RESUME NOT POSSIBLE", text)
        self.assertIn("FRESH_NAMES=(", text)
        self.assertIn('run_training "$label" "FRESH" "$config" "$output_dir"', text)

    def test_completion_and_fresh_skip_rules_are_distinct(self):
        text = _script()
        self.assertIn('RESUME_EPOCH" -ge "$TARGET_LAST_EPOCH', text)
        self.assertIn("[SKIP COMPLETED]", text)
        self.assertIn('if [[ -d "$output_dir" ]]', text)
        self.assertIn("reason=output_directory_exists", text)
        self.assertLess(text.index("[SKIP COMPLETED]"), text.index('if [[ -d "$output_dir" ]]'))

    def test_three_gpu_amp_seed_and_output_override_are_fixed(self):
        text = _script()
        self.assertIn('GPU_IDS="${CUDA_VISIBLE_DEVICES:-1,2,3}"', text)
        self.assertIn("this queue requires exactly three visible GPUs", text)
        self.assertIn("CUDA_VISIBLE_DEVICES entries must be distinct", text)
        self.assertIn("NPROC_PER_NODE=3", text)
        self.assertIn('"--nproc_per_node=$NPROC_PER_NODE"', text)
        self.assertIn('--output-dir "$output_dir"', text)
        self.assertIn("--amp", text)
        self.assertIn("--seed 0", text)

    def test_dry_run_never_reaches_training_pipeline(self):
        text = _script()
        dry_guard = text.index("if [[ $DRY_RUN -eq 1 ]]")
        dry_return = text.index("return 0", dry_guard)
        training_pipeline = text.index('CUDA_VISIBLE_DEVICES="$GPU_IDS" "${cmd[@]}"')
        self.assertLess(dry_guard, dry_return)
        self.assertLess(dry_return, training_pipeline)
        self.assertIn("DRY_RUN=YES (no training process was started)", text)

    def test_failures_continue_and_are_summarized(self):
        text = _script()
        self.assertIn("return_code=$rc", text)
        self.assertIn("last_checkpoint=$last_checkpoint", text)
        self.assertIn('run_training "$label" "FRESH" "$config" "$output_dir" || true', text)
        self.assertIn("===== QUEUE SUMMARY =====", text)
        self.assertIn("FAILED_OR_BLOCKED=$FAILURES", text)
        self.assertIn("train_fdcr_rdcf_spdr_3gpu.log", text)


if __name__ == "__main__":
    unittest.main()
