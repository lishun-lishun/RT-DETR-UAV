"""Contract tests for the fixed six-experiment PCX/ESDR/PSCA queue."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / 'tools' / 'train_pcx_esdr_psca_3gpu.sh'

EXPECTED_NAMES = (
    'rtdetr_r18vd_dut_anti_uav_pcx',
    'rtdetr_hrnetv2_w18_dut_anti_uav_pcx',
    'rtdetr_r18vd_dut_anti_uav_esdr',
    'rtdetr_hrnetv2_w18_dut_anti_uav_esdr',
    'rtdetr_r18vd_dut_anti_uav_psca',
    'rtdetr_hrnetv2_w18_dut_anti_uav_psca',
)
EXPECTED_LABELS = (
    'PResNet18+PCX',
    'HRNetV2-W18+PCX',
    'PResNet18+ESDR',
    'HRNetV2-W18+ESDR',
    'PResNet18+PSCA',
    'HRNetV2-W18+PSCA',
)


def _script():
    return SCRIPT_PATH.read_text(encoding='utf-8')


def _bash_array(text, name):
    match = re.search(
        rf'(?ms)^{re.escape(name)}=\(\s*(.*?)^\)', text)
    if match is None:
        raise AssertionError(f'Bash array {name} was not found')
    return tuple(re.findall(r'^\s*"([^"]+)"\s*$', match.group(1), re.M))


class TrainPCXESDRPSCAQueueContractTests(unittest.TestCase):
    def test_queue_has_exact_six_experiment_order(self):
        text = _script()
        self.assertEqual(_bash_array(text, 'NAMES'), EXPECTED_NAMES)
        self.assertEqual(_bash_array(text, 'LABELS'), EXPECTED_LABELS)
        self.assertIn('emit "Total experiments = 6"', text)
        self.assertIn('emit "TOTAL_EXPERIMENTS=6"', text)
        self.assertIn('EXPERIMENT $number/6', text)

    def test_every_experiment_is_fresh_and_resume_is_impossible(self):
        text = _script()
        self.assertIn('Mode = FRESH ONLY', text)
        self.assertIn('[FRESH]', text)
        self.assertIn('mode=FRESH', text)
        self.assertNotIn('RESUME', text)
        self.assertNotIn('resume', text.lower())
        command = re.search(r'(?ms)^\s*command=\(\s*(.*?)^\s*\)', text)
        self.assertIsNotNone(command)
        self.assertIsNone(re.search(
            r'^\s*-r(?:\s|$)', command.group(1), re.M))
        self.assertNotIn('checkpoint.pth', text)
        self.assertNotIn('best.pth', text)

    def test_three_distinct_gpu_contract_is_enforced(self):
        text = _script()
        self.assertIn('GPU_IDS="${CUDA_VISIBLE_DEVICES:-1,2,3}"', text)
        self.assertIn('NPROC_PER_NODE=3', text)
        self.assertIn('"--nproc_per_node=$NPROC_PER_NODE"', text)
        self.assertIn('exactly three visible GPUs are required', text)
        self.assertIn('CUDA_VISIBLE_DEVICES entries must be distinct', text)
        self.assertIn('"${GPU_LIST[0]}" == "${GPU_LIST[1]}"', text)
        self.assertIn('"${GPU_LIST[0]}" == "${GPU_LIST[2]}"', text)
        self.assertIn('"${GPU_LIST[1]}" == "${GPU_LIST[2]}"', text)

    def test_amp_seed_and_output_override_are_fixed(self):
        text = _script()
        self.assertIn('OMP_THREADS="${OMP_NUM_THREADS:-1}"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU_IDS"', text)
        self.assertIn('--output-dir "$output_dir"', text)
        self.assertIn('--amp', text)
        self.assertIn('--seed 0', text)
        self.assertIn('--master_port=$MASTER_PORT', text)
        self.assertIn('AMP=ON, seed=0', text)

    def test_existing_final_output_directory_is_skipped(self):
        text = _script()
        directory_guard = text.index('if [[ -d "$output_dir" ]]')
        skip_marker = text.index(
            'reason=output_directory_exists', directory_guard)
        loop_end = text.index('\ndone', directory_guard)
        self.assertLess(directory_guard, skip_marker)
        self.assertLess(skip_marker, loop_end)
        self.assertIn('SUMMARY+=("$number|$label|SKIP EXISTING|0")', text)
        self.assertIn('continue', text[directory_guard:loop_end])

    def test_failure_continues_queue_and_final_status_is_nonzero(self):
        text = _script()
        result_branch = text.index('if [[ $rc -eq 0 ]]')
        loop_end = text.index('\ndone', result_branch)
        branch = text[result_branch:loop_end]
        self.assertIn('PIPESTATUS[0]', text)
        self.assertIn('continuing_to_next=YES', branch)
        self.assertIn('FAILURES=$((FAILURES + 1))', branch)
        self.assertNotIn('exit ', branch)

        summary = text[loop_end:]
        self.assertIn('===== QUEUE SUMMARY =====', summary)
        self.assertIn('emit "FAILURES=$FAILURES"', summary)
        self.assertRegex(
            summary,
            r'if \[\[ \$FAILURES -gt 0 \]\]; then\s+exit 1\s+fi\s+exit 0')

    def test_missing_config_is_recorded_and_does_not_abort_loop(self):
        text = _script()
        guard = text.index('if [[ ! -f "$config" ]]')
        loop_end = text.index('\ndone', guard)
        block = text[guard:loop_end]
        self.assertIn('return_code=66 reason=missing_config', block)
        self.assertIn('FAILURES=$((FAILURES + 1))', block)
        self.assertIn('continue', block)
        self.assertNotIn('exit 66', block)

    def test_dry_run_neither_creates_output_nor_starts_training(self):
        text = _script()
        create_guard = text.index('if [[ $DRY_RUN -eq 0 ]]')
        make_directory = text.index('mkdir -p "$OUTPUT_ROOT"', create_guard)
        create_guard_end = text.index('\nfi', make_directory)
        self.assertLess(create_guard, make_directory)
        self.assertLess(make_directory, create_guard_end)

        dry_guard = text.index('if [[ $DRY_RUN -eq 1 ]]', create_guard_end)
        dry_continue = text.index('continue', dry_guard)
        training = text.index(
            'OMP_NUM_THREADS="$OMP_THREADS" CUDA_VISIBLE_DEVICES="$GPU_IDS"',
            dry_continue)
        self.assertLess(dry_guard, dry_continue)
        self.assertLess(dry_continue, training)
        self.assertIn(
            'DRY_RUN=YES (no output directory or training process was created)',
            text)


if __name__ == '__main__':
    unittest.main()
