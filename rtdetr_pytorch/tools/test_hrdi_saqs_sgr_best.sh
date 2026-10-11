#!/usr/bin/env bash
# Evaluate the fixed HRNet HRDI/SAQS/SGR ablation matrix on the true test split.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_gpu_b16_warmup_cosine}"
CONFIG_DIR="${CONFIG_DIR:-configs/rtdetr}"
GPU_ID="${GPU_ID:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash tools/test_hrdi_saqs_sgr_best.sh [--dry-run]" >&2
    exit 64
fi

EXPERIMENTS=(
    "rtdetr_hrnetv2_w18_dut_anti_uav"
    "rtdetr_hrnetv2_w18_dut_anti_uav_sgr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi"
    "rtdetr_hrnetv2_w18_dut_anti_uav_saqs"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_sgr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_saqs_sgr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_saqs"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_saqs_sgr"
)

if [[ -z "${REPORT_DIR:-}" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    REPORT_DIR="output/three_gpu_b16_warmup_cosine_test_results/hrdi_saqs_sgr_${stamp}_$$"
fi

command=(
    "$PYTHON_BIN" tools/test_all_best.py
    --root "$OUTPUT_ROOT"
    --config-dir "$CONFIG_DIR"
    --split test
    --gpu "$GPU_ID"
    --num-workers "$NUM_WORKERS"
    --report-dir "$REPORT_DIR"
    --require-ema
    --experiments "${EXPERIMENTS[@]}"
)
if [[ $DRY_RUN -eq 1 ]]; then command+=(--dry-run); fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
printf 'Evaluation command:'
printf ' %q' "${command[@]}"
echo
"${command[@]}"
test_status=$?

if [[ $DRY_RUN -eq 1 ]]; then exit "$test_status"; fi

summary_status=0
summary_path="$REPORT_DIR/summary.json"
if [[ -f "$summary_path" ]]; then
    "$PYTHON_BIN" tools/summarize_hrdi_saqs_sgr.py \
        --summary "$summary_path" --output-dir "$REPORT_DIR" || summary_status=$?
else
    echo "ERROR: evaluation did not create $summary_path" >&2
    summary_status=1
fi

echo "Evaluation exit code: $test_status"
echo "Summary exit code: $summary_status"
echo "Reports: $REPORT_DIR"
if [[ $test_status -ne 0 || $summary_status -ne 0 ]]; then exit 1; fi
