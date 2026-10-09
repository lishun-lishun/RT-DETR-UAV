#!/usr/bin/env bash
# Fixed eight-model unified test: two retained baselines plus six candidates.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_gpu_b16_warmup_cosine}"
CONFIG_DIR="${CONFIG_DIR:-configs/rtdetr}"
GPU_ID=1
NUM_WORKERS=2
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash tools/test_pcx_esdr_psca_best.sh [--dry-run]" >&2
    exit 64
fi

EXPERIMENTS=(
    "rtdetr_r18vd_dut_anti_uav"
    "rtdetr_r18vd_dut_anti_uav_pcx"
    "rtdetr_r18vd_dut_anti_uav_esdr"
    "rtdetr_r18vd_dut_anti_uav_psca"
    "rtdetr_hrnetv2_w18_dut_anti_uav"
    "rtdetr_hrnetv2_w18_dut_anti_uav_pcx"
    "rtdetr_hrnetv2_w18_dut_anti_uav_esdr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_psca"
)

if [[ -z "${REPORT_DIR:-}" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    REPORT_DIR="output/three_gpu_b16_warmup_cosine_test_results/pcx_esdr_psca_${stamp}_$$"
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
if [[ $DRY_RUN -eq 1 ]]; then
    command+=(--dry-run)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
printf 'Evaluation command:'
printf ' %q' "${command[@]}"
echo
"${command[@]}"
test_status=$?

if [[ $DRY_RUN -eq 1 ]]; then
    exit "$test_status"
fi

summary_status=0
summary_path="$REPORT_DIR/summary.json"
if [[ -f "$summary_path" ]]; then
    "$PYTHON_BIN" tools/summarize_pcx_esdr_psca.py \
        --summary "$summary_path" --output-dir "$REPORT_DIR" || summary_status=$?
else
    echo "ERROR: evaluation did not create $summary_path" >&2
    summary_status=1
fi

echo "Evaluation exit code: $test_status"
echo "Summary exit code: $summary_status"
echo "Reports: $REPORT_DIR"
if [[ $test_status -ne 0 || $summary_status -ne 0 ]]; then
    exit 1
fi
exit 0
