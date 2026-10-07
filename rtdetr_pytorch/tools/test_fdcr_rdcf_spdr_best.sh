#!/usr/bin/env bash

# Evaluate the final seven-model comparison in a fixed, reviewable order.
# tools/test_all_best.py records each failure and continues with the remaining
# models. tools/test_dut.py enforces original FP32 evaluation; all seven formal
# configs inherit the common 640x640 DUT validation transform.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_gpu_b16_warmup_cosine}"
CONFIG_DIR="${CONFIG_DIR:-configs/rtdetr}"
GPU_ID=1
NUM_WORKERS=2
DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run]"
    echo "Environment overrides: PYTHON_BIN OUTPUT_ROOT CONFIG_DIR REPORT_DIR"
    echo "Fixed protocol: best.pth, test, GPU=1, workers=2, EMA, FP32, 640x640"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

EXPERIMENTS=(
    "rtdetr_r18vd_dut_anti_uav"
    "rtdetr_r18vd_dut_anti_uav_fdcr"
    "rtdetr_r18vd_dut_anti_uav_rdcf"
    "rtdetr_hrnetv2_w18_dut_anti_uav"
    "rtdetr_hrnetv2_w18_dut_anti_uav_fdcr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_rdcf"
    "rtdetr_hrnetv2_w18_dut_anti_uav_spdr"
)

if [[ -z "${REPORT_DIR:-}" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    REPORT_DIR="output/three_gpu_b16_warmup_cosine_test_results/fdcr_rdcf_spdr_${stamp}_$$"
fi

test_command=(
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
if [[ "$DRY_RUN" == "1" ]]; then
    # A dry-run creates neither REPORT_DIR nor summary.json.
    test_command+=(--dry-run)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

printf 'Evaluation command:'
printf ' %q' "${test_command[@]}"
echo

# Do not use `set -e`: test_all_best must finish its complete fixed list even
# when one checkpoint is missing or one subprocess fails.
"${test_command[@]}"
test_status=$?

if [[ "$DRY_RUN" == "1" ]]; then
    exit "$test_status"
fi

summary_path="$REPORT_DIR/summary.json"
summary_status=0
if [[ -f "$summary_path" ]]; then
    summary_command=(
        "$PYTHON_BIN" tools/summarize_fdcr_rdcf_spdr.py
        --summary "$summary_path"
        --output-dir "$REPORT_DIR"
    )
    printf 'Summary command:'
    printf ' %q' "${summary_command[@]}"
    echo
    "${summary_command[@]}" || summary_status=$?
else
    echo "ERROR: evaluation did not create $summary_path" >&2
    summary_status=1
fi

echo "Evaluation exit code: $test_status"
echo "Summary exit code: $summary_status"
echo "Reports: $REPORT_DIR"
if [[ "$test_status" -ne 0 || "$summary_status" -ne 0 ]]; then
    exit 1
fi
exit 0
