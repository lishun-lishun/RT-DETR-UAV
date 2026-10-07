#!/usr/bin/env bash

# Evaluate the four independent LPRU/SPDR best checkpoints in the fixed
# experiment order. Each evaluation uses the real DUT test split, 640x640,
# EMA weights, FP32 and a single GPU. Failures are recorded and do not stop
# later experiments; tools/test_all_best.py returns non-zero at the end if any
# checkpoint is missing or any evaluation fails.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_gpu_b16_warmup_cosine}"
CONFIG_DIR="${CONFIG_DIR:-configs/rtdetr}"
DRY_RUN=0

EXPERIMENTS=(
    "rtdetr_r18vd_dut_anti_uav_lpru"
    "rtdetr_hrnetv2_w18_dut_anti_uav_lpru"
    "rtdetr_r18vd_dut_anti_uav_spdr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_spdr"
)

usage() {
    echo "Usage: $0 [--dry-run]"
    echo "Environment overrides: PYTHON_BIN GPU_ID NUM_WORKERS OUTPUT_ROOT CONFIG_DIR REPORT_DIR"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU_ID must be one non-negative physical GPU index; got: $GPU_ID" >&2
    exit 2
fi
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_WORKERS must be a non-negative integer; got: $NUM_WORKERS" >&2
    exit 2
fi

command=(
    "$PYTHON_BIN" tools/test_all_best.py
    --root "$OUTPUT_ROOT"
    --config-dir "$CONFIG_DIR"
    --split test
    --gpu "$GPU_ID"
    --num-workers "$NUM_WORKERS"
    --require-ema
    --experiments "${EXPERIMENTS[@]}"
)
if [[ -n "${REPORT_DIR:-}" ]]; then
    command+=(--report-dir "$REPORT_DIR")
fi
if [[ "$DRY_RUN" == "1" ]]; then
    command+=(--dry-run)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

echo "Fixed evaluation order (${#EXPERIMENTS[@]} experiments):"
for index in "${!EXPERIMENTS[@]}"; do
    printf '%d. %s\n' "$((index + 1))" "${EXPERIMENTS[$index]}"
done
printf 'Command:'
printf ' %q' "${command[@]}"
echo

exec "${command[@]}"
