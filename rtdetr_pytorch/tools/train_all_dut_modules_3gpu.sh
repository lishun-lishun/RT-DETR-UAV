#!/usr/bin/env bash

# Sequentially train every DUT-Anti-UAV RT-DETR module configuration on exactly
# three GPUs. HRNetV2-W18 is first; the previous fixed order and randomized
# remainder policy are retained. An existing final experiment output path is
# skipped exactly as requested. Any ordinary training failure stops the queue.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

GPU_IDS="${GPU_IDS:-1,2,3}"
NPROC_PER_NODE=3
BASE_PORT="${BASE_PORT:-9909}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/three_gpu_b16_warmup_cosine}"

usage() {
    echo "Usage: $0 [--dry-run]"
    echo "  --dry-run  print the complete RUN/SKIP plan only; create nothing"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -ne $NPROC_PER_NODE ]]; then
    echo "ERROR: GPU_IDS must contain exactly three comma-separated GPU IDs; got: $GPU_IDS" >&2
    exit 2
fi

HRNET_CONFIG="configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml"
FIXED_CONFIGS=(
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr3.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr34.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_pdr34_nogate.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml"
)

for config in "$HRNET_CONFIG" "${FIXED_CONFIGS[@]}"; do
    if [[ ! -f "$config" ]]; then
        echo "ERROR: required config not found: $config" >&2
        exit 3
    fi
done

# Only direct DUT module YAMLs are included. COCO configs and uav_tuning are
# intentionally excluded. The previous queue shuffled this tail; keep that
# behavior so HRNet insertion does not alter the established scheduling policy.
mapfile -t REMAINING_CONFIGS < <(
    find configs/rtdetr -maxdepth 1 -type f \
        -name 'rtdetr_r18vd_dut_anti_uav_*.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_pdr3.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_pdr34.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_pdr34_nogate.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_bpdp.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_msdconv.yml' \
        -print | sort | shuf
)

CONFIGS=("$HRNET_CONFIG" "${FIXED_CONFIGS[@]}" "${REMAINING_CONFIGS[@]}")
TOTAL=${#CONFIGS[@]}
declare -a OUTPUT_DIRS PLAN_STATUS
RUN_COUNT=0
SKIP_COUNT=0

# Resolve inherited YAML using the project parser. The queue passes an output
# override, so that final CLI path is the exact path tested for existence.
for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$index]}"
    name="$(basename "$config" .yml)"
    cli_output="$OUTPUT_ROOT/$name"
    if ! output_dir="$("$PYTHON_BIN" tools/resolve_config_output.py \
            "$config" --output-dir "$cli_output")"; then
        echo "ERROR: failed to resolve output_dir for $config" >&2
        exit 4
    fi
    OUTPUT_DIRS[$index]="$output_dir"
    if [[ -e "$output_dir" ]]; then
        PLAN_STATUS[$index]="SKIP"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    else
        PLAN_STATUS[$index]="RUN"
        RUN_COUNT=$((RUN_COUNT + 1))
    fi
done

echo "=================================================="
echo "Three-GPU DUT experiment plan"
echo "Root: $ROOT_DIR"
echo "CUDA devices: $GPU_IDS"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "Total configs: $TOTAL"
echo "Will run: $RUN_COUNT"
echo "Will skip: $SKIP_COUNT"
echo "=================================================="
for index in "${!CONFIGS[@]}"; do
    number=$((index + 1))
    printf '%2d. [%s] %s\n' "$number" "${PLAN_STATUS[$index]}" "${CONFIGS[$index]}"
    printf '    Output: %s\n' "${OUTPUT_DIRS[$index]}"
done
echo "=================================================="

# Stop before date generation, mkdir, log creation, exports or torchrun.
if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN complete: no directory was created and no training was launched."
    exit 0
fi

RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-output/three_gpu_queue_logs/$RUN_TAG}"
mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summary.tsv"
printf 'index\tstatus\texit_code\tconfig\toutput\n' > "$SUMMARY_FILE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
trap 'echo "Interrupted: queue stopped by user."; exit 130' INT TERM

PASSED=0
for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$index]}"
    run_number=$((index + 1))
    port=$((BASE_PORT + index))
    name="$(basename "$config" .yml)"
    output_dir="${OUTPUT_DIRS[$index]}"

    if [[ "${PLAN_STATUS[$index]}" == "SKIP" ]]; then
        echo "=================================================="
        echo "[SKIP $run_number/$TOTAL]"
        echo "Config: $config"
        echo "Output: $output_dir"
        echo "Reason: output directory already exists"
        echo "=================================================="
        continue
    fi

    log_file="$LOG_DIR/$(printf '%02d' "$run_number")_${name}.log"
    command=(torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$port"
             tools/train.py -c "$config" --output-dir "$output_dir"
             --amp --seed "$SEED")

    echo "=================================================="
    echo "[RUN $run_number/$TOTAL]"
    echo "Config: $config"
    echo "Output: $output_dir"
    echo "GPUs: $CUDA_VISIBLE_DEVICES (3 ranks)"
    echo "Log: $log_file"
    printf 'Command: CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "${command[@]}"
    echo
    echo "=================================================="

    "${command[@]}" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$run_number" "FAIL" \
            "$exit_code" "$config" "$output_dir" >> "$SUMMARY_FILE"
        echo "[FAIL $run_number/$TOTAL] exit=$exit_code; stopping queue: $config" >&2
        exit "$exit_code"
    fi

    PASSED=$((PASSED + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' "$run_number" "PASS" "0" \
        "$config" "$output_dir" >> "$SUMMARY_FILE"
    echo "[PASS $run_number/$TOTAL] $config"
done

echo "Queue complete: passed=$PASSED skipped=$SKIP_COUNT"
echo "Summary: $SUMMARY_FILE"
