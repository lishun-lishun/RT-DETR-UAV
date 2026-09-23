#!/usr/bin/env bash

# Sequentially train every DUT-Anti-UAV RT-DETR-R18 module configuration on
# exactly three GPUs. The first three runs are fixed; all remaining module
# configs are shuffled. A failed run is logged and never stops the queue.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

GPU_IDS="${GPU_IDS:-1,2,3}"
NPROC_PER_NODE=3
BASE_PORT="${BASE_PORT:-9909}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-output/three_gpu_queue_logs/$RUN_TAG}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -ne $NPROC_PER_NODE ]]; then
    echo "ERROR: GPU_IDS must contain exactly three comma-separated GPU IDs; got: $GPU_IDS" >&2
    exit 2
fi

FIXED_CONFIGS=(
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_bpdp.yml"
    "configs/rtdetr/rtdetr_r18vd_dut_anti_uav_msdconv.yml"
)

for config in "${FIXED_CONFIGS[@]}"; do
    if [[ ! -f "$config" ]]; then
        echo "ERROR: required config not found: $config" >&2
        exit 3
    fi
done

# Only direct DUT module YAMLs are included. COCO configs and the uav_tuning
# learning-rate search directory are intentionally excluded.
mapfile -t REMAINING_CONFIGS < <(
    find configs/rtdetr -maxdepth 1 -type f \
        -name 'rtdetr_r18vd_dut_anti_uav_*.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_bpdp.yml' \
        ! -name 'rtdetr_r18vd_dut_anti_uav_msdconv.yml' \
        -print | sort | shuf
)

CONFIGS=("${FIXED_CONFIGS[@]}" "${REMAINING_CONFIGS[@]}")
TOTAL=${#CONFIGS[@]}

mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summary.tsv"
printf 'index\tstatus\texit_code\tconfig\n' > "$SUMMARY_FILE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

trap 'echo "Interrupted: queue stopped by user."; exit 130' INT TERM

echo "Three-GPU DUT module queue"
echo "root=$ROOT_DIR"
echo "physical_gpus=$CUDA_VISIBLE_DEVICES"
echo "world_size=$NPROC_PER_NODE"
echo "seed=$SEED"
echo "runs=$TOTAL"
echo "logs=$LOG_DIR"
echo

PASSED=0
FAILED=0

for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$index]}"
    run_number=$((index + 1))
    port=$((BASE_PORT + index))
    name="$(basename "$config" .yml)"
    log_file="$LOG_DIR/$(printf '%02d' "$run_number")_${name}.log"
    command=(torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$port"
             tools/train.py -c "$config" --amp --seed "$SEED")

    echo "[$run_number/$TOTAL] START $config"
    echo "  port=$port log=$log_file"
    printf '  command: CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "${command[@]}"
    echo

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%s\t%s\t%s\t%s\n' "$run_number" "DRY_RUN" "0" "$config" \
            >> "$SUMMARY_FILE"
        continue
    fi

    "${command[@]}" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        status="PASS"
        PASSED=$((PASSED + 1))
    else
        status="FAIL"
        FAILED=$((FAILED + 1))
    fi
    printf '%s\t%s\t%s\t%s\n' "$run_number" "$status" "$exit_code" "$config" \
        >> "$SUMMARY_FILE"
    echo "[$run_number/$TOTAL] $status (exit=$exit_code): $config"
    echo

    # Deliberately continue after every ordinary training failure.
done

echo "Queue complete: passed=$PASSED failed=$FAILED"
echo "Summary: $SUMMARY_FILE"
exit 0
