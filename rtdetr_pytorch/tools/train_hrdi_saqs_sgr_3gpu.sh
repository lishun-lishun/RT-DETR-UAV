#!/usr/bin/env bash
# Fixed HRNetV2-W18 ablation queue for HRDI / SAQS / SGR.

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash tools/train_hrdi_saqs_sgr_3gpu.sh [--dry-run]" >&2
    exit 64
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$PROJECT_ROOT/tools/train.py}"
CONFIG_ROOT="${CONFIG_ROOT:-$PROJECT_ROOT/configs/rtdetr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/three_gpu_b16_warmup_cosine}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-1,2,3}"
MASTER_PORT="${MASTER_PORT:-9909}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"
NPROC_PER_NODE=3
LOG_FILE="$OUTPUT_ROOT/train_hrdi_saqs_sgr_3gpu.log"

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ ${#GPU_LIST[@]} -ne 3 ]]; then
    echo "ERROR: exactly three visible GPUs are required; got CUDA_VISIBLE_DEVICES=$GPU_IDS" >&2
    exit 64
fi
if [[ "${GPU_LIST[0]}" == "${GPU_LIST[1]}" || \
      "${GPU_LIST[0]}" == "${GPU_LIST[2]}" || \
      "${GPU_LIST[1]}" == "${GPU_LIST[2]}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES entries must be distinct; got $GPU_IDS" >&2
    exit 64
fi

# Keep the exact user-requested priority: baseline reference, three singles,
# then pairs, and finally the full combination.
NAMES=(
    "rtdetr_hrnetv2_w18_dut_anti_uav"
    "rtdetr_hrnetv2_w18_dut_anti_uav_sgr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi"
    "rtdetr_hrnetv2_w18_dut_anti_uav_saqs"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_sgr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_saqs_sgr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_saqs"
    "rtdetr_hrnetv2_w18_dut_anti_uav_hrdi_saqs_sgr"
)

FAILURES=0
SUMMARY=()

emit() {
    printf '%s\n' "$*"
    if [[ $DRY_RUN -eq 0 ]]; then
        printf '%s\n' "$*" >> "$LOG_FILE"
    fi
}

print_command() {
    printf 'COMMAND=OMP_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q ' "$OMP_THREADS" "$GPU_IDS"
    printf '%q ' "$@"
    printf '\n'
}

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$OUTPUT_ROOT"
    touch "$LOG_FILE"
fi

emit "Total experiments = ${#NAMES[@]}"
emit "Mode = FRESH ONLY; existing output directories are skipped"
emit "GPU protocol = CUDA_VISIBLE_DEVICES=$GPU_IDS, nproc_per_node=3"
emit "Training protocol = epochs=200, batch_size=16/GPU, global_batch=48, AMP=ON, seed=0"
emit "Learning rates = main 3e-4, backbone 3e-5; warmup=5, cosine=ON"

for index in "${!NAMES[@]}"; do
    number=$((index + 1))
    name="${NAMES[$index]}"
    config="$CONFIG_ROOT/$name.yml"
    output_dir="$OUTPUT_ROOT/$name"
    command=(
        "$TORCHRUN_BIN"
        "--nproc_per_node=$NPROC_PER_NODE"
        "--master_port=$MASTER_PORT"
        "$TRAIN_ENTRY"
        -c "$config"
        --output-dir "$output_dir"
        --amp
        --seed 0
    )

    emit "===== EXPERIMENT $number/${#NAMES[@]}: $name [FRESH] ====="
    emit "config=$config"
    emit "output_dir=$output_dir"

    if [[ -d "$output_dir" ]]; then
        emit "[SKIP] experiment=$name reason=output_directory_exists"
        SUMMARY+=("$number|$name|SKIP EXISTING|0")
        continue
    fi
    if [[ ! -f "$config" ]]; then
        emit "[FAIL] experiment=$name return_code=66 reason=missing_config"
        SUMMARY+=("$number|$name|FAIL|66")
        FAILURES=$((FAILURES + 1))
        continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        print_command "${command[@]}"
        SUMMARY+=("$number|$name|DRY-RUN|0")
        continue
    fi

    print_command "${command[@]}" | tee -a "$LOG_FILE"
    OMP_NUM_THREADS="$OMP_THREADS" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
        emit "[PASS] experiment=$name return_code=0"
        SUMMARY+=("$number|$name|PASS|0")
    else
        emit "[FAIL] experiment=$name return_code=$rc; continuing_to_next=YES"
        SUMMARY+=("$number|$name|FAIL|$rc")
        FAILURES=$((FAILURES + 1))
    fi
done

emit "===== QUEUE SUMMARY ====="
for row in "${SUMMARY[@]}"; do emit "$row"; done
emit "TOTAL_EXPERIMENTS=${#NAMES[@]}"
emit "FAILURES=$FAILURES"
if [[ $DRY_RUN -eq 1 ]]; then
    emit "DRY_RUN=YES (no output directory or process was created)"
fi
[[ $FAILURES -eq 0 ]]
