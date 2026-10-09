#!/usr/bin/env bash
# Six independent, fresh PCX/ESDR/PSCA experiments in a fixed order.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash tools/train_pcx_esdr_psca_3gpu.sh [--dry-run]" >&2
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
LOG_FILE="$OUTPUT_ROOT/train_pcx_esdr_psca_3gpu.log"

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

NAMES=(
    "rtdetr_r18vd_dut_anti_uav_pcx"
    "rtdetr_hrnetv2_w18_dut_anti_uav_pcx"
    "rtdetr_r18vd_dut_anti_uav_esdr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_esdr"
    "rtdetr_r18vd_dut_anti_uav_psca"
    "rtdetr_hrnetv2_w18_dut_anti_uav_psca"
)
LABELS=(
    "PResNet18+PCX"
    "HRNetV2-W18+PCX"
    "PResNet18+ESDR"
    "HRNetV2-W18+ESDR"
    "PResNet18+PSCA"
    "HRNetV2-W18+PSCA"
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

emit "Total experiments = 6"
emit "Mode = FRESH ONLY"
emit "GPU protocol = CUDA_VISIBLE_DEVICES=$GPU_IDS, nproc_per_node=3"
emit "Training protocol = epochs=200, batch_size=16/GPU, global_batch=48, AMP=ON, seed=0"
emit "Learning rates = main 3e-4, backbone 3e-5; warmup=5, cosine=ON"
for index in "${!NAMES[@]}"; do
    number=$((index + 1))
    name="${NAMES[$index]}"
    label="${LABELS[$index]}"
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

    emit "===== EXPERIMENT $number/6: $label [FRESH] ====="
    emit "config=$config"
    emit "output_dir=$output_dir"

    if [[ -d "$output_dir" ]]; then
        emit "[SKIP] experiment=$label reason=output_directory_exists"
        SUMMARY+=("$number|$label|SKIP EXISTING|0")
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        print_command "${command[@]}"
        SUMMARY+=("$number|$label|DRY-RUN|0")
        continue
    fi

    if [[ ! -f "$config" ]]; then
        emit "[FAIL] experiment=$label return_code=66 reason=missing_config"
        SUMMARY+=("$number|$label|FAIL|66")
        FAILURES=$((FAILURES + 1))
        continue
    fi

    emit "[START] experiment=$label mode=FRESH"
    print_command "${command[@]}" | tee -a "$LOG_FILE"
    OMP_NUM_THREADS="$OMP_THREADS" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
        emit "[PASS] experiment=$label return_code=0"
        SUMMARY+=("$number|$label|PASS|0")
    else
        emit "[FAIL] experiment=$label return_code=$rc; continuing_to_next=YES"
        SUMMARY+=("$number|$label|FAIL|$rc")
        FAILURES=$((FAILURES + 1))
    fi
done

emit "===== QUEUE SUMMARY ====="
for row in "${SUMMARY[@]}"; do
    emit "$row"
done
emit "TOTAL_EXPERIMENTS=6"
emit "FAILURES=$FAILURES"
if [[ $DRY_RUN -eq 1 ]]; then
    emit "DRY_RUN=YES (no output directory or training process was created)"
fi

if [[ $FAILURES -gt 0 ]]; then
    exit 1
fi
exit 0
