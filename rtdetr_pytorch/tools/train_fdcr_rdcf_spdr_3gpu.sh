#!/usr/bin/env bash
# Sequential three-GPU queue for one strict resume and four fresh experiments.
#
# Order is intentionally fixed:
#   1) HRNetV2-W18 + SPDR (strict resume only)
#   2) PResNet18 + FDCR   (fresh only)
#   3) HRNetV2-W18 + FDCR (fresh only)
#   4) PResNet18 + RDCF   (fresh only)
#   5) HRNetV2-W18 + RDCF (fresh only)

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash tools/train_fdcr_rdcf_spdr_3gpu.sh [--dry-run]" >&2
    exit 64
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$PROJECT_ROOT/tools/train.py}"
INSPECTOR="${INSPECTOR:-$PROJECT_ROOT/tools/inspect_resume_checkpoint.py}"
CONFIG_ROOT="${CONFIG_ROOT:-$PROJECT_ROOT/configs/rtdetr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/three_gpu_b16_warmup_cosine}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-1,2,3}"
MASTER_PORT="${MASTER_PORT:-9909}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"
NPROC_PER_NODE=3
TARGET_EPOCHS=200
TARGET_LAST_EPOCH=199
LOG_FILE="$OUTPUT_ROOT/train_fdcr_rdcf_spdr_3gpu.log"

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ ${#GPU_LIST[@]} -ne 3 ]]; then
    echo "ERROR: this queue requires exactly three visible GPUs; got CUDA_VISIBLE_DEVICES=$GPU_IDS" >&2
    exit 64
fi
if [[ "${GPU_LIST[0]}" == "${GPU_LIST[1]}" || "${GPU_LIST[0]}" == "${GPU_LIST[2]}" || "${GPU_LIST[1]}" == "${GPU_LIST[2]}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES entries must be distinct; got $GPU_IDS" >&2
    exit 64
fi

SPDR_NAME="rtdetr_hrnetv2_w18_dut_anti_uav_spdr"
SPDR_CONFIG="$CONFIG_ROOT/$SPDR_NAME.yml"
SPDR_OUTPUT="$OUTPUT_ROOT/$SPDR_NAME"

FRESH_NAMES=(
    "rtdetr_r18vd_dut_anti_uav_fdcr"
    "rtdetr_hrnetv2_w18_dut_anti_uav_fdcr"
    "rtdetr_r18vd_dut_anti_uav_rdcf"
    "rtdetr_hrnetv2_w18_dut_anti_uav_rdcf"
)
FRESH_LABELS=(
    "PResNet18+FDCR"
    "HRNetV2-W18+FDCR"
    "PResNet18+RDCF"
    "HRNetV2-W18+RDCF"
)

SUMMARY_LABELS=()
SUMMARY_MODES=()
SUMMARY_STATUSES=()
SUMMARY_RCS=()
SUMMARY_CHECKPOINTS=()
FAILURES=0

emit() {
    printf '%s\n' "$*"
    if [[ $DRY_RUN -eq 0 ]]; then
        printf '%s\n' "$*" >> "$LOG_FILE"
    fi
}

record_summary() {
    SUMMARY_LABELS+=("$1")
    SUMMARY_MODES+=("$2")
    SUMMARY_STATUSES+=("$3")
    SUMMARY_RCS+=("$4")
    SUMMARY_CHECKPOINTS+=("$5")
}

print_command() {
    printf 'COMMAND=OMP_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q ' "$OMP_THREADS" "$GPU_IDS"
    printf '%q ' "$@"
    printf '\n'
}

inspect_resume() {
    local output_dir="$1"
    local inspect_text
    inspect_text="$("$PYTHON_BIN" "$INSPECTOR" "$output_dir" --target-epochs "$TARGET_EPOCHS" 2>&1)"
    INSPECT_RC=$?
    printf '%s\n' "$inspect_text"
    if [[ $DRY_RUN -eq 0 ]]; then
        printf '%s\n' "$inspect_text" >> "$LOG_FILE"
    fi
    RESUME_CHECKPOINT="$(printf '%s\n' "$inspect_text" | sed -n 's/^RESUME_CHECKPOINT=//p' | tail -n 1)"
    RESUME_EPOCH="$(printf '%s\n' "$inspect_text" | sed -n 's/^SAVED_EPOCH=//p' | tail -n 1)"
    RESUME_COMPLETED="$(printf '%s\n' "$inspect_text" | sed -n 's/^TRAINING_COMPLETED=//p' | tail -n 1)"
    RESUME_POSSIBLE="$(printf '%s\n' "$inspect_text" | sed -n 's/^STRICT_RESUME_POSSIBLE=//p' | tail -n 1)"
    [[ -n "$RESUME_CHECKPOINT" ]] || RESUME_CHECKPOINT="NONE"
    [[ -n "$RESUME_EPOCH" ]] || RESUME_EPOCH="NONE"
    [[ -n "$RESUME_COMPLETED" ]] || RESUME_COMPLETED="NO"
    [[ -n "$RESUME_POSSIBLE" ]] || RESUME_POSSIBLE="NO"
}

latest_checkpoint_after_failure() {
    local output_dir="$1"
    local text
    text="$("$PYTHON_BIN" "$INSPECTOR" "$output_dir" --target-epochs "$TARGET_EPOCHS" 2>/dev/null)"
    local rc=$?
    local checkpoint
    checkpoint="$(printf '%s\n' "$text" | sed -n 's/^RESUME_CHECKPOINT=//p' | tail -n 1)"
    if [[ $rc -eq 0 && -n "$checkpoint" ]]; then
        printf '%s' "$checkpoint"
    else
        printf 'NONE'
    fi
}

run_training() {
    local label="$1"
    local mode="$2"
    local config="$3"
    local output_dir="$4"
    local resume_path="${5:-}"
    local start_time end_time rc last_checkpoint
    local cmd=(
        "$TORCHRUN_BIN"
        "--nproc_per_node=$NPROC_PER_NODE"
        "--master_port=$MASTER_PORT"
        "$TRAIN_ENTRY"
        -c "$config"
        --output-dir "$output_dir"
        --amp
        --seed 0
    )
    if [[ "$mode" == "RESUME" ]]; then
        cmd+=( -r "$resume_path" )
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        print_command "${cmd[@]}"
        record_summary "$label" "$mode" "DRY-RUN" "0" "${resume_path:-NONE}"
        return 0
    fi

    if [[ ! -f "$config" ]]; then
        emit "[FAIL] experiment=$label return_code=66 reason=missing_config config=$config last_checkpoint=NONE"
        record_summary "$label" "$mode" "FAIL" "66" "NONE"
        FAILURES=$((FAILURES + 1))
        return 66
    fi

    start_time="$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
    emit "[START] time=$start_time experiment=$label mode=$mode config=$config checkpoint=${resume_path:-NONE}"
    print_command "${cmd[@]}" | tee -a "$LOG_FILE"
    OMP_NUM_THREADS="$OMP_THREADS" CUDA_VISIBLE_DEVICES="$GPU_IDS" "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    end_time="$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
    last_checkpoint="$(latest_checkpoint_after_failure "$output_dir")"

    if [[ $rc -eq 0 ]]; then
        emit "[PASS] end_time=$end_time experiment=$label return_code=0 last_checkpoint=$last_checkpoint"
        record_summary "$label" "$mode" "PASS" "0" "$last_checkpoint"
    else
        emit "[FAIL] end_time=$end_time experiment=$label return_code=$rc last_checkpoint=$last_checkpoint"
        record_summary "$label" "$mode" "FAIL" "$rc" "$last_checkpoint"
        FAILURES=$((FAILURES + 1))
    fi
    return "$rc"
}

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$OUTPUT_ROOT"
    touch "$LOG_FILE"
fi

emit "Total tasks = 5"
emit "GPU protocol: CUDA_VISIBLE_DEVICES=$GPU_IDS, nproc_per_node=$NPROC_PER_NODE"
emit "Training protocol: epochs=$TARGET_EPOCHS, AMP=ON, seed=0"
emit "1. HRNetV2-W18+SPDR [RESUME]"
emit "2. PResNet18+FDCR [FRESH]"
emit "3. HRNetV2-W18+FDCR [FRESH]"
emit "4. PResNet18+RDCF [FRESH]"
emit "5. HRNetV2-W18+RDCF [FRESH]"
emit ""

# Task 1: an existing SPDR directory is never enough to skip.  Completion is
# decided only from a complete checkpoint's saved epoch.
emit "===== TASK 1/5: HRNetV2-W18+SPDR [RESUME] ====="
inspect_resume "$SPDR_OUTPUT"
if [[ "$RESUME_POSSIBLE" != "YES" || "$INSPECT_RC" -ne 0 ]]; then
    emit "[FAIL] experiment=HRNetV2-W18+SPDR return_code=2 reason=STRICT_RESUME_NOT_POSSIBLE last_checkpoint=NONE"
    emit "STRICT RESUME NOT POSSIBLE"
    record_summary "HRNetV2-W18+SPDR" "RESUME" "BLOCKED" "2" "NONE"
    FAILURES=$((FAILURES + 1))
elif [[ "$RESUME_COMPLETED" == "YES" || "$RESUME_EPOCH" -ge "$TARGET_LAST_EPOCH" ]]; then
    emit "[SKIP COMPLETED] experiment=HRNetV2-W18+SPDR checkpoint=$RESUME_CHECKPOINT saved_epoch=$RESUME_EPOCH"
    record_summary "HRNetV2-W18+SPDR" "RESUME" "SKIP COMPLETED" "0" "$RESUME_CHECKPOINT"
else
    emit "[RESUME] experiment=HRNetV2-W18+SPDR checkpoint=$RESUME_CHECKPOINT saved_epoch=$RESUME_EPOCH next_epoch=$((RESUME_EPOCH + 1))"
    run_training "HRNetV2-W18+SPDR" "RESUME" "$SPDR_CONFIG" "$SPDR_OUTPUT" "$RESUME_CHECKPOINT" || true
fi

# Tasks 2-5: directory existence means skip; no historical checkpoint is ever
# supplied to these fresh experiments.
for index in "${!FRESH_NAMES[@]}"; do
    task_number=$((index + 2))
    name="${FRESH_NAMES[$index]}"
    label="${FRESH_LABELS[$index]}"
    config="$CONFIG_ROOT/$name.yml"
    output_dir="$OUTPUT_ROOT/$name"
    emit "===== TASK $task_number/5: $label [FRESH] ====="
    if [[ -d "$output_dir" ]]; then
        emit "[SKIP] experiment=$label reason=output_directory_exists output_dir=$output_dir"
        record_summary "$label" "FRESH" "SKIP EXISTING" "0" "NONE"
        continue
    fi
    emit "[FRESH] experiment=$label config=$config output_dir=$output_dir"
    run_training "$label" "FRESH" "$config" "$output_dir" || true
done

emit "===== QUEUE SUMMARY ====="
for index in "${!SUMMARY_LABELS[@]}"; do
    emit "$((index + 1)). experiment=${SUMMARY_LABELS[$index]} mode=${SUMMARY_MODES[$index]} status=${SUMMARY_STATUSES[$index]} return_code=${SUMMARY_RCS[$index]} checkpoint=${SUMMARY_CHECKPOINTS[$index]}"
done
emit "TOTAL_TASKS=5"
emit "FAILED_OR_BLOCKED=$FAILURES"
if [[ $DRY_RUN -eq 1 ]]; then
    emit "DRY_RUN=YES (no training process was started)"
fi

if [[ $FAILURES -gt 0 ]]; then
    exit 1
fi
exit 0
