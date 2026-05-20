#!/bin/bash

AVAILABLE_GPUS=(0 1 2 3)
GPUS_PER_TASK=1
TASKS_PER_GROUP=2
BASE_SCRIPT_DIR="<STRCVIT_METHODS_DIR>/scripts/Eval_Gemma_proxy/SMoLoRA"

if [ -z "${1:-}" ]; then
  echo "Usage: bash eval_gemma3_4b_SMoLoRA.sh <task_id>"
  exit 1
fi

MODEL_DIR="$1/gemma3_4b_SMoLoRA"
RESULT_DIR="<STRCVIT_METHODS_DIR>/results/StrCVIT/$MODEL_DIR"
MODEL_BASE="gemma-3-4b-pt"

export BASE_SCRIPT_DIR
export MODEL_DIR
export MODEL_BASE
export RESULT_DIR

DATASETS=(
    textcaps
    ad
    grounding
    rs
    imagenet
    ocrvqa
    gqa
    vqav2
    places365
    chartqa
    fin
)

GPU_GROUPS=()
total_gpus=${#AVAILABLE_GPUS[@]}
for ((i=0; i<total_gpus; i+=GPUS_PER_TASK)); do
    group_str=""
    for ((j=0; j<GPUS_PER_TASK; j++)); do
        idx=$((i + j))
        if [ "$idx" -lt "$total_gpus" ]; then
            if [ -n "$group_str" ]; then group_str="$group_str,"; fi
            group_str="$group_str${AVAILABLE_GPUS[$idx]}"
        fi
    done
    GPU_GROUPS+=("$group_str")
done

echo "=== Evaluation Configuration ==="
echo "Model dir: $MODEL_DIR"
echo "Model base: $MODEL_BASE"
echo "GPU groups: ${GPU_GROUPS[*]}"
echo "================================"

INDEX_FILE=$(mktemp)
LOCK_FILE=$(mktemp)
echo "0" > "$INDEX_FILE"

cleanup() {
    rm -f "$INDEX_FILE" "$LOCK_FILE"
}
trap cleanup EXIT

run_worker() {
    local my_gpus=$1
    local worker_name=$2
    exec 200>"$LOCK_FILE"

    while true; do
        flock -x 200
        local current_idx
        current_idx=$(cat "$INDEX_FILE")
        local total_tasks=${#DATASETS[@]}

        if [ "$current_idx" -ge "$total_tasks" ]; then
            flock -u 200
            break
        fi

        local dataset_key="${DATASETS[$current_idx]}"
        echo $((current_idx + 1)) > "$INDEX_FILE"
        flock -u 200

        echo "[$worker_name] Running $dataset_key on GPUs $my_gpus"
        export USE_GPUS="$my_gpus"
        export CUDA_VISIBLE_DEVICES="$my_gpus"
        bash "$BASE_SCRIPT_DIR/run_dataset.sh" "$MODEL_DIR" "$MODEL_BASE" "$dataset_key"
    done

    exec 200>&-
}

PIDS=()
for gpu_group in "${GPU_GROUPS[@]}"; do
    for ((i=1; i<=TASKS_PER_GROUP; i++)); do
        safe_name=${gpu_group//,/_}
        run_worker "$gpu_group" "Worker-GPUs${safe_name}-T${i}" &
        PIDS+=($!)
        sleep 2
    done
done

wait "${PIDS[@]}"

python -m swift.llm.eval.new_calculate_ap_proxy --result-dir "$RESULT_DIR"
