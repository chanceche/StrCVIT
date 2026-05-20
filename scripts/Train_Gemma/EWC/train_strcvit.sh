#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_HOME="<CUDA_HOME>"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"


MODEL_PATH="<MODEL_ROOT>/gemma-3-4b-pt"
DATA_ROOT="<STRCVIT_DATASET_DIR>/train"
CKPT_ROOT="<STRCVIT_METHODS_DIR>/checkpoints/StrCVIT"

if [ -z "${1:-}" ]; then
    echo "Usage: bash train_strcvit.sh data_002"
    exit 1
fi

task_id=$1
data_file="${task_id}.json"
idx_num=$(echo "$task_id" | grep -oE '[0-9]+')
prev_idx=$((10#$idx_num - 1))
prev_num=$(printf "%03d" "$prev_idx")

output_dir="${CKPT_ROOT}/${task_id}/gemma3_4b_EWC"
pre_dir="${CKPT_ROOT}/data_${prev_num}/gemma3_4b_EWC"

echo "------------------------------------------------"
echo "Current Task: $task_id"
echo "Dataset File: $data_file"
echo "Output Dir:   $output_dir"

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
MAX_PIXELS=802816 \
MASTER_PORT=29665 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATA_ROOT/$data_file" \
    --previous_task_model_path "$pre_dir" \
    --load_from_cache_file true \
    --lr_scheduler_type constant \
    --train_type lora \
    --ewc_lambda 1.0 \
    --train_aligner true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-05 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir "$output_dir" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true \
    --add_version false \
    --eval_strategy no \
    --save_strategy no
