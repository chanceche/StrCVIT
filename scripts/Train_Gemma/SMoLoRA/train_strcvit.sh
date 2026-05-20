#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_HOME="<CUDA_HOME>"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

MODEL_PATH="<MODEL_ROOT>/gemma-3-4b-pt"
STRLORA_SAMPLE_ROOT="${STRLORA_SAMPLE_ROOT:-<STRCVIT_DATASET_DIR>}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${STRLORA_SAMPLE_ROOT}/train}"
DATA_ROOT="${DATA_ROOT:-<STRCVIT_METHODS_DIR>/StrCVIT_runtime/SMoLoRA_indexed}"
CKPT_ROOT="<STRCVIT_METHODS_DIR>/checkpoints/StrCVIT"
INS_EMB_ROOT="${INS_EMB_ROOT:-<STRCVIT_METHODS_DIR>/scripts/Train_intervl/internvl3_5_8b_SMoLoRA/embeddings}"

prepare_smolora_dataset() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    python - "$src" "$dst" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r", encoding="utf-8") as f:
    data = json.load(f)
for idx, item in enumerate(data):
    item["smolora_ins_idx"] = idx
with dst.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

if [ -z "${1:-}" ]; then
    echo "Usage: bash train_strcvit.sh data_002"
    exit 1
fi

task_id=$1
data_file="${task_id}.json"
idx_num=$(echo "$task_id" | grep -oE '[0-9]+')
prev_idx=$((10#$idx_num - 1))
prev_num=$(printf "%03d" "$prev_idx")
ins_type=$((10#$idx_num - 1))
ins_emb_path="${INS_EMB_ROOT}/${task_id}.pkl"
source_data_path="${SOURCE_DATA_ROOT}/${data_file}"
data_path="${DATA_ROOT}/${data_file}"
output_dir="${CKPT_ROOT}/${task_id}/gemma3_4b_SMoLoRA"
pre_dir="${CKPT_ROOT}/data_${prev_num}/gemma3_4b_SMoLoRA"

echo "------------------------------------------------"
echo "Current Task: $task_id"
echo "Dataset File: $data_file"
echo "Output Dir:   $output_dir"
echo "Ins Type:     $ins_type"
echo "Ins Emb:      $ins_emb_path"

prepare_smolora_dataset "$source_data_path" "$data_path"

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
MAX_PIXELS=802816 \
MASTER_PORT=29665 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$data_path" \
    --previous_task_model_path "$pre_dir" \
    --load_from_cache_file true \
    --lr_scheduler_type constant \
    --train_type smolora \
    --expert_num 4 \
    --ins_type "$ins_type" \
    --ins_emb "$ins_emb_path" \
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
    --report_to none \
    --max_length 4096 \
    --output_dir "$output_dir" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --save_only_model true \
    --add_version false \
    --eval_strategy no \
    --save_strategy no
