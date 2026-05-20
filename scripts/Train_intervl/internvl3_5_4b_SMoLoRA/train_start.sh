#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_HOME="<CUDA_HOME>"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

STRLORA_SAMPLE_ROOT="${STRLORA_SAMPLE_ROOT:-<STRCVIT_DATASET_DIR>}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${STRLORA_SAMPLE_ROOT}/train}"
DATA_ROOT="${DATA_ROOT:-<STRCVIT_METHODS_DIR>/StrCVIT_runtime/SMoLoRA_indexed}"
INS_EMB_ROOT="${INS_EMB_ROOT:-<STRCVIT_METHODS_DIR>/scripts/Train_intervl/internvl3_5_8b_SMoLoRA/embeddings}"
INS_EMB_PATH="${INS_EMB_PATH:-${INS_EMB_ROOT}/data_001.pkl}"
SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-${SOURCE_DATA_ROOT}/data_001.json}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/data_001.json}"

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

prepare_smolora_dataset "$SOURCE_DATA_PATH" "$DATA_PATH"

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
MASTER_PORT=29678 \
MAX_PIXELS=802816 \
swift sft \
    --model <MODEL_ROOT>/InternVL3_5-4B-Pretrained \
    --dataset "$DATA_PATH" \
    --load_from_cache_file true \
    --train_type smolora \
    --expert_num 4 \
    --ins_type 0 \
    --ins_emb "$INS_EMB_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --train_aligner true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --report_to none \
    --max_length 4096 \
    --output_dir <STRCVIT_METHODS_DIR>/checkpoints/StrCVIT/data_001/internvl3_5_4b_SMoLoRA \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --save_only_model true \
    --add_version false \
    --eval_strategy no \
    --save_strategy no
