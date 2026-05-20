#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_HOME="<CUDA_HOME>"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"


NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
MASTER_PORT=29678 \
MAX_PIXELS=802816 \
swift sft \
    --model <MODEL_ROOT>/gemma-3-4b-pt \
    --dataset <STRCVIT_DATASET_DIR>/train/data_001.json \
    --load_from_cache_file true \
    --train_type moelora \
    --expert_num 4 \
    --train_aligner true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir <STRCVIT_METHODS_DIR>/checkpoints/StrCVIT/data_001/gemma3_4b_MoELoRA \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --save_only_model true \
    --add_version false \
    --eval_strategy no \
    --save_strategy no
