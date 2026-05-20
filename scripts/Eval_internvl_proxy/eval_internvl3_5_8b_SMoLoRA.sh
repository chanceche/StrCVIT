#!/bin/bash






# AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)


GPUS_PER_TASK=4


TASKS_PER_GROUP=2

# ===========================================


BASE_SCRIPT_DIR="<STRCVIT_METHODS_DIR>/scripts/Eval_internvl_proxy"

if [ -z "$1" ]; then
  echo "Error: Please provide the model directory prefix."
  exit 1
fi

MODEL_DIR="$1/SMoLoRA"
RESULT_DIR="<STRCVIT_METHODS_DIR>/results/StrCVIT/$MODEL_DIR"
MODEL_BASE="InternVL3_5-8B-Pretrained"

export BASE_SCRIPT_DIR
export MODEL_DIR
export MODEL_BASE
export RESULT_DIR


SCRIPTS=(

    "eval_ad.sh"
    "eval_textcaps.sh"
    "eval_grounding.sh"

    "eval_ocrvqa.sh"

    "eval_gqa.sh"
    "eval_vqav2.sh"

    "eval_places.sh"
    "eval_chartqa.sh"
    "eval_fin.sh"

    "eval_ImageNet.sh"

    "eval_rs.sh"

 

)



GPU_GROUPS=()
total_gpus=${#AVAILABLE_GPUS[@]}
first_task_gpus=$((GPUS_PER_TASK * 2))


if [ "$total_gpus" -gt 0 ]; then
    group_str=""
    for (( j=0; j<first_task_gpus; j++ )); do
        if [ $j -lt $total_gpus ]; then
            if [ -n "$group_str" ]; then group_str="$group_str,"; fi
            group_str="$group_str${AVAILABLE_GPUS[$j]}"
        fi
    done
    GPU_GROUPS+=("$group_str")
fi


for (( i=first_task_gpus; i<total_gpus; i+=GPUS_PER_TASK )); do
    group_str=""

    for (( j=0; j<GPUS_PER_TASK; j++ )); do
        idx=$((i+j))
        if [ $idx -lt $total_gpus ]; then
            if [ -n "$group_str" ]; then group_str="$group_str,"; fi
            group_str="$group_str${AVAILABLE_GPUS[$idx]}"
        fi
    done
   


    GPU_GROUPS+=("$group_str")
done

echo "=== Auto-Configuration ==="
echo "Total GPUs: ${AVAILABLE_GPUS[*]}"
echo "Strategy: first task ${first_task_gpus} GPUs, others ${GPUS_PER_TASK} GPUs per task."
echo "Generated Groups: ${GPU_GROUPS[*]}"
echo "=========================="



INDEX_FILE=$(mktemp)
echo "0" > "$INDEX_FILE"
LOCK_FILE="/tmp/gpu_task_queue.lock"
touch "$LOCK_FILE"

run_worker() {
    local my_gpus=$1
    local worker_name=$2
   
    echo "[$worker_name] Launched. Bound to GPUs: $my_gpus"
   
    exec 200>"$LOCK_FILE"

    while true; do

        flock -x 200
       
        local current_idx
        current_idx=$(cat "$INDEX_FILE")
        if [ -z "$current_idx" ]; then current_idx=9999; fi

        local total_tasks=${#SCRIPTS[@]}
       
        if [ "$current_idx" -ge "$total_tasks" ]; then
            flock -u 200
            break
        fi
       
        local task_script="${SCRIPTS[$current_idx]}"
        echo $((current_idx + 1)) > "$INDEX_FILE"
       
        flock -u 200



        echo "[$worker_name] Processing ($current_idx): $task_script on GPUs $my_gpus"
       
        export USE_GPUS="$my_gpus"
        export CUDA_VISIBLE_DEVICES="$my_gpus"
       
        if [ -n "$task_script" ]; then

            sleep $((RANDOM % 5))
            bash "$BASE_SCRIPT_DIR/$task_script" "$MODEL_DIR" "$MODEL_BASE"
            echo "[$worker_name] Done: $task_script"
        fi
        echo "----------------------------------------"
    done
    exec 200>&-
}

echo "=== Starting Parallel Evaluation ==="

PIDS=()


for gpu_group in "${GPU_GROUPS[@]}"; do
    for ((i=1; i<=TASKS_PER_GROUP; i++)); do

        safe_name=${gpu_group//,/_}
        run_worker "$gpu_group" "Worker-GPUs${safe_name}-T${i}" &
        PIDS+=($!)
        sleep 2
    done
done

echo "=== All workers launched. Waiting for completion... ==="
wait "${PIDS[@]}"

rm -f "$INDEX_FILE" "$LOCK_FILE"

echo "=== All evaluations finished. Calculating results... ==="

python -m swift.llm.eval.new_calculate_ap_proxy --result-dir "$RESULT_DIR"
