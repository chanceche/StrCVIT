#!/bin/bash
if [ -n "$USE_GPUS" ]; then
    gpu_list="$USE_GPUS"
else

    gpu_list="0,1,2,3,4,5,6,7"
fi

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

STAGE='OCRVQA_proxy'



if [ ! -n "$1" ] ;then
    MODELPATH='<STRCVIT_METHODS_DIR>/checkpoints/StrCVIT/default_model'
    RESULT_DIR="<STRCVIT_METHODS_DIR>/results/StrCVIT/default_model"
else
    MODELPATH=<STRCVIT_METHODS_DIR>/checkpoints/StrCVIT/$1
    RESULT_DIR=<STRCVIT_METHODS_DIR>/results/StrCVIT/$1
   
fi

if [ ! -n "$2" ] ;then
    model_base=<MODEL_ROOT>/InternVL3_5-2B-Pretrained
else
    model_base=<MODEL_ROOT>/$2
fi


if [ ! -n "$3" ] ;then
    Instruction="<STRCVIT_DATASET_DIR>/test/OCRVQA/test_proxy.json"
else
    Instruction=<STRCVIT_DATASET_DIR>/$3/OCRVQA/test_proxy.json
fi


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m swift.llm.eval.model_gemma  \
        --model-path $MODELPATH \
        --model-base $model_base\
        --question-file $Instruction \
        --image-folder <STRCVIT_IMAGE_ROOT>/ \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
rm $RESULT_DIR/$STAGE/${CHUNKS}_*.jsonl

python -m swift.llm.eval.eval_ocrvqa \
    --annotation-file $Instruction  \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

