#!/bin/bash

export PYTHONPATH="<STRCVIT_METHODS_DIR>:${PYTHONPATH:-}"

if [ $# -lt 3 ]; then
    echo "Usage: bash run_dataset.sh <model_dir> <model_base> <dataset_key>"
    exit 1
fi

MODEL_DIR="$1"
MODEL_BASE="$2"
DATASET_KEY="$3"
MODELPATH="<STRCVIT_METHODS_DIR>/checkpoints/StrCVIT/$MODEL_DIR"
RESULT_DIR="<STRCVIT_METHODS_DIR>/results/StrCVIT/$MODEL_DIR"
IMAGE_FOLDER="<STRCVIT_IMAGE_ROOT>/"
STRLORA_EMB_DIR="${STRLORA_EMB_DIR:-$(cd "$(dirname "$0")" && pwd)/eval_dataset_embs}"

if [ -n "${USE_GPUS:-}" ]; then
    gpu_list="$USE_GPUS"
else
    gpu_list="0,1,2,3"
fi
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

QUESTION_FILE=""
ANNOTATION_FILE=""
STAGE=""

case "$DATASET_KEY" in
    textcaps)
        STAGE="TextCaps_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/TextCaps/val_proxy.json"
        ;;
    ad)
        STAGE="AD_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/AD/test_proxy.json"
        ANNOTATION_FILE="<STRCVIT_DATASET_DIR>/test/AD/test.json"
        ;;
    rs)
        STAGE="RS_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/RS/test_proxy.json"
        ANNOTATION_FILE="<STRCVIT_DATASET_DIR>/test/RS/test.json"
        ;;
    imagenet)
        STAGE="ImageNet_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/ImageNet/test_proxy.json"
        ;;
    gqa)
        STAGE="GQA_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/GQA/test_proxy.json"
        ;;
    vqav2)
        STAGE="vqav2_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/VQAv2/val_proxy.json"
        ;;
    grounding)
        STAGE="Grounding_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/Grounding/test_proxy.json"
        ;;
    places365)
        STAGE="Places_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/Places365/val_proxy.json"
        ;;
    fin)
        STAGE="Fin_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/Fin/test_proxy.json"
        ANNOTATION_FILE="<STRCVIT_DATASET_DIR>/test/Fin/test.json"
        ;;
    ocrvqa)
        STAGE="OCRVQA_proxy"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/OCRVQA/test_proxy.json"
        ;;
    chartqa)
        STAGE="ChartQA"
        QUESTION_FILE="<STRCVIT_DATASET_DIR>/test/ChartQA/test.json"
        ;;
    *)
        echo "Unknown dataset key: $DATASET_KEY"
        exit 1
        ;;
esac

for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m swift.llm.eval.model_internvl_smolora \
        --model-path "$MODELPATH" \
        --model-base "<MODEL_ROOT>/$MODEL_BASE" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --smolora-emb-dir "$STRLORA_EMB_DIR" &
done

wait

output_file="$RESULT_DIR/$STAGE/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS - 1))); do
    cat "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
rm -f "$RESULT_DIR/$STAGE/${CHUNKS}_"*.jsonl

case "$DATASET_KEY" in
    textcaps)
        python -m swift.llm.eval.eval_textcaps_proxy_cider \
            --annotation-file "$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_caption_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    ad)
        python -m swift.llm.eval.eval_ai2d \
            --annotation-file "$ANNOTATION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        ;;
    rs)
        python -m swift.llm.eval.eval_ai2d \
            --annotation-file "$ANNOTATION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_vqa_instruction \
            --annotation-file "$ANNOTATION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    imagenet)
        python -m swift.llm.eval.eval_ImagetNet \
            --test-file "$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_imagenet_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    gqa)
        python -m swift.llm.eval.convert_gqa_for_eval \
            --src "$output_file" \
            --dst "$RESULT_DIR/$STAGE/testdev_balanced_predictions.json"
        python -m swift.llm.eval.eval_gqa \
            --tier testdev_balanced \
            --path "$RESULT_DIR/$STAGE" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_vqa_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    vqav2)
        python -m swift.llm.eval.eval_vqav2 \
            --result-file "$output_file" \
            --annotation-file "$QUESTION_FILE" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_vqa_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    grounding)
        python -m swift.llm.eval.eval_grounding \
            --test-file "$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.create_prompt \
            --rule <STRCVIT_METHODS_DIR>/swift/llm/eval/rule.json \
            --questions "$QUESTION_FILE" \
            --results "$output_file" \
            --rule_temp StrCVIT_Grounding
        ;;
    places365)
        python -m swift.llm.eval.eval_ImagetNet \
            --test-file "$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_imagenet_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    fin)
        python -m swift.llm.eval.eval_finvis \
            --annotation-file "$ANNOTATION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_vqa_instruction \
            --annotation-file "$ANNOTATION_FILE" \
            --result-file "$output_file" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl"
        ;;
    ocrvqa)
        python -m swift.llm.eval.eval_ocrvqa \
            --annotation-file "$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.create_prompt \
            --rule <STRCVIT_METHODS_DIR>/swift/llm/eval/rule.json \
            --questions "$QUESTION_FILE" \
            --results "$output_file"
        python -m swift.llm.eval.eval_vqa_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
    chartqa)
        python -m swift.llm.eval.eval_chartqa \
            --annotation-file "$QUESTION_FILE" \
            --result-file "$output_file" \
            --output-dir "$RESULT_DIR/$STAGE"
        python -m swift.llm.eval.eval_vqa_instruction \
            --annotation-file "$QUESTION_FILE" \
            --output-file "$RESULT_DIR/$STAGE/eval_instruction.jsonl" \
            --result-file "$output_file"
        ;;
esac
