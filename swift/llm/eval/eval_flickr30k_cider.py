import os
import numpy as np
import argparse
import json
from swift.llm.eval.cider import Cider  # Use CIDEr scorer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./cl_dataset/annotations.json')  # GT labels file
    parser.add_argument('--result-file', type=str, default='./results/predictions.jsonl')  # Model prediction file
    parser.add_argument('--output-dir', type=str)  # Output directory
    return parser.parse_args()

def eval_cider(annotation_file, result_file):
    # Read annotation-file (ground truth)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Convert annotations list to dict for lookup by question_id
    annotations_dict = {str(item['question_id']): item for item in annotations}

    # Read result-file (model predictions)
    results = [json.loads(line) for line in open(result_file)]

    # Prepare CIDEr inputs
    res = {}  # Model outputs (dict)
    gts = {}  # Reference answers (dict)

    for result in results:
        question_id = str(result['question_id'])
        if question_id in annotations_dict:
            # Build res in the format expected by cider.py
            res[question_id] = [result['text'].lower()]  # Use question_id as key; prediction as list

            # Put all references into gts
            gts[question_id] = [ans.lower() for ans in annotations_dict[question_id]['answer']]
          # List of multiple references

    # Initialize cider scorer
    cider_scorer = Cider()

    # Initialize final reference dict
    final_gts = {}

    # Iterate over predictions
    for question_id, res_text in res.items():
        if question_id in gts:
            temp_gts = gts[question_id]  # Five references for this question

            # Expand temp_res to five identical entries
            temp_res = res_text * len(temp_gts)

            # Use different ids (1..5) to compute against gts
            temp_res_dict = {str(i + 1): [temp_res[i]] for i in range(len(temp_res))}
            temp_gts_dict = {str(i + 1): [temp_gts[i]] for i in range(len(temp_gts))}
            
            # Compute CIDEr for each temp_res vs corresponding gts
            score, scores = cider_scorer.compute_score(temp_gts_dict, temp_res_dict)

            # Pick the best score index
            best_gts_index = np.argmax(scores)  # Index of highest-scoring reference
            
            # Get the corresponding reference
            best_gts = temp_gts[best_gts_index]

            # Add the best reference into final_gts
            final_gts[question_id] = [best_gts.lower()]


    # Compute final CIDEr score
    score, scores = cider_scorer.compute_score(res, final_gts)

    # Scale scores by 100
    score *= 100
    scores = [s * 100 for s in scores]

    # Print total CIDEr score with percent sign
    print(f"总 CIDEr 分数: {score:.2f}%")

    # Write results to file
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'cider_result.txt')
        with open(output_file, 'w') as f:
            f.write(f"总 CIDEr 分数: {score:.2f}%\n")
            for question_id, cider_score in zip(gts.keys(), scores):
                f.write(f"Question ID: {question_id}, CIDEr Score: {cider_score:.2f}%\n")

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_cider(args.annotation_file, args.result_file)
