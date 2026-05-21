import os
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
    gts = {}  # References
    res = []  # Model outputs (list)

    for result in results:
        question_id = str(result['question_id'])
        if question_id in annotations_dict:
            gts[question_id] = [annotations_dict[question_id]['answer']]  # Ground truth

            # Build res in the format expected by cider.py
            res.append({
                "image_id": question_id,  # Use question_id as unique id
                "caption": [result['text']]  # Model prediction as list
            })

    # Compute CIDEr score
    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)
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