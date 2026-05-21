import os
import argparse
import json
from swift.llm.eval.cider import Cider

def get_args():
    parser = argparse.ArgumentParser()
    # Ground-truth labels file (annotations.json)
    parser.add_argument('--annotation-file', type=str, default='./cl_dataset/annotations.json') 
    # Model predictions file (predictions.jsonl)
    parser.add_argument('--result-file', type=str, default='./results/predictions.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/')
    return parser.parse_args()

def eval_cider(annotation_file, result_file):
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 1. Build Ground Truths (gts)
    # Structure: { '200000000': ['sent1', 'sent2', 'sent3', 'sent4', 'sent5'] }
    gts = {}
    for item in annotations:
        qid = str(item['question_id']) # Ensure ID is a string
        gts[qid] = item['answer']      # Read list directly; no append needed
    
    print(f"Loaded {len(gts)} GT samples.")
    # Check the first sample: list with length 5
    first_key = list(gts.keys())[0]
    print(f"Sample GT ID: {first_key}, Refs count: {len(gts[first_key])}")

    # 2. Build Predictions (res)
    # Structure: { '200000000': ['model generated sentence'] }
    print(f"Loading results from {result_file}...")
    results = [json.loads(line) for line in open(result_file)]
    
    res = {}
    for result in results:
        qid = str(result['question_id'])
        
        # Only handle IDs that exist in gts
        if qid in gts:
            # Note: CIDEr expects predictions wrapped in a list even for one sentence
            res[qid] = [result['text']]
    
    print(f"Matched {len(res)} predictions.")

    # 3. Compute score
    # Core logic: 1 prediction sentence vs 5 reference sentences
    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)
    score *= 100
    
    print("------------------------------------------------")
    print(f"Final CIDEr Score: {score:.2f}%")
    print("------------------------------------------------")

    # Save results
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