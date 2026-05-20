import os
import argparse
import json
from swift.llm.eval.cider import Cider

def get_args():
    parser = argparse.ArgumentParser()
    # 你的真实标签文件 (annotations.json)
    parser.add_argument('--annotation-file', type=str, default='./cl_dataset/annotations.json') 
    # 你的模型预测文件 (predictions.jsonl)
    parser.add_argument('--result-file', type=str, default='./results/predictions.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/')
    return parser.parse_args()

def eval_cider(annotation_file, result_file):
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 1. 构建 Ground Truths (gts)
    # 结构: { '200000000': ['sent1', 'sent2', 'sent3', 'sent4', 'sent5'] }
    gts = {}
    for item in annotations:
        qid = str(item['question_id']) # 确保 ID 是字符串
        gts[qid] = item['answer']      # 直接读取 list，不需要再 append
    
    print(f"Loaded {len(gts)} GT samples.")
    # 检查一下第一个数据，确保是 list 且长度为 5
    first_key = list(gts.keys())[0]
    print(f"Sample GT ID: {first_key}, Refs count: {len(gts[first_key])}")

    # 2. 构建 Predictions (res)
    # 结构: { '200000000': ['model generated sentence'] }
    print(f"Loading results from {result_file}...")
    results = [json.loads(line) for line in open(result_file)]
    
    res = {}
    for result in results:
        qid = str(result['question_id'])
        
        # 只处理在 gts 中存在的 ID
        if qid in gts:
            # 注意：CIDEr 库要求预测结果也被包裹在 list 中，即使只有一句话
            res[qid] = [result['text']]
    
    print(f"Matched {len(res)} predictions.")

    # 3. 计算分数
    # 核心逻辑：res 的 1 句话 vs gts 的 5 句话
    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)
    score *= 100
    
    print("------------------------------------------------")
    print(f"Final CIDEr Score: {score:.2f}%")
    print("------------------------------------------------")

    # 保存结果
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