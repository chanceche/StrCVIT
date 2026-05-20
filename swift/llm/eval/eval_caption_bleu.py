import os
import argparse
import json
import codecs
from functools import reduce
import operator
import math

# BLEU 计算相关函数
def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count

def best_length_match(ref_l, cand_l):
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best

def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu

# 用于评估 BLEU 分数的函数
def eval_bleu(annotation_file, result_file):
    # 读取 annotation-file (真实标签)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    annotations_dict = {str(item['question_id']): item for item in annotations}

    # 读取 result-file (模型预测结果)
    results = [json.loads(line) for line in open(result_file)]

    # 准备 BLEU 评估的输入数据
    gts = []  # 参考答案
    res = []  # 模型生成的文本
    for result in results:
        question_id = str(result['question_id'])
        if question_id in annotations_dict:
            gts.append(annotations_dict[question_id]['answer'] + "\n")  # 加入换行符以匹配 fetch_data 的格式
            res.append(result['text'] + "\n")  # 模型的预测结果

    # 对每一句进行 BLEU 计算并打印结果
    for idx, (ref, hyp) in enumerate(zip(gts, res)):
        bleu_score = BLEU([hyp], [[ref]])  # 计算单句的 BLEU 分数
        print(f"Sample {idx + 1} - BLEU 分数: {bleu_score:.2f}")
    
    # 总 BLEU 分数
    bleu_score_total = BLEU(res, [gts])
    print(f"总 BLEU 分数: {bleu_score_total:.2f}")
    
    # 将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'bleu_result.txt')
        with open(output_file, 'w') as f:
            for idx, (ref, hyp) in enumerate(zip(gts, res)):
                bleu_score = BLEU([hyp], [[ref]])
                f.write(f"Sample {idx + 1} - BLEU 分数: {bleu_score:.2f}\n")
            f.write(f"总 BLEU 分数: {bleu_score_total:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./cl_dataset/annotations.json')
    parser.add_argument('--result-file', type=str, default='./results/predictions.jsonl')
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    if args.result_file is not None:
        eval_bleu(args.annotation_file, args.result_file)
