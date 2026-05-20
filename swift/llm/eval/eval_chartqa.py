import os
import argparse
import json
import re
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/ImageNet/test.json')
    parser.add_argument('--result-file', type=str, default='./results/StrCVIT/ChartQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/StrCVIT/ChartQA')
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    answers = [annotation['answer'] for annotation in annotations]
    results = [json.loads(line) for line in open(result_file)]
    print(f"Length of answers: {len(answers)}")
    total = len(results)
    print(total)
    right = 0
    false_answers = []
    for index in tqdm(range(total)):
        annotation = answers[index].replace('.', '')
        label = results[index]
        if (annotation.upper() in label['text'].upper()):
            right += 1
        else:
            label['ground_truth'] = annotation
            false_answers.append(label)
        
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
            json.dump(false_answers,f,indent=4)

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
