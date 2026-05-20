import argparse
from copy import deepcopy

from pprint import pprint
from collections import defaultdict
import pandas as pd
import json
def load_file_jsonl(path):
   with open(path) as f:
        return [json.loads(row) for row in f]

def get_avg(x):
  return sum([float(y) for y in x])/len(x)

def get_domain(x):
  for domain in ['chest_xray', 'mri', 'histology', 'gross', 'ct_scan']:
    in_domain = x['domain'][domain]
    if in_domain:
      return domain
    
import os

def main(args):
    scores_data = load_file_jsonl(args.scores_file)
    predictions = [(x['question_id'], x['type'], get_domain(x), x['gpt_eval'].split('\n')[0].split(' ')) for x in scores_data]
    
    score_type_dict = defaultdict(lambda: defaultdict(list))
    for q_id, q_type, domain, (a1_score, a2_score) in predictions:
        score_type_dict[q_type][1].append(a1_score)
        score_type_dict[q_type][2].append(a2_score)
        score_type_dict['overall'][1].append(a1_score)
        score_type_dict['overall'][2].append(a2_score)
        score_type_dict[domain][1].append(a1_score)
        score_type_dict[domain][2].append(a2_score)

    result = defaultdict(dict)
    for q_type, score_dict in score_type_dict.items():
        result[q_type]['gpt4_score'] = get_avg(score_dict[1])
        result[q_type]['pred_score'] = get_avg(score_dict[2])
        result[q_type]['pred_relative_score'] = get_avg([float(s2)/float(s1) for s1, s2 in zip(score_dict[1], score_dict[2])]) * 100
        result[q_type]['data_size'] = len(score_dict[1])

    df = pd.DataFrame.from_dict(result).filter(['conversation', 'detailed_description', 'chest_xray', 'mri', 'histology', 'gross', 'ct_scan', 'overall'])
    print(df)

    # 获取 source-file 文件所在目录并保存 results.txt
    source_file_path = args.scores_file
    output_dir = os.path.dirname(os.path.abspath(source_file_path))
    save_path = os.path.join(output_dir, "results.txt")

    with open(save_path, "w") as f:
        f.write(df.to_string())
    

if __name__ == '__main__':
   parser = argparse.ArgumentParser("GPT-4 Multimodal Chat Eval Postprocessing", add_help=True)
   parser.add_argument("--scores-file", default="", metavar="FILE", help="input path to gpt-4 score file")
   args = parser.parse_args()
   main(args)