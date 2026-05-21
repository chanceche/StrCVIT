import json
import argparse
import os
import re
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default='./cl_dataset/ScienceQA')
    return parser.parse_args()


def sci_acc(result_file):
    with open(result_file, "r") as f:
        data = json.load(f)
    return data['acc']

def text_acc(result_file):
    with open(result_file, "r") as f:
        for line in f:
            if line.startswith("Accuracy:"):
                acc_str = line.split(":")[1].strip().replace("%", "")
                return float(acc_str)
    return None  

def cider_acc(result_file):
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("总 CIDEr 分数"):
                # Extract the value after the colon and strip the percent sign
                value = line.split(":")[1].strip().replace("%", "")
                return float(value)

def gqa_acc(result_file):
    with open(result_file, "r", encoding="utf-8") as f:
        text = f.read()  # Read full content

    # Match the number after "Accuracy:" (supports decimals)
    match = re.search(r'Accuracy:\s*([\d.]+)%', text)
    if match:
        return float(match.group(1))
    else:
        return None




if __name__ == '__main__':
    args = get_args()

    results = {}
    results['AD'] = text_acc(os.path.join(args.result_dir, 'AD_proxy/Result.text'))
    results['Fin'] = text_acc(os.path.join(args.result_dir, 'Fin_proxy/Result.text'))
    results['RS'] = text_acc(os.path.join(args.result_dir, 'RS_proxy/Result.text'))
    results['ImageNet'] = text_acc(os.path.join(args.result_dir, 'ImageNet_proxy/Result.text'))
    # results['GQA'] = gqa_acc(os.path.join(args.result_dir, 'GQA_proxy/Result.text'))
    results['VQAv2'] = text_acc(os.path.join(args.result_dir, 'vqav2_proxy/Result.text'))
    results['Places'] = text_acc(os.path.join(args.result_dir, 'Places_proxy/Result.text'))
    results['TextCaps'] = cider_acc(os.path.join(args.result_dir, 'TextCaps_proxy/cider_result.txt'))
    results['OCRVQA'] = text_acc(os.path.join(args.result_dir, 'OCRVQA_proxy/Result.text'))
    results['Grounding'] = text_acc(os.path.join(args.result_dir, 'Grounding_proxy/Result.text'))
    # results['ChartQA'] = text_acc(os.path.join(args.result_dir, 'ChartQA/Result.text'))
    



    results['AP'] = sum(results.values()) / len(results)

    # Save as JSON file
    with open(os.path.join(args.result_dir, 'Ap_proxy_result.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Print results
    for k, v in results.items():
        print(f"{k}: {v:.2f}")