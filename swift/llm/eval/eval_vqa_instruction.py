import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='./cl_dataset/ScienceQA')
    parser.add_argument('--result-file', type=str, default='./results/StrCVIT/Qwen/ScienceQA/Finetune/merge.jsonl')
    parser.add_argument('--output-file', type=str, default='./results/StrCVIT/Qwen/ScienceQA/Finetune/output.jsonl')
    parser.add_argument('--annotation-file', type=str, default='./annotations.json')  # Read correct answers
    return parser.parse_args()

def count_words(sentence):
    """Count words in a sentence."""
    return len(sentence.split())

if __name__ == "__main__":
    args = get_args()

    # Read prediction and annotation files
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}

    annotations = json.load(open(args.annotation_file))  # Read answer annotations
    answers = {item['question_id']: item['answer'] for item in annotations}

    results = {'positive': [], 'negative': []}

    for pred in predictions.values():
        question_id = pred['question_id']
        pred_text = pred['text'].strip()  # Strip leading/trailing whitespace

        # Count words
        word_count = count_words(pred_text)

        # Determine positive/negative sample
        if word_count <= 2:
            results['positive'].append({
                'question_id': question_id,
                'pred_text': pred_text,
                'valid': True
            })
        else:
            # Check whether it is a correct answer
            correct_answer = answers.get(question_id, "")
            if pred_text.lower() == correct_answer.lower():  # Case-insensitive compare
                results['positive'].append({
                    'question_id': question_id,
                    'pred_text': pred_text,
                    'valid': True
                })
            else:
                results['negative'].append({
                    'question_id': question_id,
                    'pred_text': pred_text,
                    'valid': False
                })

    # Compute accuracy
    correct = len(results['positive'])
    total = correct + len(results['negative'])
    accuracy = correct / total * 100 if total > 0 else 0

    # Save results to output file
    with open(args.output_file, 'w') as f:
        # Write accuracy at the top
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        json.dump(results, f, indent=2)

    # Print summary
    print(f"Total Positive Samples: {len(results['positive'])}, Total Negative Samples: {len(results['negative'])}")
    print(f"Accuracy: {accuracy:.2f}%")
