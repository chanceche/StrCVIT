import argparse
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

from ._smolora_eval_utils import DEFAULT_STRLORA_EMB_DIR, apply_smolora_eval_embedding, sanitize_generated_text


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    transformers.logging.set_verbosity_error()

    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base) if args.model_base else None

    if model_path:
        print(f"Loading model base: {model_base}")
        model = AutoModel.from_pretrained(
            model_base,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        )
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(
                os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {
                (k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()
            }
            if any(k.startswith('model.') for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False, assign=True)

        from StrLoRA.peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        emb_path = apply_smolora_eval_embedding(model, args.question_file, args.smolora_emb_dir)
        print(f"Loaded SMoLoRA eval embedding: {emb_path}")
    else:
        print(f"Loading model from: {model_path}")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        )

    model.eval()
    tokenizer_path = model_base if model_base else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    generation_config = dict(max_new_tokens=50, do_sample=False)

    print(f"Start inference on chunk {args.chunk_idx}...")
    show_progress = (args.chunk_idx == 0) if hasattr(args, 'chunk_idx') else True

    for i, line in enumerate(tqdm(questions, disable=not show_progress, desc="Evaluating", ncols=100)):
        idx = line.get("question_id", i)
        question = line["text"]
        pixel_values = None

        if "image" in line:
            image_path = os.path.join(args.image_folder, line["image"])
            try:
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                if '<image>' not in question:
                    question = '<image>\n' + question
            except Exception as e:
                tqdm.write(f"Error loading image {image_path}: {e}")
                pixel_values = None

        try:
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config=generation_config,
                history=None,
                return_history=True
            )
            response = sanitize_generated_text(response)
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": question,
                "text": response,
                "metadata": {}
            }) + "\n")
            ans_file.flush()
        except Exception as e:
            tqdm.write(f"Error processing question_id {idx}: {e}")
            ans_file.write(json.dumps({
                "question_id": idx,
                "text": f"Error: {str(e)}"
            }) + "\n")

    ans_file.close()
    print(f"Evaluation finished. Results saved to {answers_file}")
    if hasattr(model, "save_router_stats"):
        stats_path = os.path.join(os.path.dirname(answers_file), "router_stats.json")
        model.save_router_stats(stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--smolora-emb-dir", type=str, default=DEFAULT_STRLORA_EMB_DIR)
    args = parser.parse_args()

    eval_model(args)
