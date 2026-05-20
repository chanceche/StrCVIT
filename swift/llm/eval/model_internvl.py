import math
import numpy as np
import argparse
import torch
import logging
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import transformers
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel as peftmodel
import json
import os
import re

# 如果需要处理视频，请取消下面两行的注释
# from decord import VideoReader, cpu

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------
# 1. InternVL3.5 Preprocessing Logic (Directly from official Snippet)
# ----------------------------------------------------------------------

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ----------------------------------------------------------------------
# 2. Video Logic (Updated to InternVL3.5 version)
# ----------------------------------------------------------------------

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    # Ensure VideoReader is imported if you use this function
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# ----------------------------------------------------------------------
# 3. User Utilities (Split List, etc.)
# ----------------------------------------------------------------------

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

from huggingface_hub import hf_hub_download
def load_from_hf(repo_id, filename, subfolder=None):
    cache_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder)
    return torch.load(cache_file, map_location='cpu')

# ----------------------------------------------------------------------
# 4. Evaluation Loop
# ----------------------------------------------------------------------

def eval_model(args):
    # 全局屏蔽非 Error 级别的日志
    transformers.logging.set_verbosity_error()

    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base) if args.model_base else None

    # Load Model
    if model_path:
        print(f"Loading model base: {model_base}")
        # InternVL3.5 Base Model Loading
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
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False, assign=True)

        from StrLoRA.peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model_path_upper = model_path.upper()
        if not any(name in model_path_upper for name in ("MOE", "SMOLORA", "STRLORA")):
            model = model.merge_and_unload()

    else:
        # If no model_base provided, assume model_path is the full model
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
    
    # InternVL3.5 uses 'model_path' or 'model_base' for tokenizer
    tokenizer_path = model_base if model_base else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)

    # Load Question Data
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 3.5 Generation Config
    generation_config = dict(max_new_tokens=100, do_sample=False)

    print(f"Start inference on chunk {args.chunk_idx}...")
    
    show_progress = (args.chunk_idx == 0) if hasattr(args, 'chunk_idx') else True

    for i, line in enumerate(tqdm(questions, disable=not show_progress, desc="Evaluating", ncols=100)):
        idx = line.get("question_id", i)
        question = line["text"]
        
        pixel_values = None
        
        if "image" in line.keys():
            image_path = os.path.join(args.image_folder, line["image"])
            # Load Image (InternVL3.5 logic)
            # max_num=12 is standard for 3.5 to handle high res
            try:
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                
                if '<image>' not in question:
                    question = '<image>\n' + question
            except Exception as e:
                tqdm.write(f"Error loading image {image_path}: {e}")
                # Skip this sample or continue with text only depending on requirement
                # Here we continue assuming text-only fallback or fail
                pixel_values = None
        else:
            pixel_values = None

        try:
            # InternVL3.5 Chat Interface
            # signature: chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            # Returns: (response, history)
            
            response, _ = model.chat(
                tokenizer, 
                pixel_values, 
                question, 
                generation_config=generation_config,
                history=None,       # Single turn evaluation
                return_history=True 
            )
            
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
    parser.add_argument("--model-base", type=str, default=None, help="Base model if model-path is a LoRA adapter")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
