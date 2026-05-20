import os
import pickle
import re
from typing import Dict

import torch


DEFAULT_STRLORA_EMB_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "scripts/Eval_internvl_proxy/SMoLoRA/eval_dataset_embs",
)

DATASET_KEY_TO_EMB_FILE: Dict[str, str] = {
    "textcaps": "textcaps.pkl",
    "ad": "ad.pkl",
    "rs": "rs.pkl",
    "imagenet": "imagenet.pkl",
    "gqa": "gqa.pkl",
    "vqav2": "vqav2.pkl",
    "grounding": "grounding.pkl",
    "places365": "places365.pkl",
    "fin": "fin.pkl",
    "ocrvqa": "ocrvqa.pkl",
    "chartqa": "chartqa.pkl",
}


def sanitize_generated_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def infer_dataset_key_from_question_file(question_file: str) -> str:
    question_path = os.path.abspath(os.path.expanduser(question_file)).lower()
    if "/textcaps/" in question_path:
        return "textcaps"
    if "/ad/" in question_path:
        return "ad"
    if "/rs/" in question_path:
        return "rs"
    if "/imagenet/" in question_path:
        return "imagenet"
    if "/gqa/" in question_path:
        return "gqa"
    if "/vqav2/" in question_path:
        return "vqav2"
    if "/grounding/" in question_path:
        return "grounding"
    if "/places365/" in question_path:
        return "places365"
    if "/fin/" in question_path:
        return "fin"
    if "/ocrvqa/" in question_path:
        return "ocrvqa"
    if "/chartqa/" in question_path:
        return "chartqa"
    raise ValueError(f"Cannot infer SMoLoRA dataset key from question file: {question_file}")


def resolve_smolora_emb_path(question_file: str, emb_dir: str) -> str:
    dataset_key = infer_dataset_key_from_question_file(question_file)
    emb_file = DATASET_KEY_TO_EMB_FILE[dataset_key]
    emb_path = os.path.join(os.path.abspath(os.path.expanduser(emb_dir)), emb_file)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"SMoLoRA embedding file not found for {dataset_key}: {emb_path}")
    return emb_path


def load_single_embedding(emb_path: str) -> torch.Tensor:
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    emb = emb.detach().cpu()
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    if emb.dim() != 2 or emb.size(0) != 1:
        raise ValueError(f"Expected a single SMoLoRA embedding with shape (1, hidden), got {tuple(emb.shape)}")
    return emb


def apply_smolora_eval_embedding(model, question_file: str, emb_dir: str = DEFAULT_STRLORA_EMB_DIR) -> str:
    from StrLoRA.peft.tuners.smolora import SMoLoraLinear

    emb_path = resolve_smolora_emb_path(question_file, emb_dir)
    emb = load_single_embedding(emb_path)
    emb_list = emb.tolist()

    matched_modules = 0
    for module in model.modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_emb = emb_list
            module.ins_type = 0
            matched_modules += 1

    if hasattr(model, "peft_config"):
        for config in model.peft_config.values():
            if hasattr(config, "ins_emb"):
                config.ins_emb = emb_list
            if hasattr(config, "ins_type"):
                config.ins_type = 0

    if matched_modules == 0:
        raise ValueError("No SMoLoRA modules were found while applying eval embeddings.")

    return emb_path
