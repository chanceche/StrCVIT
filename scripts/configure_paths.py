#!/usr/bin/env python3
import os
import sys
import sysconfig
from pathlib import Path


def detect_cuda_home():
    for key in ("purelib", "platlib"):
        nvidia_dir = Path(sysconfig.get_paths()[key]) / "nvidia"
        for path in sorted(nvidia_dir.glob("cu*")):
            if (path / "bin" / "nvcc").is_file():
                return str(path)
    return ""


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cuda_home = os.environ.get("CUDA_HOME") or detect_cuda_home()
    required = [
        "MODEL_ROOT",
        "STRCVIT_DATASET_DIR",
        "STRCVIT_IMAGE_ROOT",
    ]
    missing = [name for name in required if not os.environ.get(name)]
    if not cuda_home:
        missing.append("CUDA_HOME")
    if missing:
        print("Missing required environment variables:")
        for name in missing:
            print(f"  {name}")
        print("\nExample:")
        print("  export MODEL_ROOT=/path/to/models")
        print("  export STRCVIT_DATASET_DIR=/path/to/StrCVIT_dataset")
        print("  export STRCVIT_IMAGE_ROOT=/path/to/raw_images")
        print("  export CUDA_HOME=/path/to/cuda  # optional when nvidia-cuda-nvcc is installed")
        return 1

    pairs = {
        "<STRCVIT_METHODS_DIR>": os.environ.get("STRCVIT_METHODS_DIR", str(repo_root)).rstrip("/"),
        "<MODEL_ROOT>": os.environ["MODEL_ROOT"].rstrip("/"),
        "<STRCVIT_DATASET_DIR>": os.environ["STRCVIT_DATASET_DIR"].rstrip("/"),
        "<STRCVIT_IMAGE_ROOT>": os.environ["STRCVIT_IMAGE_ROOT"].rstrip("/"),
        "<CUDA_HOME>": cuda_home.rstrip("/"),
    }

    targets = []
    targets.extend((repo_root / "scripts").rglob("*.sh"))
    targets.extend((repo_root / "swift" / "llm" / "eval").rglob("*.py"))
    dataset_dir = Path(os.environ["STRCVIT_DATASET_DIR"])
    if dataset_dir.exists():
        targets.extend(dataset_dir.rglob("*.json"))

    updated = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, new in pairs.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            updated.append(path)

    print("Configuring StrCVIT paths:")
    for old, new in pairs.items():
        print(f"  {old} -> {new}")
    print(f"Updated {len(updated)} files.")
    for path in updated[:20]:
        print(f"  {path}")
    if len(updated) > 20:
        print(f"  ... {len(updated) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
