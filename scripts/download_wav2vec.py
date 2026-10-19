#!/usr/bin/env python3
"""
下载 fairseq 版本的 wav2vec2 预训练模型到本地

用法:
    python scripts/download_wav2vec.py
    python scripts/download_wav2vec.py --model base --output_dir pretrained
"""

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

# 官方 fairseq 预训练模型下载地址
MODEL_URLS = {
    "base": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
        "filename": "wav2vec_small.pt",
        "desc": "Wav2Vec 2.0 Base (no finetuning, Librispeech 960h)",
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
        "filename": "wav2vec_large.pt",
        "desc": "Wav2Vec 2.0 Large (no finetuning, Librispeech)",
    },
    # 备用镜像（如果官方慢）
    # "base_hf": {
    #     "url": "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_small.pt",
    #     "filename": "wav2vec_small.pt",
    # },
}


def download_file(url: str, dest: Path, desc: str = ""):
    print(f"Downloading {desc or url}")
    print(f"  → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 / total_size)
            mb = block_num * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {percent:5.1f}%  ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    try:
        urlretrieve(url, str(dest), reporthook=progress)
        print("\n  Done.")
    except Exception as e:
        print(f"\n  Failed: {e}")
        if dest.exists():
            dest.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description="Download fairseq wav2vec2 models")
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=list(MODEL_URLS.keys()),
        help="Which model to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrained",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    info = MODEL_URLS[args.model]
    dest = out_dir / info["filename"]

    if dest.exists() and not args.force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {dest} ({size_mb:.1f} MB)")
        print("Use --force to re-download.")
        return

    download_file(info["url"], dest, info["desc"])
    print(f"\nSaved to: {dest.resolve()}")
    print(f"\n请在 config.yaml 中设置:")
    print(f'  model.backbone: "{dest.resolve()}"')
    print(f'  或相对路径: "pretrained/{info["filename"]}"')


if __name__ == "__main__":
    main()