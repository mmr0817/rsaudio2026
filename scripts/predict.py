#!/usr/bin/env python3
"""
推理脚本

用法:
    python scripts/predict.py --model outputs/models/best --audio path/to/file.wav
    python scripts/predict.py --model outputs/models/best --audio_dir path/to/folder
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import load_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Predict speaker quality from sweep wav")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/best",
        help="Path to saved model directory",
    )
    parser.add_argument("--audio", type=str, default=None, help="Single wav file")
    parser.add_argument(
        "--audio_dir", type=str, default=None, help="Directory of wav files"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config yaml"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to json"
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    predictor = load_predictor(
        model_path=model_path,
        config_path=args.config,
        device=args.device,
    )

    results = []

    if args.audio:
        path = Path(args.audio)
        if not path.is_absolute():
            path = ROOT / path
        res = predictor.predict_file(path)
        results.append(res)
        print_result(res)

    elif args.audio_dir:
        folder = Path(args.audio_dir)
        if not folder.is_absolute():
            folder = ROOT / folder
        files = sorted(list(folder.rglob("*.wav")) + list(folder.rglob("*.WAV")))
        print(f"Found {len(files)} wav files")
        for f in files:
            res = predictor.predict_file(f)
            results.append(res)
            print_result(res)
    else:
        print("Please provide --audio or --audio_dir")
        return

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")


def print_result(res: dict):
    print("-" * 50)
    print(f"File       : {res['path']}")
    print(f"Prediction : {res['prediction_cn']} ({res['prediction']})")
    print(f"Confidence : {res['confidence']:.4f}")
    if "probabilities_cn" in res:
        probs_str = " | ".join(
            f"{k}: {v:.3f}" for k, v in res["probabilities_cn"].items()
        )
        print(f"Probs      : {probs_str}")


if __name__ == "__main__":
    main()