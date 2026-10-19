#!/usr/bin/env python3
"""
在测试集上评估已训练模型

用法:
    python scripts/evaluate.py --model outputs/models/best --data_dir data/raw
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.inference.predict import load_predictor
from src.training.trainer import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/best")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    predictor = load_predictor(model_path, config_path=config_path, device=args.device)
    model = predictor.model
    device = predictor.device

    data_cfg = cfg["data"]
    loaders = create_dataloaders(
        root_dir=data_dir,
        sample_rate=data_cfg["sample_rate"],
        max_duration=data_cfg["max_duration"],
        batch_size=args.batch_size,
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        num_workers=0,
        seed=cfg["project"]["seed"],
    )

    label_names = list(data_cfg.get("label_map", {}).keys())
    metrics = evaluate_model(model, loaders["test"], device, label_names=label_names)

    print("=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"F1 (macro)    : {metrics['f1']:.4f}")
    print(f"F1 (weighted) : {metrics['f1_weighted']:.4f}")
    print(f"Loss          : {metrics['loss']:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("Labels:", label_names)
    for row in metrics["confusion_matrix"]:
        print(row)
    print("\nDetailed Report:")
    print(json.dumps(metrics["report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()