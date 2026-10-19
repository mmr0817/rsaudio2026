#!/usr/bin/env python3
"""
训练脚本

用法:
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml --data_dir data/raw
"""

import argparse
import sys
from pathlib import Path

import yaml

# 把项目根目录加入 path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.models.classifier import build_model
from src.training.trainer import train_model, evaluate_model
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Speaker Quality Classifier")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config yaml"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data.raw_dir in config",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行覆盖
    if args.data_dir:
        cfg["data"]["raw_dir"] = args.data_dir
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.seed:
        cfg["project"]["seed"] = args.seed

    set_seed(cfg["project"]["seed"])

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    data_dir = Path(data_cfg["raw_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    print("=" * 60)
    print("Speaker Quality Classifier - Training")
    print("=" * 60)
    print(f"Data dir   : {data_dir}")
    print(f"Backbone   : {cfg['model']['backbone']}")
    print(f"Epochs     : {train_cfg['num_epochs']}")
    print(f"Batch size : {train_cfg['batch_size']}")
    print(f"LR         : {train_cfg['learning_rate']}")
    print("=" * 60)

    # DataLoaders
    loaders = create_dataloaders(
        root_dir=data_dir,
        sample_rate=data_cfg["sample_rate"],
        max_duration=data_cfg["max_duration"],
        batch_size=train_cfg["batch_size"],
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        num_workers=train_cfg.get("num_workers", 4),
        seed=cfg["project"]["seed"],
    )

    # 计算类别权重（逆频率），缓解不平衡
    from collections import Counter
    import torch
    train_labels = []
    for batch in loaders["train"]:
        train_labels.extend(batch["labels"].tolist())
    cnt = Counter(train_labels)
    num_classes = cfg["model"].get("num_labels", 2)
    total = sum(cnt.values())
    weights = []
    for i in range(num_classes):
        c = cnt.get(i, 1)
        w = total / (num_classes * c)
        weights.append(w)
    class_weight = torch.tensor(weights, dtype=torch.float32)
    print(f"Class counts: {dict(cnt)}")
    print(f"Class weights: {class_weight.tolist()}")
    cfg["model"]["class_weight"] = class_weight

    # Model
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable:,}")

    # Train
    model = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cfg=cfg,
    )

    # 最终在 test 集评估
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = list(data_cfg.get("label_map", {}).keys())
    test_metrics = evaluate_model(
        model, loaders["test"], device, label_names=label_names
    )

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"F1-macro : {test_metrics['f1']:.4f}")
    print(f"F1-weight: {test_metrics['f1_weighted']:.4f}")
    import json
    print("\nClassification Report:")
    print(json.dumps(test_metrics["report"], indent=2, ensure_ascii=False))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("Labels order:", label_names)
    for row in test_metrics["confusion_matrix"]:
        print(row)
    print(f"\nBest model saved to: {train_cfg['output_dir']}/best")


if __name__ == "__main__":
    main()