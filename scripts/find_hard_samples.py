#!/usr/bin/env python3
"""
反复划分并训练，筛选难分类音频。

保证：
  - 每个样本在多轮中都会进入 train / val / test
  - 每个样本被测试多次（RepeatedStratifiedKFold），错误率更可靠

用法:
  python scripts/find_hard_samples.py --config config.yaml --n_splits 5 --n_repeats 2 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SpeakerQualityDataset, collate_fn, prepare_dataset_from_folders
from src.models.classifier import build_model
from src.training.trainer import train_model
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--n_splits", type=int, default=5, help="每轮 K 折")
    p.add_argument("--n_repeats", type=int, default=2, help="重复几轮不同随机种子的 K 折")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs/hard_samples")
    p.add_argument("--val_ratio", type=float, default=0.15, help="从 train fold 中划 val 的比例")
    return p.parse_args()


@torch.no_grad()
def predict_all(model, file_paths, labels, sample_rate, max_duration, batch_size, device):
    ds = SpeakerQualityDataset(file_paths, labels, sample_rate, max_duration)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    all_preds, all_probs, all_labels, all_paths = [], [], [], []
    model.eval()
    for batch in loader:
        x = batch["input_values"].to(device)
        out = model(input_values=x)
        probs = out["probs"].cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_preds.append(preds)
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        all_paths.extend(batch["paths"])
    return {
        "paths": all_paths,
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
    }


def make_loader(paths, labels, idxs, sample_rate, max_duration, batch_size, shuffle, num_workers):
    ds = SpeakerQualityDataset(
        [paths[i] for i in idxs],
        labels[idxs].tolist(),
        sample_rate=sample_rate,
        max_duration=max_duration,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["raw_dir"] = args.data_dir
    cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    cfg["project"]["seed"] = args.seed
    cfg["training"]["early_stopping_patience"] = max(
        cfg["training"].get("early_stopping_patience", 5), args.epochs
    )

    set_seed(args.seed)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    data_dir = Path(data_cfg["raw_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    paths, labels, df = prepare_dataset_from_folders(data_dir)
    paths = [Path(p) for p in paths]
    labels = np.array(labels)
    n_samples = len(paths)
    num_classes = int(labels.max()) + 1
    cfg["model"]["num_labels"] = num_classes

    total_rounds = args.n_splits * args.n_repeats
    print("=" * 60)
    print("Hard Sample Mining (保证 train/val/test 全覆盖 + 多次测试)")
    print("=" * 60)
    print(f"样本数        : {n_samples}")
    print(f"类别数        : {num_classes}")
    print(f"K-Fold        : {args.n_splits}")
    print(f"重复轮数      : {args.n_repeats}")
    print(f"总训练次数    : {total_rounds}")
    print(f"每折 epoch    : {args.epochs}")
    print(f"每样本约被测试: {args.n_repeats} 次 (每轮 K 折中恰好 1 次)")
    print("=" * 60)

    # 统计：角色次数 + 测试表现
    stats = {
        i: {
            "path": str(paths[i]),
            "true_label": int(labels[i]),
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "n_wrong": 0,
            "conf_sum": 0.0,
            "pred_conf_sum": 0.0,
            "details": [],
        }
        for i in range(n_samples)
    }
    path_to_idx = {str(p): i for i, p in enumerate(paths)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 2)
    run_id = 0

    for rep in range(args.n_repeats):
        rep_seed = args.seed + rep * 1000
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=rep_seed)

        for fold, (trainval_idx, test_idx) in enumerate(skf.split(paths, labels), start=1):
            run_id += 1
            print(f"\n----- Repeat {rep+1}/{args.n_repeats} | Fold {fold}/{args.n_splits} (run {run_id}/{total_rounds}) -----")

            # 从 trainval 再划 train / val，保证本折有明确 val
            tr_idx, va_idx = train_test_split(
                trainval_idx,
                test_size=args.val_ratio,
                stratify=labels[trainval_idx],
                random_state=rep_seed + fold,
            )

            # 记录每个样本本轮角色
            for i in tr_idx:
                stats[i]["n_train"] += 1
            for i in va_idx:
                stats[i]["n_val"] += 1
            for i in test_idx:
                stats[i]["n_test"] += 1

            train_loader = make_loader(
                paths, labels, tr_idx,
                data_cfg["sample_rate"], data_cfg["max_duration"],
                batch_size, True, num_workers,
            )
            val_loader = make_loader(
                paths, labels, va_idx,
                data_cfg["sample_rate"], data_cfg["max_duration"],
                batch_size, False, num_workers,
            )

            cnt = Counter(labels[tr_idx].tolist())
            total = sum(cnt.values())
            weights = [total / (num_classes * cnt.get(i, 1)) for i in range(num_classes)]
            class_weight = torch.tensor(weights, dtype=torch.float32)
            print(f"  train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)} weight={weights}")

            fold_out = ROOT / train_cfg.get("output_dir", "outputs/models") / f"rep{rep+1}_fold{fold}"
            fold_out.mkdir(parents=True, exist_ok=True)
            cfg_fold = json.loads(
                json.dumps(cfg, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            )
            cfg_fold["training"]["output_dir"] = str(fold_out)
            cfg_fold["training"]["log_dir"] = str(
                ROOT / train_cfg.get("log_dir", "outputs/logs") / f"rep{rep+1}_fold{fold}"
            )
            cfg_fold["model"]["class_weight"] = class_weight

            model = build_model(cfg_fold)
            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_fold,
                device=device,
            )

            # 仅在本折 test 上统计（未见样本）
            results = predict_all(
                model,
                [paths[i] for i in test_idx],
                labels[test_idx].tolist(),
                data_cfg["sample_rate"],
                data_cfg["max_duration"],
                batch_size,
                device,
            )

            for path, pred, prob, lab in zip(
                results["paths"], results["preds"], results["probs"], results["labels"]
            ):
                idx = path_to_idx[path]
                conf_true = float(prob[int(lab)])
                conf_pred = float(prob[int(pred)])
                wrong = int(pred != lab)
                stats[idx]["n_wrong"] += wrong
                stats[idx]["conf_sum"] += conf_true
                stats[idx]["pred_conf_sum"] += conf_pred
                stats[idx]["details"].append({
                    "rep": rep + 1,
                    "fold": fold,
                    "pred": int(pred),
                    "prob_true": conf_true,
                    "prob_pred": conf_pred,
                    "wrong": wrong,
                })

            acc = (results["preds"] == results["labels"]).mean()
            print(f"  test accuracy: {acc:.4f}")

    # ---------- 汇总 ----------
    rows = []
    coverage_ok = 0
    for i, s in stats.items():
        n_test = s["n_test"]
        if n_test == 0:
            continue
        err_rate = s["n_wrong"] / n_test
        avg_conf_true = s["conf_sum"] / n_test
        avg_conf_pred = s["pred_conf_sum"] / n_test
        difficulty = err_rate * 0.6 + (1.0 - avg_conf_true) * 0.4
        covered = s["n_train"] > 0 and s["n_val"] > 0 and s["n_test"] > 0
        if covered:
            coverage_ok += 1
        rows.append({
            "path": s["path"],
            "true_label": s["true_label"],
            "label_name": {0: "good", 1: "bad", 2: "critical"}.get(s["true_label"], str(s["true_label"])),
            "n_train": s["n_train"],
            "n_val": s["n_val"],
            "n_test": s["n_test"],
            "n_wrong": s["n_wrong"],
            "error_rate": round(err_rate, 4),
            "avg_conf_true": round(avg_conf_true, 4),
            "avg_conf_pred": round(avg_conf_pred, 4),
            "difficulty": round(difficulty, 4),
            "fully_covered": covered,  # 是否 train/val/test 都出现过
        })

    hard_df = pd.DataFrame(rows).sort_values(
        ["difficulty", "error_rate"], ascending=[False, False]
    )

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "hard_samples_ranked.csv"
    hard_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 难样本：测试至少 2 次，且错误率高或置信度低
    hard_only = hard_df[
        (hard_df["n_test"] >= max(1, args.n_repeats))
        & ((hard_df["error_rate"] >= 0.5) | (hard_df["avg_conf_true"] < 0.5))
    ].copy()
    hard_only_path = out_dir / "hard_samples_filtered.csv"
    hard_only.to_csv(hard_only_path, index=False, encoding="utf-8-sig")

    # 未完整覆盖的样本（理论上 n_repeats>=1 且 n_splits>=2 时应很少）
    not_covered = hard_df[~hard_df["fully_covered"]]
    not_covered.to_csv(out_dir / "not_fully_covered.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "total_runs": total_rounds,
        "n_samples": n_samples,
        "n_fully_covered_train_val_test": coverage_ok,
        "n_hard_filtered": int(len(hard_only)),
        "by_label": hard_only.groupby("label_name").size().to_dict() if len(hard_only) else {},
        "top20": hard_df.head(20).to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    detail = {str(paths[i]): stats[i]["details"] for i in range(n_samples)}
    with open(out_dir / "per_sample_fold_details.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"train/val/test 都出现过的样本数: {coverage_ok} / {n_samples}")
    print(f"全部排序: {csv_path}")
    print(f"难样本  : {hard_only_path} ({len(hard_only)} 条)")
    print(f"未全覆盖: {out_dir / 'not_fully_covered.csv'} ({len(not_covered)} 条)")
    print("\nTop-15 最难样本:")
    cols = ["path", "label_name", "n_test", "n_wrong", "error_rate", "avg_conf_true", "difficulty", "fully_covered"]
    print(hard_df.head(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
