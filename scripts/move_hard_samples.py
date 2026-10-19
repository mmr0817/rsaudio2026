#!/usr/bin/env python3
"""
将 hard_samples_filtered.csv（或 ranked）中的样本
按 good/bad 移动到 outputs/hard_samples/hard_wav_files/

用法:
  python scripts/move_hard_samples.py
  python scripts/move_hard_samples.py --csv outputs/hard_samples/hard_samples_filtered.csv
  python scripts/move_hard_samples.py --copy   # 只复制，不从原位置删除
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Move hard samples into hard_wav_files/{good,bad}")
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV 路径（默认自动找 filtered / ranked）",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs/hard_samples/hard_wav_files",
        help="输出目录",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="复制而不是移动（保留 data/raw 中的原文件）",
    )
    return p.parse_args()


def resolve_csv(user_csv: str | None) -> Path:
    if user_csv:
        p = Path(user_csv)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"找不到 CSV: {p}")
        return p

    candidates = [
        ROOT / "outputs/hard_samples/hard_samples_filtered.csv",
        ROOT / "outputs/hard_samples/hard_samples_ranked.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "找不到 hard_samples_filtered.csv 或 hard_samples_ranked.csv，"
        "请用 --csv 指定路径"
    )


def infer_label(row: dict, src: Path) -> str:
    label = (row.get("label_name") or "").strip().lower()
    if label in ("good", "bad"):
        return label

    true_label = str(row.get("true_label", "")).strip()
    if true_label == "0":
        return "good"
    if true_label == "1":
        return "bad"

    path_str = str(src).replace("\\", "/")
    for name, mapped in (
        ("good", "good"),
        ("bad", "bad"),
        ("好", "good"),
        ("坏", "bad"),
    ):
        if f"/{name}/" in path_str or src.parent.name == name:
            return mapped
    return "unknown"


def main():
    args = parse_args()
    csv_path = resolve_csv(args.csv)

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    (out_root / "good").mkdir(parents=True, exist_ok=True)
    (out_root / "bad").mkdir(parents=True, exist_ok=True)

    print(f"CSV    : {csv_path}")
    print(f"输出目录: {out_root}")
    print(f"模式  : {'复制' if args.copy else '移动'}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "path" not in fields:
            raise SystemExit(f"CSV 缺少 path 列，当前列: {fields}")
        for row in reader:
            rows.append(row)

    moved, missing = 0, []
    log = []

    for row in rows:
        src = Path(row["path"].strip())
        label = infer_label(row, src)
        dest_dir = out_root / (label if label in ("good", "bad") else "unknown")
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            missing.append(str(src))
            log.append({"src": str(src), "dest": "", "status": "missing", "label": label})
            continue

        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{src.parent.name}_{src.name}"

        if args.copy:
            shutil.copy2(src, dest)
            status = "copied"
        else:
            shutil.move(str(src), str(dest))
            status = "moved"

        moved += 1
        log.append({"src": str(src), "dest": str(dest), "status": status, "label": label})

    log_path = out_root / "move_log.csv"
    with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src", "dest", "status", "label"])
        w.writeheader()
        w.writerows(log)

    print(f"完成: {moved} 个文件")
    print(f"缺失: {len(missing)}")
    for m in missing[:10]:
        print(f"  - {m}")
    print(f"日志: {log_path}")
    for sub in sorted(out_root.iterdir()):
        if sub.is_dir():
            n = len(list(sub.glob("*.wav"))) + len(list(sub.glob("*.WAV")))
            print(f"  {sub.name}/ : {n} 个 wav")


if __name__ == "__main__":
    main()
    sys.exit(0)
