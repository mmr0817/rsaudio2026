"""训练与评估逻辑（支持 TensorBoard）"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.classifier import SpeakerQualityClassifier
from src.utils.seed import set_seed


def compute_metrics(
    preds: np.ndarray, labels: np.ndarray, label_names: Optional[list] = None
) -> Dict[str, float]:
    """计算分类指标（自动适配实际出现的类别数）"""
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    # 只使用实际出现的类别
    unique_labels = sorted(set(labels.tolist()) | set(preds.tolist()))
    if label_names is not None:
        # 按 unique_labels 过滤名称
        names = [label_names[i] if i < len(label_names) else str(i) for i in unique_labels]
    else:
        names = [str(i) for i in unique_labels]

    report = classification_report(
        labels,
        preds,
        labels=unique_labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "f1": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
    }


@torch.no_grad()
def evaluate_model(
    model: SpeakerQualityClassifier,
    dataloader: DataLoader,
    device: torch.device,
    label_names: Optional[list] = None,
) -> Dict:
    """在给定 DataLoader 上评估"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_values=input_values, labels=labels)
        loss = outputs.get("loss")
        if loss is not None:
            total_loss += loss.item()
            n_batches += 1

        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_preds), np.array(all_labels), label_names=label_names
    )
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    return metrics


def train_model(
    model: SpeakerQualityClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: Optional[torch.device] = None,
) -> SpeakerQualityClassifier:
    """
    完整训练循环，集成 TensorBoard 监控。

    Returns:
        训练后的模型（已加载最佳权重）
    """
    train_cfg = cfg.get("training", {})
    set_seed(cfg.get("project", {}).get("seed", 42))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on {device}")

    # 优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.get("learning_rate", 3e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    num_epochs = train_cfg.get("num_epochs", 20)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(train_loader) // grad_accum) * num_epochs
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(total_steps * warmup_ratio)

    # 简单线性 warmup + 衰减
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(train_cfg.get("output_dir", "outputs/models"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(train_cfg.get("log_dir", "outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- TensorBoard ----------
    tb_dir = log_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard log dir: {tb_dir.resolve()}")
    print(f"  启动命令: tensorboard --logdir {tb_dir.resolve()}")

    best_metric = -float("inf")
    patience = train_cfg.get("early_stopping_patience", 5)
    patience_counter = 0
    metric_name = train_cfg.get("metric_for_best_model", "f1")
    greater_is_better = train_cfg.get("greater_is_better", True)
    history = []

    use_amp = train_cfg.get("fp16", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    label_names = list(cfg.get("data", {}).get("label_map", {}).keys())
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for step, batch in enumerate(pbar):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs["loss"] / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # TensorBoard: 每 step 记录 loss 和 lr
                writer.add_scalar("train/loss_step", loss.item() * grad_accum, global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            pbar.set_postfix({"loss": f"{epoch_loss / n_batches:.4f}"})

        # 验证
        val_metrics = evaluate_model(model, val_loader, device, label_names=label_names)
        avg_train_loss = epoch_loss / max(n_batches, 1)

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }
        history.append(log_entry)

        # TensorBoard: epoch 级别指标
        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        writer.add_scalar("val/f1_weighted", val_metrics["f1_weighted"], epoch)

        # 每个类别的 F1
        if "report" in val_metrics:
            for lbl in label_names:
                if lbl in val_metrics["report"]:
                    writer.add_scalar(
                        f"val/f1_{lbl}",
                        val_metrics["report"][lbl]["f1-score"],
                        epoch,
                    )

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        # 保存最佳
        current = val_metrics.get(metric_name, val_metrics["f1"])
        improved = (
            (current > best_metric) if greater_is_better else (current < best_metric)
        )
        if improved:
            best_metric = current
            patience_counter = 0
            best_path = output_dir / "best"
            best_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path / "pytorch_model.bin",
            )
            save_cfg = json.loads(json.dumps(cfg, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
            with open(best_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(save_cfg, f, indent=2, ensure_ascii=False)
            print(f"  → Saved best model (metric={current:.4f})")
            writer.add_scalar("best/metric", current, epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 每 epoch 保存 checkpoint
        ckpt_path = output_dir / f"checkpoint-epoch{epoch}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "pytorch_model.bin")

    writer.close()

    # 保存训练历史
    with open(log_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 加载最佳权重返回
    best_ckpt = torch.load(
        output_dir / "best" / "pytorch_model.bin", map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded best model from epoch {best_ckpt['epoch']}")
    print(f"\nTensorBoard 查看训练曲线:")
    print(f"  tensorboard --logdir {tb_dir.resolve()} --port 6006")
    return model
