"""
音响质量鉴别模型
前端: torchaudio Wav2Vec2
后端: 分类鉴别器 + class weight
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        scores = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(hidden_states * weights, dim=1)


def load_wav2vec_backbone(checkpoint_path: Optional[str] = None):
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if ckpt.exists() and ckpt.suffix == ".pt":
            try:
                import fairseq
                from torchaudio.models.wav2vec2.utils import import_fairseq_model
                models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(ckpt)])
                model = import_fairseq_model(models[0])
                print(f"[Info] Loaded local fairseq checkpoint: {ckpt}")
                return model, 768
            except Exception as e:
                print(f"[Warning] Failed to load local fairseq ({e}), fallback to torchaudio")

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    print("[Info] Using torchaudio.pipelines.WAV2VEC2_BASE")
    return model, 768


class SpeakerQualityClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        num_labels: int = 2,
        freeze_feature_extractor: bool = True,
        freeze_base_model: bool = False,
        pooling: str = "mean",
        dropout: float = 0.1,
        hidden_size: int = 256,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pooling_type = pooling

        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None

        self.wav2vec2, backbone_dim = load_wav2vec_backbone(checkpoint_path)
        self.backbone_dim = backbone_dim

        if freeze_feature_extractor:
            if hasattr(self.wav2vec2, "feature_extractor"):
                for p in self.wav2vec2.feature_extractor.parameters():
                    p.requires_grad = False
        if freeze_base_model:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

        self.pooler = AttentionPooling(backbone_dim) if pooling == "attention" else None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.id2label = id2label or {0: "good", 1: "bad"}
        self.label2id = label2id or {"good": 0, "bad": 1}

    def extract_features(self, input_values: torch.Tensor) -> torch.Tensor:
        features, _ = self.wav2vec2.extract_features(input_values)
        return features[-1]

    def forward(self, input_values, attention_mask=None, labels=None, return_dict=True):
        hidden = self.extract_features(input_values)
        if self.pooling_type == "mean":
            pooled = hidden.mean(dim=1)
        elif self.pooling_type == "max":
            pooled = hidden.max(dim=1).values
        else:
            pooled = self.pooler(hidden)

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        result = {"logits": logits, "probs": probs, "features": pooled}

        if labels is not None:
            weight = getattr(self, "class_weight", None)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            result["loss"] = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return result


def build_model(cfg: dict) -> SpeakerQualityClassifier:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    label_map = data_cfg.get("label_map", {"good": 0, "bad": 1})
    id2label = {v: k for k, v in label_map.items()}

    backbone = model_cfg.get("backbone", None)
    if backbone is not None:
        p = Path(backbone)
        if not p.is_absolute():
            root = Path(__file__).resolve().parents[2]
            candidate = root / backbone
            backbone = str(candidate) if candidate.exists() else None
            if backbone is None:
                print("[Warning] backbone not found, use torchaudio default")

    return SpeakerQualityClassifier(
        checkpoint_path=backbone,
        num_labels=model_cfg.get("num_labels", 2),
        freeze_feature_extractor=model_cfg.get("freeze_feature_encoder", True),
        freeze_base_model=model_cfg.get("freeze_base_model", False),
        pooling=model_cfg.get("pooling", "mean"),
        dropout=model_cfg.get("dropout", 0.1),
        hidden_size=model_cfg.get("hidden_size", 256),
        id2label=id2label,
        label2id=label_map,
        class_weight=model_cfg.get("class_weight", None),
    )
