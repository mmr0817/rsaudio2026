"""推理接口（兼容 fairseq 模型）"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from src.models.classifier import SpeakerQualityClassifier, build_model
from src.utils.audio import load_audio, pad_or_truncate


LABEL_CN = {0: "好", 1: "坏", 2: "临界"}
LABEL_EN = {0: "good", 1: "bad", 2: "critical"}


class Predictor:
    """单条 / 批量推理"""

    def __init__(
        self,
        model: SpeakerQualityClassifier,
        sample_rate: int = 16000,
        max_duration: float = 2.5,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_file(
        self, path: Union[str, Path], return_probs: bool = True
    ) -> Dict:
        """预测单个 wav 文件"""
        waveform, _ = load_audio(path, target_sr=self.sample_rate, mono=True)
        peak = np.max(np.abs(waveform)) + 1e-8
        waveform = waveform / peak
        waveform = pad_or_truncate(waveform, self.max_length)

        input_values = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        outputs = self.model(input_values=input_values)

        logits = outputs["logits"][0]
        probs = outputs["probs"][0].cpu().numpy()
        pred_id = int(logits.argmax().item())

        result = {
            "path": str(path),
            "prediction": LABEL_EN[pred_id],
            "prediction_cn": LABEL_CN[pred_id],
            "label_id": pred_id,
            "confidence": float(probs[pred_id]),
        }
        if return_probs:
            result["probabilities"] = {
                LABEL_EN[i]: float(probs[i]) for i in range(len(probs))
            }
            result["probabilities_cn"] = {
                LABEL_CN[i]: float(probs[i]) for i in range(len(probs))
            }
        return result

    @torch.no_grad()
    def predict_batch(
        self, paths: List[Union[str, Path]], return_probs: bool = True
    ) -> List[Dict]:
        return [self.predict_file(p, return_probs=return_probs) for p in paths]

    def predict_waveform(self, waveform: np.ndarray) -> Dict:
        """直接对 numpy 波形预测（已是目标采样率）"""
        peak = np.max(np.abs(waveform)) + 1e-8
        waveform = (waveform / peak).astype(np.float32)
        waveform = pad_or_truncate(waveform, self.max_length)
        input_values = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        outputs = self.model(input_values=input_values)
        probs = outputs["probs"][0].cpu().numpy()
        pred_id = int(outputs["logits"][0].argmax().item())
        return {
            "prediction": LABEL_EN[pred_id],
            "prediction_cn": LABEL_CN[pred_id],
            "label_id": pred_id,
            "confidence": float(probs[pred_id]),
            "probabilities": {LABEL_EN[i]: float(probs[i]) for i in range(len(probs))},
        }


def load_predictor(
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    device: str = "auto",
) -> Predictor:
    """
    从保存的 checkpoint 加载预测器。

    model_path: 指向 best/ 目录或包含 pytorch_model.bin 的目录
    """
    model_path = Path(model_path)
    if (model_path / "pytorch_model.bin").exists():
        ckpt_path = model_path / "pytorch_model.bin"
        cfg_path = model_path / "config.json"
    else:
        ckpt_path = model_path
        cfg_path = None

    # 加载配置
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif cfg_path and cfg_path.exists():
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "model": {
                "backbone": "pretrained/wav2vec_small.pt",
                "num_labels": 3,
                "freeze_feature_encoder": True,
                "pooling": "mean",
                "dropout": 0.1,
                "hidden_size": 256,
            },
            "data": {
                "sample_rate": 16000,
                "max_duration": 2.5,
                "label_map": {"good": 0, "bad": 1, "critical": 2},
            },
        }

    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        # 只加载分类头等可训练部分（backbone 权重已在 build_model 中从 fairseq 加载）
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in state["model_state_dict"].items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    data_cfg = cfg.get("data", {})
    return Predictor(
        model=model,
        sample_rate=data_cfg.get("sample_rate", 16000),
        max_duration=data_cfg.get("max_duration", 2.5),
        device=dev,
    )
