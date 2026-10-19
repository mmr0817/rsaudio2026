"""音频加载与预处理工具"""

from pathlib import Path
from typing import Tuple, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch


def load_audio(
    path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    加载 wav 文件并重采样到目标采样率。

    Args:
        path: 音频文件路径
        target_sr: 目标采样率（wav2vec2 默认 16kHz）
        mono: 是否转为单声道

    Returns:
        waveform (float32, shape [T]), sample_rate
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # soundfile 更快更稳定，librosa 作为后备
    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if waveform.ndim > 1 and mono:
            waveform = waveform.mean(axis=1)
    except Exception:
        waveform, sr = librosa.load(str(path), sr=None, mono=mono)

    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 保证 float32 且范围大致在 [-1, 1]
    waveform = waveform.astype(np.float32)
    return waveform, sr


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """重采样音频"""
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr).astype(
        np.float32
    )


def normalize_audio(
    waveform: np.ndarray,
    target_db: Optional[float] = -20.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    峰值归一化或 RMS 归一化到目标 dB。
    扫频信号建议使用峰值归一化以保留动态范围。
    """
    peak = np.max(np.abs(waveform)) + eps
    waveform = waveform / peak

    if target_db is not None:
        # 可选 RMS 归一化
        rms = np.sqrt(np.mean(waveform**2) + eps)
        target_rms = 10 ** (target_db / 20)
        waveform = waveform * (target_rms / rms)
        # 防止削波
        peak = np.max(np.abs(waveform)) + eps
        if peak > 1.0:
            waveform = waveform / peak

    return waveform.astype(np.float32)


def pad_or_truncate(
    waveform: np.ndarray,
    target_length: int,
    mode: str = "constant",
) -> np.ndarray:
    """将音频 pad 或截断到固定长度（采样点数）"""
    length = len(waveform)
    if length == target_length:
        return waveform
    if length > target_length:
        # 居中截断
        start = (length - target_length) // 2
        return waveform[start : start + target_length]
    # pad
    pad_total = target_length - length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(waveform, (pad_left, pad_right), mode=mode)


def to_tensor(waveform: np.ndarray) -> torch.Tensor:
    """numpy -> torch tensor (1D)"""
    return torch.from_numpy(waveform).float()