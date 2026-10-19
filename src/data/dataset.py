"""数据集与 DataLoader 构建"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.utils.audio import load_audio, pad_or_truncate


LABEL_MAP = {"good": 0, "bad": 1, "critical": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_CN = {"good": "好", "bad": "坏", "critical": "临界"}


class SpeakerQualityDataset(Dataset):
    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        labels: List[int],
        sample_rate: int = 16000,
        max_duration: float = 2.5,
        normalize: bool = True,
    ):
        assert len(file_paths) == len(labels), "paths and labels length mismatch"
        self.file_paths = [Path(p) for p in file_paths]
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = load_audio(path, target_sr=self.sample_rate, mono=True)

        if self.normalize:
            peak = np.max(np.abs(waveform)) + 1e-8
            waveform = waveform / peak

        waveform = pad_or_truncate(waveform, self.max_length)

        return {
            "input_values": torch.from_numpy(waveform).float(),
            "labels": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    input_values = torch.stack([b["input_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    paths = [b["path"] for b in batch]
    return {
        "input_values": input_values,
        "labels": labels,
        "paths": paths,
    }


def prepare_dataset_from_folders(
    root_dir: Union[str, Path],
    label_map: Optional[Dict[str, int]] = None,
    extensions: Tuple[str, ...] = (".wav", ".WAV"),
) -> Tuple[List[Path], List[int], pd.DataFrame]:
    root_dir = Path(root_dir)
    label_map = label_map or LABEL_MAP

    records = []
    for label_name, label_id in label_map.items():
        folder = root_dir / label_name
        if not folder.exists():
            cn_map = {"good": "好", "bad": "坏", "critical": "临界"}
            folder = root_dir / cn_map.get(label_name, label_name)
        if not folder.exists():
            print(f"[Warning] Folder not found: {folder}")
            continue

        for f in folder.rglob("*"):
            if f.suffix in extensions:
                records.append({
                    "path": str(f.resolve()),
                    "label_name": label_name,
                    "label": label_id,
                })

    if not records:
        raise FileNotFoundError(
            f"No wav files found under {root_dir}. "
            "Please organize as data/raw/{{good,bad,critical}}/*.wav"
        )

    df = pd.DataFrame(records)
    print(f"Found {len(df)} samples:")
    print(df["label_name"].value_counts().to_string())
    return [Path(p) for p in df["path"]], df["label"].tolist(), df


def create_dataloaders(
    root_dir: Union[str, Path],
    sample_rate: int = 16000,
    max_duration: float = 2.5,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    paths, labels, df = prepare_dataset_from_folders(root_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - relative_val), stratify=temp_labels, random_state=seed,
    )

    datasets = {
        "train": SpeakerQualityDataset(train_paths, train_labels, sample_rate, max_duration),
        "val": SpeakerQualityDataset(val_paths, val_labels, sample_rate, max_duration),
        "test": SpeakerQualityDataset(test_paths, test_labels, sample_rate, max_duration),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        print(f"{split}: {len(ds)} samples")

    return loaders
