"""音频预处理（fairseq 版本不需要 HF FeatureExtractor，主要做 pad/normalize）"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from src.utils.audio import pad_or_truncate


class AudioPreprocessor:
    """
    简单预处理：归一化 + 固定长度。
    fairseq wav2vec2 直接吃原始波形。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 2.5,
        do_normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        self.do_normalize = do_normalize

    def __call__(
        self,
        waveforms: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        sampling_rate: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(waveforms, torch.Tensor):
            waveforms = waveforms.cpu().numpy()

        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 1:
            waveforms = [waveforms]
        elif isinstance(waveforms, list):
            waveforms = [np.asarray(w, dtype=np.float32) for w in waveforms]
        else:
            waveforms = [np.asarray(w, dtype=np.float32) for w in waveforms]

        processed = []
        for w in waveforms:
            if self.do_normalize:
                peak = np.max(np.abs(w)) + 1e-8
                w = w / peak
            w = pad_or_truncate(w, self.max_length)
            processed.append(w)

        input_values = torch.from_numpy(np.stack(processed)).float()
        return {"input_values": input_values}

    def process_single(
        self,
        waveform: np.ndarray,
        sampling_rate: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.__call__(waveform, sampling_rate=sampling_rate)
