#!/usr/bin/env python3
"""
简单 Gradio 演示界面（可选）

用法:
    python scripts/demo_gradio.py --model outputs/models/best
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/best")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Please install gradio: pip install gradio")
        return

    from src.inference.predict import load_predictor

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    predictor = load_predictor(model_path)

    def predict(audio):
        if audio is None:
            return "请上传音频", None
        # gradio 返回 (sr, numpy) 或 文件路径
        if isinstance(audio, tuple):
            sr, wav = audio
            import numpy as np
            from src.utils.audio import resample_audio

            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != predictor.sample_rate:
                wav = resample_audio(wav.astype("float32"), sr, predictor.sample_rate)
            result = predictor.predict_waveform(wav.astype("float32"))
        else:
            result = predictor.predict_file(audio)

        label = f"{result['prediction_cn']} ({result['prediction']})"
        conf = f"{result['confidence']*100:.1f}%"
        probs = result.get("probabilities_cn", result.get("probabilities", {}))
        detail = "\n".join(f"  {k}: {v*100:.1f}%" for k, v in probs.items())
        text = f"预测结果: {label}\n置信度: {conf}\n\n概率分布:\n{detail}"
        return text, result

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Audio(type="filepath", label="上传扫频 WAV 文件"),
        outputs=[
            gr.Textbox(label="鉴别结果"),
            gr.JSON(label="详细信息"),
        ],
        title="音响扫频质量鉴别系统",
        description="上传约 2.1 秒的扫频音频，模型将判断为【好 / 坏 / 临界】",
        examples=None,
    )
    iface.launch(share=args.share)


if __name__ == "__main__":
    main()