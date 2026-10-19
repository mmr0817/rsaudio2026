# Speaker Quality Classifier  
## 音响扫频音频质量鉴别系统

基于 **fairseq Wav2Vec2**（本地模型）特征提取 + 自定义鉴别器的三分类模型。

| 标签 | 中文 | 说明 |
|------|------|------|
| `good` | 好 | 响应正常，频响平整 |
| `bad` | 坏 | 明显问题（谐振、失真等）|
| `critical` | 临界 | 临界状态 |

输入为约 **2.1 秒** 的扫频 WAV 文件。

---

## 代码位置

项目根目录：

```
/home/workdir/artifacts/speaker_quality_classifier/
```

完整结构：

```
speaker_quality_classifier/
├── config.yaml                 # 配置文件
├── requirements.txt
├── README.md
├── pretrained/                 # 下载的 fairseq 模型放这里
│   └── wav2vec_small.pt
├── data/raw/{good,bad,critical}/
├── src/
│   ├── data/                   # 数据集
│   ├── models/classifier.py    # fairseq 特征提取 + 鉴别器
│   ├── training/trainer.py     # 训练（含 TensorBoard）
│   ├── inference/predict.py
│   └── utils/
├── scripts/
│   ├── download_wav2vec.py     # 下载 fairseq 模型
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── demo_gradio.py
└── outputs/
    ├── models/best/
    └── logs/tensorboard/       # TensorBoard 日志
```

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/workdir/artifacts/speaker_quality_classifier
pip install -r requirements.txt
```

> fairseq 安装若失败，可尝试：
> ```bash
> source .venv/bin/activate 
> pip install fairseq --no-deps
> pip install omegaconf hydra-core bitarray cython

> ```
> 或从源码：`pip install git+https://github.com/facebookresearch/fairseq.git`



### 2. 下载 fairseq Wav2Vec2 模型到本地

```bash
python scripts/download_wav2vec.py
# 默认下载 base 模型到 pretrained/wav2vec_small.pt （约 360MB）
```

下载完成后 `config.yaml` 中已配置好路径：
```yaml
model:
  backbone: "pretrained/wav2vec_small.pt"
```

### 3. 准备数据

```
data/raw/
├── good/          # 好样本 *.wav
├── bad/           # 坏样本
└── critical/      # 临界样本
```

### 4. 训练（自动开启 TensorBoard）

```bash
python scripts/train.py --config config.yaml
```

训练开始后会打印 TensorBoard 日志路径，例如：

```
TensorBoard log dir: .../outputs/logs/tensorboard
  启动命令: tensorboard --logdir .../outputs/logs/tensorboard
```

**另开终端启动 TensorBoard：**

```bash
tensorboard --logdir outputs/logs/tensorboard --port 6006
```

浏览器访问 `http://localhost:6006` 即可实时查看：

- `train/loss_step`、`train/loss_epoch`
- `train/lr`
- `val/loss`、`val/accuracy`、`val/f1`
- 各类别 F1（`val/f1_good` 等）

### 5. 推理

```bash
python scripts/predict.py --model outputs/models/best --audio your.wav
```

---

## 模型架构

```
Input WAV (≈2.1s, 16kHz)
       │
       ▼
┌──────────────────────────┐
│  fairseq Wav2Vec2        │  ← 本地 pretrained/wav2vec_small.pt
│  (Feature Extractor +    │
│   Transformer Encoder)   │
└──────────┬───────────────┘
           │ [B, T', 768]
           ▼
┌──────────────────────────┐
│  Pooling (mean/max/attn) │
└──────────┬───────────────┘
           │ [B, 768]
           ▼
┌──────────────────────────┐
│  Discriminator Head      │  Linear → GELU → Dropout → Linear
└──────────┬───────────────┘
           ▼
    Logits [B, 3] → {good, bad, critical}
```

---

## 主要命令汇总

| 操作 | 命令 |
|------|------|
| 下载模型 | `python scripts/download_wav2vec.py` |
| 训练 | `python scripts/train.py --config config.yaml` |
| TensorBoard | `tensorboard --logdir outputs/logs/tensorboard --port 6006` |
| 评估 | `python scripts/evaluate.py --model outputs/models/best` |
| 推理 | `python scripts/predict.py --model outputs/models/best --audio xx.wav` |

---

## 注意事项

1. **必须先下载模型**：`python scripts/download_wav2vec.py`
2. 所有音频会自动重采样到 **16kHz**
3. 数据少时建议保持 `freeze_feature_encoder: true`
4. TensorBoard 日志在 `outputs/logs/tensorboard/`，训练结束后也可回看

---

MIT License
