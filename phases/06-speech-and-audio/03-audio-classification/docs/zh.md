# Audio Classification — From k-NN on MFCCs to AST and BEATs / 音频分类：从 MFCC 上的 k-NN 到 AST 与 BEATs

> 从 “dog barking vs siren” 到 “这是什么语言”，本质上都是 audio classification。特征是 mels。架构每十年换一轮。评估指标始终离不开 AUC、F1 和 per-class recall。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从 k-NN on MFCCs、2D CNN、AST、BEATs 到 Whisper encoder，梳理音频分类 baseline 的演进
- 构建 MFCC summary embedding 和 cosine k-NN classifier
- 解释 class imbalance、domain shift、label noise 对音频分类的影响
- 为不同数据规模和部署约束选择合适的 backbone、augmentation 和 evaluation metric

## The Problem / 问题

你拿到一段 10 秒音频，想知道：“它是什么？”可能是城市声音（警笛、钻孔、狗叫），speech command（yes/no/stop），language ID（en/es/ar），speaker emotion（angry/neutral），也可能是环境声音（indoor/outdoor、babble）。这些都是 *audio classification*。到 2026 年，baseline 架构已经成熟：log-mel → CNN 或 Transformer → softmax。

真正的难点不是网络，而是数据。音频数据集往往 class imbalance 严重，domain shift 很强（clean vs noisy），label noise 也很大（到底是谁决定 “urban babble” 和 “restaurant noise” 的边界？）。80% 的工作是 curation、augmentation 和 evaluation，而不是把 CNN 换成 Transformer。

## The Concept / 概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**k-NN on MFCCs (the 1990s baseline) / MFCC 上的 k-NN（1990 年代 baseline）。** 对每个 clip flatten MFCC，和一个带标签样本库计算 cosine similarity，返回 top K 的 majority vote。在干净的小数据集上（Speech Commands、ESC-50）意外地强，而且不需要 GPU。

**2D CNN on log-mels (2015-2019) / log-mels 上的 2D CNN。** 把 `(T, n_mels)` log-mel 当成图片。套 ResNet-18 或 VGG-style 网络。对 time axis 做 global mean pool，再对类别做 softmax。它仍然是多数 2026 kaggle competition 里的 baseline。

**Audio Spectrogram Transformer, AST (2021-2024)。** 把 log-mel patchify（例如 16×16 patches），加 position embeddings，送入 ViT。它曾是 AudioSet 上 supervised learning 的 state of the art（mAP 0.485）。

**BEATs and WavLM-base (2024-2026)。** 在数百万小时音频上做 self-supervised pretraining。对你的任务 fine-tune 时，只需要原来 1–10% 的 supervised data。到 2026 年，这是非语音音频任务的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高 1–2 mAP，同时只用 1/4 compute。

**Whisper-encoder as a frozen backbone (2024) / 把 Whisper encoder 当 frozen backbone。** 取 Whisper 的 encoder，丢掉 decoder，接一个 linear classifier。在 language ID 和简单事件分类上接近 SOTA，几乎不需要 audio augmentation。这是一个“免费午餐” baseline。

### Class imbalance is the real challenge / Class imbalance 才是真正挑战

ESC-50：50 类，每类 40 个 clips，均衡、简单。UrbanSound8K：10 类，10:1 不均衡。AudioSet：632 类，长尾达到 100,000:1。有效技术包括：

- 训练时做 balanced sampling（评估时不要）。
- Mixup：把两个 clips（以及它们的 labels）线性插值，作为 augmentation。
- SpecAugment：随机 mask time 和 frequency bands。简单，但关键。

### Evaluation / 评估

- Multiclass exclusive（Speech Commands）：top-1 accuracy、top-5 accuracy。
- Multiclass multi-label（AudioSet、UrbanSound-style）：mean average precision (mAP)。
- 严重不均衡：per-class recall + macro F1。

你应该知道的 2026 年数字：

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## Build It / 动手构建

### Step 1: featurize / 第 1 步：提取特征

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2: fixed-length summary / 第 2 步：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但很强：跨时间计算 mean + variance，可以把 13-coef MFCC 变成 26 维固定 embedding。它运行很快。直到 2017 年前后，它在 ESC-50 上仍能击败一些当时的 state-of-the-art NN baselines。

### Step 3: k-NN / 第 3 步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4: upgrade to CNN on log-mels / 第 4 步：升级到 log-mels 上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

约 3M parameters。在单张 RTX 4090 上训练 ESC-50 约 10 分钟。准确率可达 80%+。

### Step 5: the 2026 default — fine-tune BEATs / 第 5 步：2026 默认做法：fine-tune BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs 可通过 `beats` library 使用 `microsoft/BEATs-base`；transformers API 的形态相同。

## Use It / 应用它

2026 年的 stack：

| Situation | Start with |
|-----------|-----------|
| Tiny dataset (<1000 clips) | MFCC means 上的 k-NN（你的 baseline）+ audio augmentation |
| Medium dataset (1K–100K) | BEATs 或 AST fine-tune |
| Large dataset (>100K) | 从零训练，或 fine-tune Whisper-encoder |
| Real-time, edge | 40-MFCC CNN，quantized to int8（KWS-style） |
| Multi-label (AudioSet) | BEATs-iter3 + BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID, SpeechBrain VoxLingua107 baseline |

决策规则：**从 frozen backbone 开始，而不是从新模型开始。** Fine-tuning 一个 BEATs head，能在几小时而不是几周内拿到 95% 的 SOTA。

## Ship It / 交付它

保存为 `outputs/skill-classifier-designer.md`。为给定音频分类任务选择 architecture、augmentations、class-balance strategy 和 eval metric。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会在一个 4 类 synthetic dataset（不同 pitch 的纯音）上训练 k-NN MFCC baseline。报告 confusion matrix。
2. **Medium / 中等。** 用 [mean, var, skew, kurtosis] 替换 `summarize`。在同一个 synthetic dataset 上，4-moment pooling 是否优于 mean+var？
3. **Hard / 困难。** 使用 `torchaudio`，在 ESC-50 fold 1 上训练 2D CNN。报告 5-fold cross-validation accuracy。加入 SpecAugment（time mask = 20, freq mask = 10）并报告 delta。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频界的 ImageNet | Google 的 2M-clip、632-class 弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类 benchmark | 50 类 × 每类 40 段环境声音。 |
| AST | Audio Spectrogram Transformer | log-mel patches 上的 ViT；2021 年 SOTA。 |
| BEATs | Self-supervised audio | Microsoft 模型；截至 2026 年 iter3 领先 AudioSet。 |
| Mixup | 成对 augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于 mask 的 augmentation | 随机把 spectrogram 的 time 和 frequency bands 置零。 |
| mAP | 主要 multi-label 指标 | 跨类别和阈值计算 mean average precision。 |

## Further Reading / 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024 年的代表性架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 的默认起点。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流音频 augmentation。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 仍在使用的 50-class benchmark。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632-class YouTube taxonomy；仍是 gold standard。
