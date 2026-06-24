# Speaker Recognition & Verification / 说话人识别与验证

> ASR 问的是“他们说了什么？”Speaker recognition 问的是“是谁说的？”数学形式看起来一样，都是 embeddings 加 cosine；但每个生产决策都压在一个 EER 数字上。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分 speaker verification、identification、open-set 和 diarization 的任务边界
- 用 MFCC statistics 构造 toy speaker embedding，并实现 cosine threshold verification
- 从 same/different similarity pairs 计算 EER 和 threshold_at_eer
- 理解 ECAPA-TDNN、WavLM-SV、PLDA、score normalization 和 diarization pipeline 的生产取舍

## The Problem / 问题

用户说出一个口令短语。你想知道：这是不是他声称的那个人（*verification*, 1:1）？或者他是不是注册语音库中的某个人（*identification*, 1:N）？又或者两者都不是，这是一个未知说话人（*open-set*）？

2018 年以前：GMM-UBM + i-vectors。EER 还可以，但对 channel shift（电话 vs 笔记本）和情绪很脆弱。2018–2022：x-vectors（用 angular margin 训练的 TDNN backbone）。2022+：ECAPA-TDNN 和 WavLM-large embeddings。到 2026 年，这个领域主要由三个模型和一个指标主导。

这个指标是 **EER**，即 Equal Error Rate。调整决策阈值，让 False Accept Rate = False Reject Rate。这个交叉点就是 EER。每篇论文、每个 leaderboard、每次采购评审都会用它。

## The Concept / 概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**The pipeline / 流程。** Enrollment：录制目标说话人 5–30 秒语音；计算固定维度 embedding（ECAPA-TDNN 是 192-d，WavLM-large 是 256-d）。Verification：获取测试 utterance embedding；计算 cosine similarity；与 threshold 比较。

**ECAPA-TDNN (2020, still dominant 2026)。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。1D conv blocks，带 squeeze-excitation、multi-head attention pooling，最后接 linear layer 输出 192-d。在 VoxCeleb 1+2（2,700 speakers，1.1M utterances）上用 Additive Angular Margin loss（AAM-softmax）训练。

**WavLM-SV (2022+)。** 用 AAM loss fine-tune pretrained WavLM-large SSL backbone。质量更高但更慢，300+ MB vs 15 MB。

**x-vector (baseline)。** TDNN + statistics pooling。经典方案；在 CPU / edge 上仍然有用。

**AAM-softmax。** 在 angular space 中给正确类别加 margin `m` 的标准 softmax：`cos(θ + m)`。它会强制扩大 inter-class angular separation。典型值是 `m=0.2`，scale `s=30`。

### Scoring / 打分

- **Cosine** between enrollment and test embeddings。基于 threshold 决策。
- **PLDA (Probabilistic LDA)。** 把 embeddings 投影到 latent space，让 same-speaker vs different-speaker 有 closed-form likelihood ratio。加在 cosine 上可降低 10–20% EER。2020 年以前是标准做法；现在只在 closed-set setups 中常见。
- **Score normalization。** `S-norm` 或 `AS-norm`：用一组 imposter cohort 的 means 和 stds 归一化每个 score。跨领域评估时很关键。

### Numbers you should know (2026) / 你应该知道的 2026 年数字

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization / 说话人分离标注

“谁在什么时候说话”。Pipeline：VAD → segment → embed each segment → cluster（agglomerative 或 spectral）→ smooth boundaries。现代 stack 是 `pyannote.audio` 3.1，它把 speaker segmentation + embedding + clustering 封装在一个调用后面。2026 年 AMI 上的 SOTA DER 约为 15%（从 2022 年的 23% 下降）。

## Build It / 动手构建

### Step 1: toy embedding from MFCC statistics / 第 1 步：用 MFCC statistics 构造 toy embedding

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离 SOTA 很远，只用于教学。`code/main.py` 会在 synthetic speaker data 上用它做 proof-of-concept。

### Step 2: cosine similarity + threshold / 第 2 步：cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3: EER from similarity pairs / 第 3 步：从 similarity pairs 计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 `(eer, threshold_at_eer)`。两个都要报告。

### Step 4: production with SpeechBrain / 第 4 步：用 SpeechBrain 做生产实现

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Step 5: diarize with pyannote / 第 5 步：用 pyannote 做 diarization

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Use It / 应用它

2026 年的 stack：

| Situation | Pick |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST or RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Pitfalls / 常见坑

- **Channel mismatch / 通道不匹配。** 在 VoxCeleb（web video）上训练的模型 ≠ phone-call audio。一定要在目标 channel 上评估。
- **Short utterances / 短 utterance。** 测试音频低于 3 秒时 EER 会明显变差。
- **Enrollment with noise / 带噪 enrollment。** 一个 noisy enrollment 会污染 anchor。使用 ≥3 个干净样本并取平均。
- **Fixed threshold across conditions / 跨条件固定阈值。** 一定要在目标领域 held-out dev set 上 tune threshold。
- **Cosine on non-normalized embeddings / 对未归一化 embedding 做 cosine。** 先 L2-normalize，否则 magnitude 会主导结果。

## Ship It / 交付它

保存为 `outputs/skill-speaker-verifier.md`。选择 model、enrollment protocol、threshold-tuning plan 和 fraud safeguards。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会构造 synthetic “speakers”（不同 tone profiles）、做 enrollment，并在 100-pair trial list 上计算 EER。
2. **Medium / 中等。** 在 30 条 VoxCeleb1 utterances（5 speakers × 每人 6 条）上使用 SpeechBrain ECAPA。分别用 cosine 和 PLDA 计算 EER。
3. **Hard / 困难。** 用 `pyannote.audio` 构建完整 enroll → diarize → verify pipeline。在 AMI dev set 上评估 DER。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| EER | headline metric | False Accept = False Reject 时的 threshold。 |
| Verification | 1:1 | “这是 Alice 吗？” |
| Identification | 1:N | “谁在说话？” |
| Open-set | 可能未知 | Test set 可能包含未 enrollment 的说话人。 |
| Enrollment | 注册 | 计算某个说话人的 reference embedding。 |
| AAM-softmax | loss | 带 additive angular margin 的 softmax；强制 cluster separation。 |
| PLDA | 经典 scoring | Probabilistic LDA；在 embeddings 上做 likelihood-ratio scoring。 |
| DER | Diarization metric | Diarization Error Rate，即 miss + false alarm + confusion。 |

## Further Reading / 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典 deep-embedding 论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026 年主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 用于 SV 和 diarization 的 SSL backbone。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产 diarization + embedding stack。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 各模型当前 EER 排名。
