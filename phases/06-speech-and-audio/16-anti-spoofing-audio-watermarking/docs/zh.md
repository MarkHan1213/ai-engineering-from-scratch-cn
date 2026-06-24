# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify / 语音反欺骗与音频水印：ASVspoof 5、AudioSeal、WaveVerify

> Voice cloning 上线速度快过防御。2026 年生产语音系统需要两件事：一个 detector（AASIST、RawNet2），用于判断 real vs fake speech；一个 watermark（AudioSeal），能经受 compression 和 editing。两者都上线，否则不要上线 voice cloning。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 anti-spoofing / deepfake detection、audio watermarking 和 authenticated provenance 三层防御
- 理解 ASVspoof 5、AASIST、RawNet2、AudioSeal、WavMark、WaveVerify 和 C2PA 的角色
- 实现 toy spectral detector、AudioSeal embed/detect、EER 计算和 safe_tts 集成框架
- 识别 watermark detector 缺失、校准不足、pitch-shift 攻击、metadata stripping 和 liveness 误用等上线风险

## The Problem / 问题

三种相关防御：

1. **Anti-spoofing / deepfake detection。** 给定一段 audio clip，它是 synthetic 还是真实？ASVspoof benchmarks（ASVspoof 2019 → 2021 → 5）是 gold standard。
2. **Audio watermarking / 音频水印。** 在生成音频中嵌入不可感知信号，后续 detector 可以提取。AudioSeal（Meta）和 WavMark 是开放选项。
3. **Authenticated provenance / 认证来源。** 对 audio files + metadata 做 cryptographic signing。C2PA / Content Authenticity Initiative。

Detection 处理不配合的 adversaries。Watermarking 处理 compliance，即 AI-generated audio 应该可识别。2026 年二者都需要。

## The Concept / 概念

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — the 2024-2025 benchmark / ASVspoof 5：2024–2025 benchmark

相比之前版本，最大变化是：

- **Crowdsourced data**（不是 studio clean）—— 条件更真实。
- **~2000 speakers**（以前约 100）。
- **32 attack algorithms。** TTS + voice conversion + adversarial perturbation。
- **Two tracks。** Countermeasure (CM) standalone detection；用于 biometric systems 的 Spoofing-robust ASV (SASV)。

ASVspoof 5 上的 state-of-the-art：约 7.23% EER。旧版 ASVspoof 2019 LA 上是 0.42% EER。真实部署中，对 in-the-wild clips 预期 5–10% EER。

### AASIST and RawNet2 — detection model families / AASIST 与 RawNet2：检测模型家族

**AASIST**（2021，持续更新到 2026）。在 spectral features 上做 graph-attention。当前 ASVspoof 5 countermeasure task 的 SOTA。

**RawNet2。** Raw waveform 上的 convolutional front-end + TDNN backbone。更简单的 baseline；fine-tuning 后仍有竞争力。

**NeXt-TDNN + SSL features。** 2025 变体：ECAPA-style + WavLM features + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal — the 2024 watermark default / AudioSeal：2024 水印默认选择

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 于 2024 年 12 月）。关键设计：

- **Localized / 局部化。** 在 16 kHz sample resolution（1/16000 s）上逐 frame 检测 watermark。
- **Generator + detector jointly trained / 生成器和检测器联合训练。** Generator 学会嵌入不可闻信号；detector 学会在 augmentations 后找出它。
- **Robust / 鲁棒。** 可经受 MP3 / AAC compression、EQ、speed-shift ±10%、noise mix +10 dB SNR。
- **Fast / 快。** Detector 运行 485× realtime；比 WavMark 快 1000×。
- **Capacity / 容量。** 16-bit payload（可编码 model ID、generation timestamp、user ID）可嵌入每个 utterance。

### WavMark / WavMark

AudioSeal 之前的开放 baseline。Invertible neural network，32 bits/sec。问题：

- Synchronization brute-force 很慢。
- 可能被 Gaussian noise 或 MP3 compression 移除。
- 不适合 real-time。

### WaveVerify (July 2025) / WaveVerify（2025 年 7 月）

处理 AudioSeal 的弱点，尤其是 temporal manipulations（reversal、speed）。使用 FiLM-based generator + Mixture-of-Experts detector。在标准攻击下与 AudioSeal 有竞争力；能处理 temporal edits。

### The gap adversaries exploit / 攻击者利用的缺口

AudioMarkBench 指出：“under pitch shift, all watermarks show Bit Recovery Accuracy below 0.6, indicating near-complete removal.” **Pitch-shift 是通用攻击。** 2026 年还没有水印能完全抵抗激进 pitch modification。这就是为什么 watermarking 之外还需要 detection（AASIST）。

### C2PA / Content Authenticity Initiative / C2PA / Content Authenticity Initiative

不是 ML 技术，而是一种 manifest format。Audio files 带有关于 creation tool、author、date 的 cryptographically signed metadata。Audobox / Seamless 会使用它。它适合 provenance；但如果坏人重新编码并剥离 metadata，它就无能为力。

## Build It / 动手构建

### Step 1: a simple spectral-feature detector (toy) / 第 1 步：简单 spectral-feature detector（toy）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Synthetic speech 往往有异常平坦的高频能量。生产 detector 用 AASIST，而不是这个 toy。但直觉成立。

### Step 2: AudioSeal embed + detect / 第 2 步：AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Step 3: evaluation — EER / 第 3 步：评估 EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4: the production integration / 第 4 步：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次 generation 都要带：(1) watermark，(2) signed manifest，(3) 满足 retention policy 的 audit log。

## Use It / 应用它

| Use case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | AudioSeal embed on every output (non-negotiable) |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
| Call-center fraud detection | AASIST on 20% sample of incoming calls |
| Podcast authenticity | C2PA signing on upload, AudioSeal if AI-generated |
| Research / training detectors | ASVspoof 5 train/dev/eval sets |

## Pitfalls / 常见坑

- **Watermark without detector ever running / 有 watermark 但 detector 从不运行。** 没意义。把 detector 放进 CI。
- **Detection without calibration / 检测不校准。** 在 ASVspoof LA 上训练的 AASIST 会 overfit；真实准确率下降。用你的 domain 校准。
- **Pitch-shift gap / pitch-shift 缺口。** 激进 pitch shift 会移除大多数 watermarks。准备 detection fallback。
- **Metadata strip-and-rehost / 剥离 metadata 后重新托管。** C2PA 很容易被 re-encoding 绕过。始终把 cryptographic + perceptual（watermark）防御一起用。
- **Liveness as detection / 把 liveness 当 detection。** 让用户说随机短语可以防 replay attacks，但防不了 real-time cloning。

## Ship It / 交付它

保存为 `outputs/skill-spoof-defender.md`。为 voice-gen deployment 选择 detection model、watermark、provenance manifest 和 operational playbook。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。Toy detector + toy watermark embed/detect on synthetic audio。
2. **Medium / 中等。** 安装 `audioseal`，在 TTS output 中嵌入 16-bit payload，再 decode。用 noise 破坏音频并测量 Bit Recovery Accuracy。
3. **Hard / 困难。** 在 ASVspoof 2019 LA 上 fine-tune RawNet2 或 AASIST。测量 EER。在 F5-TTS 生成 clips 的 held-out set 上测试，观察 OOD detection 如何退化。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| ASVspoof | benchmark | 两年一度 challenge；2024 = ASVspoof 5。 |
| CM (countermeasure) | detector | 分类器：real speech vs synthetic / converted。 |
| SASV | Speaker verif + CM | 集成 biometric + spoof detection。 |
| AudioSeal | Meta watermark | Localized，16-bit payload，比 WavMark 快 485×。 |
| Bit Recovery Accuracy | Watermark survival | 攻击后恢复出的 payload bits 比例。 |
| C2PA | Provenance manifest | 关于 creation / authorship 的 cryptographic metadata。 |
| AASIST | detector family | 基于 graph-attention 的 anti-spoofing SOTA。 |

## Further Reading / 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前 benchmark。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — watermark 默认选择。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于 temporal attacks 的 MoE detector。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA detection backbone。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — robustness evaluation。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — provenance manifest format。
