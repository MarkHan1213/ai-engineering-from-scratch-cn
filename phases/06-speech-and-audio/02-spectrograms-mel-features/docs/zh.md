# Spectrograms, Mel Scale & Audio Features / 频谱图、Mel 标度与音频特征

> 神经网络并不擅长直接消费 raw waveform。它们更适合消费 spectrogram，更适合消费 mel spectrogram。2026 年的 ASR、TTS 和 audio classifier，成败很大程度取决于这一个预处理选择。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 01 (Audio Fundamentals)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 STFT、log-magnitude、mel scale、mel filterbank、log-mel 和 MFCC 的关系
- 从 waveform 手写 frame、Hann window、STFT magnitude、mel filterbank 和 DCT-II
- 判断 frame size、hop、FFT size、mel count 对时间/频率分辨率的影响
- 识别 sample-rate mismatch、normalization drift 和 padding leakage 等生产问题

## The Problem / 问题

拿一段 10 秒、16 kHz 的音频来说，它有 160,000 个浮点数，都在 `[-1, 1]` 里，却几乎不会直接和标签 “dog barking” 或 “the word cat” 对齐。Raw waveform 确实包含信息，但形式并不利于模型提取。两个相同音素只要相隔 100 ms，raw samples 就会完全不同。

Spectrogram 解决了这个问题。它压缩了人类听觉不敏感的时间细节（微秒级 jitter），保留了人类会关注的结构（哪些频率有能量，以及这些能量如何随约 10–25 ms 的窗口变化）。

Mel spectrogram 更进一步。人类对 pitch 的感知近似是对数的：100 Hz 到 200 Hz 听起来的“距离”，和 1000 Hz 到 2000 Hz 类似。Mel scale 会扭曲频率轴，让它更贴近这种感知。2010 到 2026 年，mel-scaled spectrogram 一直是语音机器学习中最重要的单一特征。

## The Concept / 概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform) / 短时傅里叶变换。** 把 waveform 切成重叠 frame（典型值：25 ms window、10 ms hop，也就是 16 kHz 下 400 samples / 160 samples）。每个 frame 乘上 window function（Hann 是默认选择；Hamming 取舍略有不同）。对每个 frame 做 FFT。把 magnitude spectrum 堆成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是 spectrogram。

**Log-magnitude / 对数幅值。** Raw magnitudes 跨越 5–6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。所有生产 pipeline 都用 log-magnitude，而不是 raw magnitude。

**Mel scale / Mel 标度。** Hz 中的频率 `f` 映射到 mel `m` 的公式是 `m = 2595 * log10(1 + f / 700)`。这个映射在 1 kHz 以下大致线性，在 1 kHz 以上大致对数。覆盖 0–8 kHz 的 80 个 mel bins 是 ASR 输入标准。

**Mel filterbank / Mel 滤波器组。** 一组在 mel scale 上等距分布的三角滤波器。每个滤波器都是相邻 FFT bins 的加权和。用 filterbank matrix 乘 STFT magnitude，就能通过一次 matmul 得到 mel spectrogram。

**Log-mel spectrogram / 对数 Mel 频谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。也是 2026 年通用的 audio frontend。

**MFCCs / Mel 频率倒谱系数。** 对 log-mel spectrogram 应用 DCT (type II)，保留前 13 个系数。它会去相关并进一步压缩特征。大约到 2015 年之前，MFCC 一直是主导特征；后来基于 raw log-mels 的 CNN/Transformer 追了上来。它仍用于 speaker recognition（x-vectors、ECAPA）。

**Resolution trade / 分辨率取舍。** 更大的 FFT 带来更好的频率分辨率，但时间分辨率更差。25 ms / 10 ms 是 audio-ML 默认值；音乐常用 50 ms / 12.5 ms；transient detection（鼓点、爆破音）常用 5 ms / 2 ms。

```figure
spectrogram-window
```

## Build It / 动手构建

### Step 1: frame the waveform / 第 1 步：给 waveform 分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的音频，在 `frame_len=400, hop=160` 时会产生 998 个 frames。

### Step 2: Hann window / 第 2 步：Hann window

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 前逐元素相乘。这样可以减少在非零端点截断 frame 造成的 spectral leakage。

### Step 3: STFT magnitude / 第 3 步：STFT 幅值

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT 且向量化）。这里的循环是教学用的；`code/main.py` 中它可以在短音频上运行。

### Step 4: mel filterbank / 第 4 步：Mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

使用 `n_fft=400`、覆盖 0–8 kHz 的 80 个 mels，会得到一个 `(80, 201)` 矩阵。把 `(n_frames, 201)` 的 STFT magnitude 乘上它的转置，就得到 `(n_frames, 80)` 的 mel spectrogram。

### Step 5: log-mel / 第 5 步：log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案包括 `librosa.power_to_db`（按 reference 归一化的 dB）和 `10 * log10(power + eps)`。Whisper 使用更复杂的 clip + normalize 流程（见 Whisper 的 `log_mel_spectrogram`）。

### Step 6: MFCCs / 第 6 步：MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个 log-mel frame 应用 DCT，保留前 13 个系数。这就是 MFCC 矩阵。第一个系数通常会被丢弃，因为它编码的是整体能量。

## Use It / 应用它

2026 年的 stack：

| Task | Features |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop，用于细粒度时间控制 |
| Audio classification (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels 或 raw-waveform SSL |
| Music (MusicGen, Stable Audio 2) | EnCodec discrete tokens（不是 mels） |
| Keyword spotting | 小设备上常用 40 MFCCs |

经验法则：**如果你不是在做音乐，先从 80 log-mels 开始。** 任何偏离都需要拿出证据。

## Pitfalls that still ship in 2026 / 2026 年仍会上线的坑

- **Mel count mismatch / Mel 数量不匹配。** 训练用 80 mels，推理用 128 mels。静默失败。训练和推理两端都要 log feature shape。
- **Sample-rate mismatch upstream / 上游采样率不匹配。** 22.05 kHz 计算出的 mels 和 16 kHz 不一样。先修正 SR，再做 featurization。
- **dB vs log / dB 和 log 混淆。** Whisper 期望 log-mel，不是 dB-mel。有些 HF pipeline 会自动检测；你的自定义代码不会。
- **Normalization drift / 归一化漂移。** 训练时 per-utterance normalization，推理时 global normalization。这是会让 WER 翻倍的生产 bug。
- **Leakage from padding / padding 泄漏。** 在 clip 末尾 zero-padding 会在尾部 frames 中产生平坦频谱。应对称 padding 或复制边界。

## Ship It / 交付它

保存为 `outputs/skill-feature-extractor.md`。这个 skill 会为给定目标模型选择 feature type、mel count、frame/hop 和 normalization。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会合成一个 chirp（频率从 200 → 4000 Hz 扫频），并打印每个 frame 的 argmax mel bin。可选绘图，并确认它符合 sweep。
2. **Medium / 中等。** 将 `n_mels` 设为 `{40, 80, 128}`，将 `frame_len` 设为 `{200, 400, 800}` 后重新运行。测量时间轴上的 sharp-peak bandwidth。哪个组合最能分辨 chirp？
3. **Hard / 困难。** 实现 `power_to_db`，并在 AudioMNIST 上比较一个 tiny CNN classifier 使用三种特征时的 ASR accuracy：(a) raw log-mel，(b) `ref=max` 的 dB-mel，(c) MFCC-13 + delta + delta-delta。报告 top-1 accuracy。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Frame | 一个切片 | 送入一次 FFT 的 25 ms waveform chunk。 |
| Hop | 步幅 | 连续 frames 之间相隔的 samples；ASR 默认 10 ms。 |
| Window | Hann/Hamming 那个东西 | 逐点乘子，把 frame 边缘渐变到零。 |
| STFT | spectrogram 生成器 | Framed + windowed FFT；输出 time × frequency 矩阵。 |
| Mel | 扭曲后的频率 | 对数感知标度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那个矩阵 | 把 STFT 投影到 mel bins 的三角滤波器。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年已标准化。 |
| MFCC | 老派特征 | log-mel 的 DCT；13 个系数，去相关。 |

## Further Reading / 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 mel scale 论文。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级 pipeline。
