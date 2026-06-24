# Audio Fundamentals — Waveforms, Sampling, Fourier Transform / 音频基础：波形、采样与傅里叶变换

> 波形是原始信号。频谱图是表示形式。Mel 特征是更适合机器学习的形态。现代 ASR 和 TTS pipeline 都会沿着这级台阶向上走，而第一阶就是理解采样和傅里叶。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 waveform、sample rate、bit depth、DFT、FFT 和 STFT 在音频 pipeline 中的角色
- 判断采样率不匹配、混叠和原始样本/频域表示混用带来的失败模式
- 从零合成正弦波、手写 DFT，并把频率 bin 映射回 Hz
- 用一个可复现实验演示 Nyquist 频率和 aliasing 的影响

## The Problem / 问题

麦克风产生的是“声压随时间变化”的信号。神经网络消费的是 tensor。二者之间夹着一整套约定；一旦违反，就会出现非常隐蔽的 bug：模型看起来训练正常，但 WER 翻倍；TTS 上线后带底噪；或者 voice cloning 系统记住了麦克风，而不是说话人。

语音系统里的很多问题都可以追溯到三个问题：

1. 数据录制时的 sample rate 是多少，模型期望的又是多少？
2. 信号是否发生了 aliasing？
3. 你处理的是 raw samples，还是频率表示？

这些问题先做对，Phase 6 后面的内容就可控。做错了，即使 Whisper-Large-v4 也会输出垃圾。

## The Concept / 概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform / 波形。** 一个取值在 `[-1.0, 1.0]` 的一维浮点数组。索引单位是 sample number。要换算成秒，就除以 sample rate：`t = n / sr`。一段 16 kHz、10 秒的音频，就是 160,000 个浮点数。

**Sampling rate (sr) / 采样率。** 每秒采多少个样本。2026 年常见采样率：

| Rate | Use |
|------|-----|
| 8 kHz | 电话、旧式 VOIP。Nyquist 频率只有 4 kHz，会损失辅音信息，ASR 应避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz。 |
| 22.05 kHz | 较老 TTS 模型的 vocoder 训练常用。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**Nyquist-Shannon / 奈奎斯特-香农。** 采样率为 `sr` 时，最多只能无歧义表示到 `sr/2` 的频率。`sr/2` 这个边界就是 *Nyquist frequency*。高于 Nyquist 的能量会发生 *aliasing*，也就是折叠到低频，污染信号。降采样前一定要先做低通滤波。

**Bit depth / 位深。** 16-bit PCM（signed int16，范围 ±32,767）是通用交换格式。音乐常用 24-bit，内部 DSP 常用 32-bit float。`soundfile` 这类库可以读取 int16，但暴露给你的通常是 `[-1, 1]` 范围内的 float32 数组。

**Fourier Transform / 傅里叶变换。** 任意有限信号都可以看成不同频率正弦波的叠加。Discrete Fourier Transform (DFT) 对 `N` 个样本计算 `N` 个复数系数，每个系数对应一个 frequency bin。`bin k` 对应频率 `k · sr / N` Hz。幅值表示该频率上的振幅，角度表示相位。

**FFT / 快速傅里叶变换。** Fast Fourier Transform 是在 `N` 为 2 的幂时计算 DFT 的 `O(N log N)` 算法。每个音频库底层都会用 FFT。16 kHz 下 1024-sample FFT 会给出 512 个可用频率 bin，覆盖 0–8 kHz，分辨率为 15.6 Hz。

**Framing + window / 分帧与加窗。** 我们不会对整段音频做一次 FFT。做法是把它切成重叠的 *frames*（通常 25 ms frame，10 ms hop），给每个 frame 乘上 window function（Hann、Hamming），减少边缘不连续，然后对每个 frame 做 FFT。这就是 Short-Time Fourier Transform (STFT)。第 02 课会从这里继续。

```figure
mel-scale
```

## Build It / 动手构建

### Step 1: read a clip and plot the waveform / 第 1 步：读取音频片段并绘制波形

`code/main.py` 只使用 stdlib 的 `wave` 模块，让 demo 不依赖第三方库。生产环境中你会使用 `soundfile` 或 `torchaudio.load`（二者都返回 `(waveform, sr)` tuple）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Step 2: synthesize a sine wave from first principles / 第 2 步：从一阶原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

在 16 kHz 下合成 1 秒 440 Hz 正弦波（标准音 A），会得到 16,000 个浮点数。写入文件时用 `wave.open(..., "wb")` 和 16-bit PCM 编码。

### Step 3: compute the DFT by hand / 第 3 步：手写 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

复杂度是 `O(N²)`。用 `N=256` 验证正确性没问题，但真实音频完全不实用。生产代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### Step 4: find the dominant frequency / 第 4 步：找到主频

幅值峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行时，峰值应该落在 `440 * N / sr` 对应的 bin 附近。

### Step 5: demonstrate aliasing / 第 5 步：演示混叠

用 10 kHz 采样率采样一个 7 kHz 正弦波（Nyquist = 5 kHz）。7 kHz 高于 Nyquist，会折叠到 `10 − 7 = 3 kHz`。FFT 峰值会出现在 3 kHz。这是经典 aliasing demo，也是每个 DAC/ADC 都会配 brick-wall low-pass filter 的原因。

## Use It / 应用它

2026 年你实际会交付的 stack：

| Task | Library | Why |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` or `librosa.resample` | 内置正确的 anti-aliasing。 |
| STFT / Mel | `torchaudio` or `librosa` | GPU-friendly；适配 PyTorch 生态。 |
| 实时流式处理 | `sounddevice` or `pyaudio` | 跨平台 PortAudio binding。 |
| 检查文件 | `ffprobe` or `soxi` | CLI、快速、报告 sr/channels/codec。 |

决策规则：**先匹配 sample rate，再匹配其他东西**。Whisper 期望 16 kHz mono float32。你传入 44.1 kHz stereo，就会得到看起来像模型 bug 的垃圾输出。

## Ship It / 交付它

保存为 `outputs/skill-audio-loader.md`。这个 skill 帮你检查音频输入是否匹配下游模型期望，并在不匹配时正确重采样。

## Exercises / 练习

1. **Easy / 简单。** 在 16 kHz 下合成 1 秒 220 Hz + 440 Hz + 880 Hz 的混合信号。运行 DFT。确认预期 bin 上出现三个峰。
2. **Medium / 中等。** 用 48 kHz 录制一段 3 秒自己的声音 WAV。先用 `torchaudio.transforms.Resample`（带 anti-aliasing）降采样到 16 kHz，再用朴素 decimation（每三个 sample 取一个）降到 16 kHz。分别做 FFT。aliasing 出现在哪里？
3. **Hard / 困难。** 只用 `math` 和第 3 步的 DFT 从零构建 STFT。Frame size 400，hop 160，Hann window。用 `matplotlib.pyplot.imshow` 绘制幅值。这就是第 02 课的 spectrogram。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Sample rate | 每秒采样多少次 | ADC 测量信号的频率，单位 Hz。 |
| Nyquist | 你能表示的最高频率 | `sr/2`；高于它的能量会 alias 回低频。 |
| Bit depth | 每个 sample 的分辨率 | `int16` = 65,536 个 level；`float32` = `[-1, 1]` 中约 24-bit 精度。 |
| DFT | 序列版傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，通常要求 `N` 是 2 的幂。 |
| Bin | 频率列 | `k · sr / N` Hz；resolution = `sr / N`。 |
| STFT | spectrogram 的底层机制 | 随时间做 framed + windowed FFT。 |
| Aliasing | 奇怪的频率“幽灵” | 高于 Nyquist 的能量镜像折叠到更低的 bin。 |

## Further Reading / 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — sampling theorem 背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、经典的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实践 walkthrough。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 理解真实世界音频为什么不是干净正弦波的参考书。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟理清 frequency bin 直觉。
