# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split / 神经音频编解码器：EnCodec、SNAC、Mimi、DAC 与语义-声学拆分

> 2026 年的音频生成几乎全是 tokens。EnCodec、SNAC、Mimi 和 DAC 把连续 waveform 变成 transformer 可以预测的离散序列。semantic-vs-acoustic token split，也就是 first-codebook 作为 semantic、其余作为 acoustic，是 Transformer 之后音频架构中最重要的变化。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 neural audio codec 如何把 continuous waveform 离散化成可由 transformer 预测的 tokens
- 理解 RVQ、frame rate、codebook 数量和 bitrate 对重建质量与 LM 序列长度的影响
- 区分 EnCodec、DAC、SNAC、Mimi 的设计目标和 2026 年适用场景
- 掌握 semantic codebook 与 acoustic codebooks 的分工，以及它们为什么支撑现代 speech generation

## The Problem / 问题

Language models 处理离散 tokens。Audio 是连续信号。如果你想为 speech / music 构建 LLM-style 模型，比如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus，就首先需要一个 **neural audio codec**：一个 learned encoder，把音频离散化成小词表 tokens；以及匹配的 decoder，把 waveform 重建出来。

现在出现了两个家族：

1. **Reconstruction-first codecs / 重建优先 codec** —— EnCodec、DAC。优化感知音频质量。Tokens 是 “acoustic” 的，会捕获包括 speaker identity、timbre、background noise 在内的一切。
2. **Semantic-first codecs / 语义优先 codec** —— Mimi（Kyutai）、SpeechTokenizer。强制第一个 codebook 编码 linguistic / phonetic content（通常从 WavLM distill）。后续 codebooks 负责 acoustic detail。

2024–2026 年的洞察是：**当你尝试从文本生成语音时，纯 reconstruction codec 会带来模糊语音。** Codec tokens 上的 LLM 必须在同一个 codebook 中同时学习语言结构和声学结构，这无法很好扩展。把它们拆开，即 semantic codebook 0、acoustic codebooks 1-N，才让 Moshi 和 Sesame CSM 能工作。

## The Concept / 概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### The core trick: Residual Vector Quantization (RVQ) / 核心技巧：Residual Vector Quantization (RVQ)

现代音频 codec 不使用一个巨大的 codebook（高质量会需要数百万 codes），而是使用 **RVQ**：一串小 codebooks 的级联。第一个 codebook 量化 encoder output；第二个量化 residual；依此类推。每个 codebook 有 1024 个 codes。8 个 codebooks 的有效词表是 1024^8 = 10^24。

推理时，decoder 会把每个 frame 选择的所有 codes 相加来重建。

### The four codecs that matter in 2026 / 2026 年重要的四个 codec

**EnCodec (Meta, 2022)。** Baseline。Waveform 上的 encoder-decoder，RVQ bottleneck。24 kHz，最多 32 codebooks，默认 4 codebooks @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。MusicGen 使用它。

**DAC (Descript, 2023)。** RVQ，带 L2-normalized codebooks、periodic activation functions 和改进的 losses。开放 codec 中重建保真度最高，有时 12 codebooks 下的语音与原始几乎不可区分。44.1 kHz full-band。

**SNAC (Hubert Siuzdak, 2024)。** Multi-scale RVQ，coarse codebooks 的 frame rate 低于 fine codebooks。它等效于分层建模音频：约 12 Hz 的粗略“sketch”加 50 Hz 的细节。Orpheus-3B 使用它，因为层次结构很适合基于 LM 的生成。

**Mimi (Kyutai, 2024)。** 2026 年的 game-changer。12.5 Hz frame rate（极低），8 codebooks @ 4.4 kbps。Codebook 0 **distilled from WavLM**，训练目标是预测 WavLM 的 speech-content features。Codebooks 1-7 是 acoustic residuals。这个拆分支撑 Moshi（第 15 课）和 Sesame CSM。

### Frame rates matter for language modeling / Frame rate 对 language modeling 很关键

更低 frame rate = 更短 sequence = 更快 LM。

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

在 12.5 Hz 下，10 秒 utterance 只有 125 个 codec frames，transformer 可以轻松预测。

### Semantic vs acoustic tokens / 语义 token 与声学 token

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token（Mimi 中的 codebook 0）。** 编码说了什么，即 phonemes、words、content。通过 auxiliary prediction loss 从 WavLM distill。
- **Acoustic tokens（codebooks 1-7）。** 编码 timbre、speaker identity、prosody、background noise 和细节。

AR LM 会先预测 semantic token（以 text 为条件），再预测 acoustic tokens（以 semantic + speaker reference 为条件）。这个 factorization 正是现代 TTS 能 zero-shot-clone voices 的原因：semantic model 处理内容；acoustic model 处理音色。

### 2026 reconstruction quality (bits per sec, lower bitrate is better) / 2026 重建质量（bits per sec，bitrate 越低越好）

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统 codec 如 Opus 在每 bit 感知质量上仍然胜出。Neural codecs 胜在 **discrete tokens**（Opus 不产生这种 tokens）和 **generative-model quality**（LM 能对这些 tokens 做什么）。

## Build It / 动手构建

### Step 1: encode with EnCodec / 第 1 步：用 EnCodec encode

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 下 `n_codebooks=8`。每个 code 是 0-1023（10-bit）。

### Step 2: decode and measure reconstruction / 第 2 步：decode 并测量重建

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Step 3: the semantic-acoustic split (Mimi-style) / 第 3 步：semantic-acoustic split（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 与 WavLM 对齐。你可以训练一个 text-to-semantic transformer，它的 vocabulary 远小于 direct-to-audio。然后再让单独的 acoustic-to-waveform decoder 以 speaker reference 为条件。

### Step 4: why AR LM over codec tokens works / 第 4 步：为什么 codec tokens 上的 AR LM 能工作

对一个 10 s speech clip，Mimi 的 12.5 Hz × 8 codebooks：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens 对 transformer 来说是很轻的 context。现代 GPU 上，一个 256M-parameter transformer 可以在毫秒级生成 10 秒语音。

## Use It / 应用它

问题 → codec 映射：

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC or Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

经验法则：**如果你在构建 generative model，从 Mimi 或 SNAC 开始。如果你在构建 compression pipeline，用 Opus。**

## Pitfalls / 常见坑

- **Too many codebooks / codebooks 过多。** 增加 codebooks 会线性提升 fidelity，但也线性增加 LM sequence length。通常停在 8–12。
- **Frame-rate mismatch / frame rate 不匹配。** 在 12.5 Hz Mimi 上训练 LM，再在 50 Hz EnCodec 上 fine-tune，会静默失败。
- **Assuming all codebooks equal / 以为所有 codebooks 等价。** 在 Mimi 中，codebook 0 承载 content；丢掉它会毁掉 intelligibility。丢掉 codebook 7 几乎听不出来。
- **Using reconstruction quality as the only metric / 只看重建质量。** Codec 可能重建很好，但如果 semantic structure 很差，就不适合 LM-based generation。

## Ship It / 交付它

保存为 `outputs/skill-codec-picker.md`。为给定 generative 或 compression task 选择 codec。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它实现了一个 toy scalar + residual quantizer，并测量增加 codebooks 时的 reconstruction error。
2. **Medium / 中等。** 安装 `encodec`，在 held-out speech clip 上比较 1、4、8、32 codebooks。绘制 PESQ 或 MSE vs bitrate。
3. **Hard / 困难。** 加载 Mimi。Encode 一个 clip。把 codebook 0 替换为随机整数并 decode；再同样替换 codebook 7。比较两种破坏：codebook 0 corruption 应该毁掉 intelligibility；codebook 7 corruption 应该几乎不改变。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | 小 codebooks 的级联；每个量化前一个 residual。 |
| Frame rate | codec speed | 每秒多少 token-frames。更低 = LM 更快。 |
| Semantic codebook | Codebook 0 (Mimi) | 从 SSL features distill 的 codebook；编码 content。 |
| Acoustic codebooks | 其他全部 | Timbre、prosody、noise、fine detail。 |
| PESQ / ViSQOL | perceptual quality | 与 MOS 相关的客观指标。 |
| EnCodec | Meta codec | RVQ baseline；MusicGen 使用。 |
| Mimi | Kyutai codec | 12.5 Hz frame rate；semantic-acoustic split；支撑 Moshi。 |

## Further Reading / 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ baseline。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 最高保真的开放 codec。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split，WavLM distillation。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — two-stage semantic/acoustic paradigm。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 原始 streamable RVQ codec。
