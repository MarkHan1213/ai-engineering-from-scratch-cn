# Audio Generation / 音频生成

> 音频是 16-48 kHz 的 1-D signal。5 秒 clip 就有 80-240k samples。没有 transformer 会直接 attend 这么长的序列。2026 年每个生产音频模型的解法都一样：用 neural codec（Encodec、SoundStream、DAC）把音频压成 50-75 Hz 的 discrete tokens，再用 transformer 或 diffusion model 生成 tokens。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释音频生成为什么必须先用 neural codec 降采样成 tokens 或 latents
- 区分 text-to-speech、music generation 和 sound design 的生成需求
- 理解 RVQ、token-autoregressive、latent diffusion 和 flow matching 在音频中的位置
- 根据 streaming、质量、授权和实时性选择 2026 年音频生成栈

## The Problem / 问题

三类音频生成任务：

1. **Text-to-speech / 文本转语音。** 给定文本，生成语音。干净语音是 narrow-band，且有强 phonetic structure；transformer-over-tokens 已经解决得很好。VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **Music generation / 音乐生成。** 给定 prompt（文本、旋律、和弦进行、genre），生成音乐。分布宽得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **Audio effects / sound design / 音效与声音设计。** 给定 prompt，生成环境声或 Foley。AudioGen、AudioLDM 2、Stable Audio Open。

三者都跑在同一个 substrate 上：neural audio codec + token-AR 或 diffusion generator。

## The Concept / 概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Neural Audio Codecs / 神经音频编码器

Encodec（Meta, 2022）、SoundStream（Google, 2021）、Descript Audio Codec（DAC, 2023）。Convolutional encoder 把 waveform 压成 per-timestep vector；residual vector quantization（RVQ）把每个 vector 变成 K 个 codebook indices 的级联；decoder 再把它还原。24 kHz audio at 2 kbps using 8 RVQ codebooks at 75 Hz = 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Two Generative Paradigms on Top / 上层两种生成范式

**Token-autoregressive。** 把 RVQ tokens 展平成序列，运行 decoder-only transformer。MusicGen 用 “delayed parallel” 让 K 个 codebook streams 带 offset 并行输出。VALL-E 从 text prompt + 3 秒 voice sample 生成 speech tokens。

**Latent diffusion。** 把 codec tokens 打包成 continuous latents，或用 categorical diffusion 建模。Stable Audio 2.5 在 continuous audio latents 上使用 flow matching。AudioLDM 2 使用 text-to-mel-to-audio diffusion。

2024-2026 年趋势：flow matching 在音乐上胜出（推理更快、样本更干净），而 token-AR 仍主导语音，因为它天然 causal，容易 streaming。

## Production Landscape / 生产格局

| System / 系统 | Task / 任务 | Backbone | Latency / 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## Build It / 动手构建

`code/main.py` 模拟核心思想：在 synthetic “audio token” sequences 上训练一个 tiny next-token transformer。数据有两种 “styles”：style A 是低高 token 交替，style B 是单调 ramp。模型按 style condition 采样。

### Step 1: synthetic audio tokens / 第 1 步：合成 audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: train a tiny token predictor / 第 2 步：训练 tiny token predictor

一个按 style condition 的 bigram-style predictor。重点不是模型大小，而是模式：codec tokens → cross-entropy training → autoregressive sampling。

### Step 3: sample conditionally / 第 3 步：条件采样

给定 style token 和 starting token，从预测分布里 sample next token。持续 20-40 tokens。

## Pitfalls / 常见坑

- **Codec quality caps output quality。** 如果 codec 无法忠实表示某种声音，再好的 generator 也没用。DAC 是当前 open best。
- **RVQ error accumulation。** 每个 RVQ layer 建模上一层 residual。Layer 1 的错误会传播。在高层 sampling 时 temperature 0 有帮助。
- **Musical structure。** 75 Hz 下 30 秒 tokens 是 20k+ tokens。Transformer 很难。MusicGen 使用 sliding window + prompt continuation；Stable Audio 使用 shorter clips + crossfading。
- **Artifacts at boundaries。** 生成 clip 之间 crossfading 需要仔细 overlap-add。
- **Clean-data appetite。** 音乐生成器需要数万小时 licensed music。Suno / Udio RIAA lawsuit（2024）把这个问题推到台前。
- **Voice cloning ethics。** 3 秒 sample 加 text prompt 就足以让 VALL-E / XTTS / ElevenLabs clone voice。每个生产模型都需要 abuse detection + opt-out lists。

## Use It / 应用它

| Task / 任务 | 2026 stack |
|------|------------|
| Commercial TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| Voice cloning (consent-verified) | XTTS v2（open）或 ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API、Suno 或 Udio |
| Music with lyrics | Suno v4 或 Udio v1.5 |
| Sound effects / Foley | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| Real-time voice agent | GPT-4o realtime 或 Gemini Live |
| Open-weights music research | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| Dubbing / translation | HeyGen、ElevenLabs Dubbing |

## Ship It / 交付它

保存 `outputs/skill-audio-brief.md`。Skill 接收 audio brief（task、duration、style、voice、license），并输出：model + hosting、prompt format（genre tags、style descriptors、structural markers）、codec + generator + vocoder chain、seed protocol 和 eval plan（MOS / CLAP score / CER for TTS / user A/B）。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py` 并显式设置 style。验证生成序列符合对应 style pattern。
2. **Medium / 中等。** 加入 delayed parallel decoding：模拟 2 条必须保持 1-step offset 的 token streams。训练 joint predictor。
3. **Hard / 困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同 prompts 生成 10 秒 clip，并做 style adherence A/B。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Codec | "Neural compression" | 音频 encoder / decoder；典型输出是 50-75 Hz tokens。 |
| RVQ | "Residual VQ" | K 个 quantizers 的级联；每层建模上一层 residual。 |
| Token | "One codec symbol" | Codebook 中的离散 index；1024 或 2048 常见。 |
| Delayed parallel | "Offset codebooks" | 以 staggered offsets 输出 K 条 token streams，降低 sequence length。 |
| Flow matching | "The 2024 win for audio" | 比 diffusion 路径更直的替代方案；采样更快。 |
| Voice prompt | "3-second sample" | Speaker embedding 或 token prefix，用来引导 cloned voice。 |
| Mel spectrogram | "The visual" | Log-magnitude perceptual spectrogram；许多 TTS 系统使用。 |
| Vocoder | "Mel to wave" | 把 mel spectrogram 转回 audio 的神经组件。 |

## Production Note: Audio Is a Streaming Problem / 生产备注：音频是 streaming 问题

音频是用户期望 *边生成边到达* 的输出模态，而不是一次性返回。用生产术语说，TPOT 很重要，因为用户的听觉速度就是目标吞吐，而不是阅读速度。对 16kHz、约 75 tokens/second（Encodec）的音频 tokenization，服务器必须为每个用户生成 ≥75 tokens/sec 才能平滑播放。

两个架构后果：

- **Flow-matching audio models cannot stream trivially / Flow-matching 音频模型不能自然 streaming。** Stable Audio 2.5 和 AudioCraft 2 一次渲染固定长度 clip。要 streaming，就要切 chunk 并 overlap boundaries，类似 sliding-window diffusion，相比 codec AR 增加 100-300ms 延迟开销。

如果产品是 “live voice chat” 或 “real-time music continuation”，选择 codec AR 路线。如果是 “提交后渲染 30 秒 clip”，flow-matching 在质量和总延迟上更好。

## Further Reading / 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec 标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的 neural audio codec。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 text-to-music with flow matching。
