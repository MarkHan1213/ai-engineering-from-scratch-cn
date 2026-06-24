# Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro / 文本转语音（TTS）：从 Tacotron 到 F5 与 Kokoro

> ASR 把语音反转成文本；TTS 把文本反转成语音。2026 年的 stack 分三部分：text → tokens，tokens → mel，mel → waveform。每一部分都有能装进笔记本的默认模型。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 TTS text frontend、acoustic model 和 vocoder 三段式 mental model
- 对比 Tacotron 2、FastSpeech 2、VITS、F5-TTS、Kokoro 和商业 TTS 的架构取舍
- 用 phonemizer、Kokoro、F5-TTS 和 HiFi-GAN shape 搭建 TTS pipeline 的关键路径
- 识别 text normalization、OOV proper nouns、clipping 和 sample-rate mismatch 等常见问题

## The Problem / 问题

你有一个字符串：“Please remind me to water the plants at 6 pm.” 你需要生成一段 3 秒音频：听起来自然，有正确 prosody（停顿、重音），能用正确元音读出 “plants”，并且在 live voice assistant 中能在 CPU 上 300 ms 内完成。你还需要切换声音，处理 code-switched input（“remind me at 6 pm, daijoubu?”），并且不要在姓名上出丑。

现代 TTS pipeline 通常这样：

1. **Text frontend。** Normalize text（日期、数字、email），转换为 phonemes 或 subword tokens，预测 prosody features。
2. **Acoustic model。** Text → mel spectrogram。Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024)。
3. **Vocoder。** Mel → waveform。WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), 2024+ 的 neural codec vocoders。

到 2026 年，随着 end-to-end diffusion 和 flow-matching models，acoustic + vocoder 的边界开始模糊。但调试时，“三部分”的 mental model 仍然成立。

## The Concept / 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017)。** Seq2seq：char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder 输出 mel frames。慢（AR），长文本不稳定。仍常作为 baseline 被引用。

**FastSpeech 2 (2020)。** 非自回归。Duration predictor 输出每个 phoneme 对应多少个 mel frames。单次前向，速度比 Tacotron 快 10×。因为采用 monotonic alignment，会损失一些自然度，但到处都能上线。

**VITS (2021)。** 通过 variational inference，把 encoder + flow-based duration + HiFi-GAN vocoder 端到端联合训练。质量高，单模型。2022–2024 年主导开源 TTS。变体包括 YourTTS（multi-speaker zero-shot）、XTTS v2（2024, Coqui）。

**F5-TTS (2024)。** 基于 flow matching 的 diffusion transformer。Prosody 自然，用 5 秒 reference audio 就能 zero-shot voice cloning。2026 年开源 TTS leaderboard 前列。335M params。

**Kokoro (2024)。** 小（82M）、CPU 可运行，实时 English TTS 里同级最佳。Closed-vocabulary English-only，apache-2.0。

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3。** 商业 state of the art。ElevenLabs v2.5 的 emotion tags（"[whispered]", "[laughing]"）和 character voices 在 2026 年主导 audiobook production。

### Vocoder evolution / Vocoder 演进

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | 发布时 SOTA |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

到 2026 年，大多数 “TTS” 模型都是从 text 到 waveform 的 end-to-end 模型；mel spectrogram 变成了内部表示。

### Evaluation / 评估

- **MOS (Mean Opinion Score)。** 1–5 分，crowd-sourced。仍是 gold standard，但很慢。
- **CMOS (Comparative MOS)。** A-vs-B preference。每条 annotation 的 confidence interval 更窄。
- **UTMOS, DNSMOS。** 无参考 neural MOS predictors。leaderboard 常用。
- **CER (Character Error Rate) via ASR。** 用 Whisper 跑 TTS 输出，再和输入文本计算 CER。作为 intelligibility proxy。
- **SECS (Speaker Embedding Cosine Similarity)。** 衡量 voice-cloning quality。

LibriTTS test-clean 上的 2026 年数字：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## Build It / 动手构建

### Step 1: phonemize input / 第 1 步：把输入转成 phonemes

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes 是通用桥梁。除非模型已经达到 VITS-level quality，否则不要把 raw text 直接喂给下游。

### Step 2: run Kokoro (2026 CPU default) / 第 2 步：运行 Kokoro（2026 CPU 默认选择）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M params。

### Step 3: run F5-TTS with voice cloning / 第 3 步：用 F5-TTS 做 voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒 reference clip + 它的 transcript；F5 会 clone prosody 和 timbre。

### Step 4: HiFi-GAN vocoder from scratch / 第 4 步：从零理解 HiFi-GAN vocoder

完整模型太大，不适合塞进 tutorial script，但形状是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练方式：adversarial（discriminator 看短窗口）+ mel-spectrogram reconstruction loss + feature-matching loss。这个部分已经商品化，直接用 `hifi-gan` repo 或 nvidia-NeMo 的 pretrained checkpoints。

### Step 5: the full pipeline (pseudocode) / 第 5 步：完整 pipeline（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Use It / 应用它

2026 年的 stack：

| Situation | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
| Voice cloning from 5 s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
| Low-resource language | Train VITS on 5–20 h target-lang data |
| Expressive / emotion tags | ElevenLabs v2.5 or StyleTTS 2 fine-tune |

截至 2026 年的开源领先选择：**F5-TTS 追求质量，Kokoro 追求效率**。除非你在做技术考古，不要再从 Tacotron 开始。

## Pitfalls / 常见坑

- **No text normalizer / 没有 text normalizer。** “Dr. Smith” 是读成 “Doctor” 还是 “Drive”？“2026” 是 “twenty twenty six” 还是 “two zero two six”？先 normalize，再 phonemizer。
- **OOV proper nouns / 未登录专有名词。** “Ghumare” → “ghyu-mair”？需要为未知 token 上线 fallback grapheme-to-phoneme model。
- **Clipping / 削波。** Vocoder 输出很少 clip，但 inference 时 mel scaling mismatch 可能超过 ±1.0。总是 `np.clip(wav, -1, 1)`。
- **Sample-rate mismatch / 采样率不匹配。** Kokoro 输出 24 kHz；你的下游 pipeline 期望 16 kHz → 重采样，否则会 aliasing。

## Ship It / 交付它

保存为 `outputs/skill-tts-designer.md`。为给定 voice、latency 和 language target 设计 TTS pipeline。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会从 toy vocab 构建 phoneme dictionary，估计每个 phoneme duration，并打印一个假的 “mel” schedule。
2. **Medium / 中等。** 安装 Kokoro，用 voice `af_bella` 和 `am_adam` 合成同一句话。比较 audio durations 和主观质量。
3. **Hard / 困难。** 录制你自己的 5 秒 reference clip。用 F5-TTS 克隆。报告 reference 与 cloned output 之间的 SECS。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme | sound unit | 抽象 sound class；英语中 ARPABet 有 39 个。 |
| Duration predictor | 每个 phoneme 持续多久 | 非自回归模型输出；每个 phoneme 对应整数 frames。 |
| Vocoder | Mel → waveform | 把 mel-spec 映射到 raw samples 的神经网络。 |
| HiFi-GAN | 标准 vocoder | 基于 GAN；2020–2024 年主导。 |
| MOS | 主观质量 | 人类评分者给出的 1–5 mean opinion score。 |
| SECS | Voice-clone metric | target 与 output speaker embedding 的 cosine similarity。 |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion；zero-shot cloning。 |
| Kokoro | CPU English leader | 82M-param model，Apache 2.0。 |

## Further Reading / 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq baseline。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前 open-source SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍会上线的 vocoder。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS。
