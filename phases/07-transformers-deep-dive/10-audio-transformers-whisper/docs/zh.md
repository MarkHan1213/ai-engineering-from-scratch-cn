# Audio Transformers — Whisper Architecture / 音频 Transformer：Whisper 架构

> Audio 是频率随时间变化的一张图。Whisper 像吃 mel spectrogram 的 ViT，然后把文字说回来。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 拆解 Whisper 从 audio waveform 到 log-mel spectrogram、encoder、decoder、text tokens 的 pipeline
- 理解 task tokens 如何控制 transcription、translation、language ID 和 timestamping
- 识别 Whisper 30-second window、chunking、VAD 与 streaming wrapper 的边界
- 根据 latency、language、edge constraints 和 diarization 需求选择 ASR stack

## The Problem / 问题

Whisper（OpenAI, Radford et al. 2022）之前，state-of-the-art automatic speech recognition（ASR）通常意味着 wav2vec 2.0 和 HuBERT：self-supervised feature extractors 加 fine-tuned head。质量高，但 data pipelines 昂贵，domain-brittle。Multilingual speech recognition 还需要每个 language family 单独模型。

Whisper 做了三个押注：

1. **Train on everything / 在所有东西上训练。** 从互联网抓取 97 种语言、680,000 小时 weakly-labeled audio。没有干净 academic corpus，没有 phoneme labels。
2. **Multi-task single model / 单模型多任务。** 一个 decoder 通过 task tokens 联合训练 transcription、translation、voice activity detection、language ID 和 timestamping。
3. **Standard encoder-decoder transformer / 标准 encoder-decoder transformer。** Encoder 消费 log-mel spectrograms。Decoder autoregressively 产生 text tokens。没有 vocoder，没有 CTC，没有 HMM。

结果是：Whisper large-v3 在 accents、noise 和零 clean labeled data 的语言上都很鲁棒。到 2026 年，它是每个 open-source voice assistant 和多数商业 voice assistant 的默认 speech front-end。

## The Concept / 概念

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Step 1 — resample + window / 第 1 步：resample + window

Audio 采样率 16 kHz。Clip/pad 到 30 秒。计算 log-mel spectrogram：80 mel bins，10 ms stride → 约 3,000 frames × 80 features。这就是 Whisper 看到的 “input image”。

### Step 2 — convolutional stem / 第 2 步：convolutional stem

两个 kernel 3、stride 2 的 Conv1D layers，把 3,000 frames 降到 1,500。在不增加太多参数的情况下把 sequence length 减半。

### Step 3 — encoder / 第 3 步：encoder

一个 24-layer（large）transformer encoder，处理 1,500 timesteps。Sinusoidal positional encoding、self-attention、GELU FFN。输出 1,500 × 1,280 hidden states。

### Step 4 — decoder / 第 4 步：decoder

一个 24-layer transformer decoder。它基于 BPE vocabulary autoregressively 生成 tokens；这个 vocabulary 是 GPT-2 的 superset，并加了一些 audio-specific special tokens。

### Step 5 — task tokens / 第 5 步：task tokens

Decoder prompt 以 control tokens 开始，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或：

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按这个 convention 训练的。你通过 prefix 控制 task。它是 2026 年 instruction-tuning 的 speech 版。

### Step 6 — output / 第 6 步：output

Beam search（width 5）配 log-prob threshold。若没有 `<|notimestamps|>` token，timestamps 会按每 0.02 秒音频预测一次。

### Whisper sizes / Whisper 尺寸

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo（2024）把 decoder 从 32 层削到 4 层。Decoding 快 8×，WER regression 小于 1 point。这个 decode speed unlock 正是 Whisper-turbo 成为 2026 年 real-time voice agents 默认选择的原因。

### What Whisper does not do / Whisper 不做什么

- 不做 diarization（谁在说话）。这要配 pyannote。
- 原生不做 real-time streaming，30-second window 是固定的。现代 wrappers（`faster-whisper`、`WhisperX`）通过 VAD + overlap 补上 streaming。
- 没有 external chunking 时，不支持超过 30 s 的 long-form context。实践中效果仍好，因为人类 speech transcription 很少需要长程 context。

### 2026 landscape / 2026 年格局

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine is 4× faster on edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, but Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

## Build It / 动手构建

见 `code/main.py`。我们不训练 Whisper；我们构建 log-mel spectrogram pipeline + task-token prompt formatter。这些才是 production 中你实际会碰到的部分。

### Step 1: synthesize audio / 第 1 步：synthesize audio

生成一个 1 秒、440 Hz、16 kHz sampled 的 sine wave。共 16,000 samples。

### Step 2: log-mel spectrogram (simplified) / 第 2 步：log-mel spectrogram（简化版）

完整 mel spectrogram 需要 FFT。这里用 simplified framing + per-frame energy 版本，不依赖 `librosa`，但能展示 pipeline：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。与 Whisper windowing 匹配。Per-frame energy 在教学上替代 mel bins。

### Step 3: pad to 30 s / 第 3 步：pad 到 30 秒

Whisper 总是处理 30-second chunks。把 spectrogram pad（或 clip）到 3,000 frames。

### Step 4: build the prompt tokens / 第 4 步：构建 prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整 task-control surface。一个 4-token prefix。

## Use It / 应用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，且兼容 OpenAI 风格：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**When to pick Whisper in 2026 / 2026 年何时选择 Whisper：**

- 一个模型覆盖 multilingual ASR。
- 鲁棒转写 noisy、diverse audio。
- Research / prototype ASR 的最快起点。

**When to pick something else / 何时选择别的方案：**

- Edge 上 ultra-low latency streaming：Moonshine 在 matched quality 下击败 Whisper。
- 需要 <200 ms 的 real-time conversational AI：使用 dedicated streaming ASR。
- Speaker diarization：Whisper 不做，需要加 pyannote。

## Ship It / 交付它

见 `outputs/skill-asr-configurator.md`。这个 skill 会为新的 speech application 选择 ASR model、decoding parameters 和 preprocessing pipeline。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。确认 16 kHz、10 ms hop 的 1 秒 signal frame count 约为 100。30 秒约为 3,000 frames。
2. **Medium / 中等。** 使用 `numpy.fft` 构建完整 log-mel spectrogram。验证 80 mel bins 与 `librosa.feature.melspectrogram(n_mels=80)` 在 numerical error 内一致。
3. **Hard / 困难。** 实现 streaming inference：把 audio 切成 10 s windows，带 2 s overlap，对每个 chunk 运行 Whisper，再 merge transcripts。测量 5-minute podcast sample 上与 single-pass 相比的 word-error rate。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mel spectrogram | “Audio image” | 2D representation：一个轴是 frequency bins，另一个轴是 time frames；每格是 log-scaled energy。 |
| Log-mel | “What Whisper sees” | 经过 log 的 mel spectrogram；近似人类对 loudness 的感知。 |
| Frame | “One time slice” | 一个 25 ms samples window；以 10 ms stride 重叠。 |
| Task token | “Prompt prefix for speech” | Decoder prompt 中的 `<\|transcribe\|>` / `<\|translate\|>` 这类 special tokens。 |
| Voice activity detection (VAD) | “Find the speech” | ASR 前移除 silence 的 gate；能大量降低成本。 |
| CTC | “Connectionist Temporal Classification” | 经典 ASR alignment-free training loss；Whisper 不使用它。 |
| Whisper-turbo | “Small decoder, full encoder” | large-v3 encoder + 4-layer decoder；decoding 快 8×。 |
| Faster-whisper | “The production wrapper” | CTranslate2 reimplementation；int8 quantization；比 OpenAI reference 快 4×。 |

## Further Reading / 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — reference code + model weights。阅读 `whisper/model.py`，可以在约 400 行里自顶向下看到 Conv1D stem + encoder + decoder。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — Steps 5–6 描述的 beam-search + task-token logic 在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些场景下仍是 SOTA features。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — production wrapper，比 reference 快 4×。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年 edge-friendly ASR，Whisper-shaped 但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — canonical fine-tuning recipe，包括 mel spectrogram preprocessor 和 token-timestamp handling。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（encoder、decoder、cross-attention、generation），对应本课架构图。
