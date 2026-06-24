# Whisper — Architecture & Fine-Tuning / Whisper：架构与微调

> Whisper 是一个 30 秒窗口的 transformer encoder-decoder，在 68 万小时多语言弱监督 audio-text pairs 上训练。一个架构，多种任务，覆盖 99 种语言且鲁棒。它是 2026 年的参考级 ASR。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 拆解 Whisper 的 30-second log-mel input、encoder、decoder、special token prompt 和 BPE 输出
- 正确处理 long-form、chunked 和 near-streaming Whisper inference
- 判断何时 fine-tune Whisper，以及如何用 LoRA 控制显存和过拟合风险
- 识别 VAD、mel preprocessing、prompt 选项和 short-clip padding 引发的常见生产问题

## The Problem / 问题

OpenAI 在 2022 年 9 月发布的 Whisper，是第一个像商品一样可用的 ASR 模型：贴入音频，得到文本，支持 99 种语言，抗噪，笔记本也能跑。到 2024 年 OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已是从 podcast transcription 到 voice assistants 再到 YouTube subtitles 的默认 baseline。

但 Whisper 不能永远当黑盒 pipeline 使用。Domain shift 会击穿它：技术术语、说话人口音、专有名词、短音频、静音都会出问题。你需要知道：

1. 它内部到底是什么。
2. 如何正确喂给它 chunked、streaming 或 long-form audio。
3. 什么时候要 fine-tune，以及怎么做。

## The Concept / 概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Architecture / 架构。** 标准 transformer encoder-decoder。

- Input：30 秒 log-mel spectrogram，80 mels，10 ms hop → 3000 frames。更短的 clip 会 zero-padded，更长的 clip 会 chunked。
- Encoder：conv-downsample（stride 2）+ `N` 个 transformer blocks。Large-v3：32 layers，1280-dim，20 heads。
- Decoder：`N` 个 transformer blocks，包含 causal self-attn + 对 encoder output 的 cross-attn。尺寸与 encoder 相同。
- Output：基于 51,865-token vocab 的 BPE tokens。

Large-v3 有 1.55B params。Turbo 把 decoder 从 32 层降到 4 层，延迟降低 8×，WER 损失小于 1%。

**The prompt format / Prompt 格式。** Whisper 是一个 multitask model，由 decoder prompt 中的 special tokens 控制：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — language tag；控制 translation-vs-transcription 行为。
- `<|transcribe|>` or `<|translate|>` — 对任意语言输入输出英文翻译，或逐字转写。
- `<|notimestamps|>` — 跳过 word-level timestamps（更快）。

Prompt 让一个模型能执行多种任务。把 `<|en|>` 改成 `<|fr|>`，它就会转写法语。

**30-second window / 30 秒窗口。** 一切都绑定到 30 秒。更长的 clip 需要 chunking；更短的 clip 会 padding。窗口不是原生 streaming 的，这也是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel normalization / Log-mel 归一化。** `(log_mel - mean) / std`，统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的 preprocessing（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### Variants in 2026 / 2026 年的变体

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning / 微调

2026 年的 canonical workflow：

1. 收集 10–100 小时目标领域音频，并配 aligned transcripts。
2. 用 `transformers.Seq2SeqTrainer` 运行训练，配 `generate_with_loss` callback。
3. 参数高效做法：在 attention layers 的 `q_proj`、`k_proj`、`v_proj` 上做 LoRA，GPU memory 降低 4×，WER 代价小于 0.3。
4. 如果你只有 <10 小时数据，freeze encoder，只 tune decoder。
5. 使用 Whisper 自己的 tokenizer 和 prompt format；不要替换 tokenizer。

社区结果：在 20 小时医学听写数据上 fine-tune Medium，可以把医学词汇 WER 从 12% 降到 4.5%。在 4 小时冰岛语数据上 fine-tune Turbo，可以把 WER 从 18% 降到 6%。

## Build It / 动手构建

### Step 1: run Whisper out of the box / 第 1 步：直接运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该总是覆盖的关键默认值：`temperature=0.0`（采样默认走 0.0 → 0.2 → 0.4 … fallback chain），`condition_on_previous_text=False`（避免级联 hallucination），以及 `no_speech_threshold=0.6`（静音检测）。

### Step 2: chunked long-form / 第 2 步：chunked long-form

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了三件事：(1) Silero VAD gating，(2) 通过 wav2vec 2.0 做 word-level alignment，(3) 通过 `pyannote.audio` 做 diarization。它是 2026 年 production transcription 的主力。

### Step 3: fine-tune with LoRA / 第 3 步：用 LoRA fine-tune

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后使用标准 Trainer loop。每 1000 steps 保存 checkpoint。在 held-out 集上用 WER 评估。

### Step 4: inspect what each layer learns / 第 4 步：观察每层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用 heatmap 可视化，你会看到 decoder steps 扫过 encoder frames 时形成的 diagonal alignment。这条对角线就是 Whisper 对 word timestamps 的理解。

## Use It / 应用它

2026 年的 stack：

| Situation | Pick |
|-----------|------|
| General English, offline | Large-v3-turbo via `whisperx` |
| Mobile / edge | Whisper-Tiny quantized (int8) or Moonshine |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
| Low-resource language | Fine-tune Medium or Turbo with LoRA |
| Streaming (2 s latency) | Whisper-Streaming or Parakeet-TDT |
| Word-level timestamps | WhisperX (forced alignment via wav2vec 2.0) |

`faster-whisper`（CTranslate2 backend）是 2026 年最快的 CPU+GPU inference runtime，比 vanilla 快 4×，输出一致。

## Pitfalls that still ship in 2026 / 2026 年仍会上线的坑

- **Hallucinated text on silence / 静音上产生幻觉文本。** Whisper 在 captions 上训练过，会输出 "Thanks for watching!"、"Subscribe!"、歌词。调用前一定要做 VAD-gate。
- **`condition_on_previous_text` cascade / 上下文级联。** 一次 hallucination 会污染后续窗口。除非你需要跨 chunks 的流畅性，否则设为 `False`。
- **Short-clip padding / 短音频 padding。** 2 秒 clip padding 到 30 秒，会在尾部静音中 hallucinate。使用 `pad=False` 或 VAD-gate。
- **Wrong mel stats / Mel 统计量错误。** 用 librosa 的 mels 替代 Whisper 的 mels 会产生近似随机输出。使用 `whisper.audio.log_mel_spectrogram`。

## Ship It / 交付它

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper fine-tune 或 inference pipeline。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会 tokenize 一个 Whisper-style prompt，计算 decoded shape budgets，并打印 10-minute clip 的 chunk schedule。
2. **Medium / 中等。** 安装 `faster-whisper`，转写一段 10-minute podcast，并与 human transcript 比较 WER。尝试 `language="auto"` 与强制 `language="en"`。
3. **Hard / 困难。** 使用 HF `datasets`，选择一种 Whisper 表现困难的语言（例如 Urdu），在 2 小时数据上用 LoRA fine-tune Medium 2 个 epochs，并报告 WER delta。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 30-sec window | Whisper 的限制 | 硬输入上限；更长音频要 chunk。 |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` 启动 decoder prompt。 |
| Timestamps token | 时间对齐 | 51k vocab 中每 0.02 s offset 都有一个 special token。 |
| Turbo | 快速变体 | 4 层 decoder，快 8×，WER regression <1%。 |
| WhisperX | long-form wrapper | VAD + Whisper + wav2vec alignment + diarization。 |
| LoRA fine-tune | 高效微调 | 给 attention 加 low-rank adapters；训练约 0.3% params。 |
| Hallucination | 静音失败模式 | Whisper 从噪声/静音中生成流畅英文。 |

## Further Reading / 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构和训练 recipe。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层 decoder，8× speedup。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form、word-aligned、diarized。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-backed，快 4×。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — canonical LoRA / full-FT walkthrough。
