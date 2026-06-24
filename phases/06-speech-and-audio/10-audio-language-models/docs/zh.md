# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio / 音频语言模型：Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的 audio-language models 能对 speech + environmental sound + music 做推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上匹配 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上超过 Gemini 2.5 Pro。开放模型和闭源模型之间的差距基本关闭，除了 multi-audio tasks：所有人都接近随机。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分纯 ASR 与 LALM / ALM 的输入、输出和推理能力
- 理解 audio encoder、projector、LLM decoder 三段式模板及三阶段训练流程
- 用 Qwen2.5-Omni 查询音频，并实现一个最小 AudioProjector
- 针对 speech、sound、music、multi-audio 和 long-audio 场景选择模型与 benchmark subset

## The Problem / 问题

你有 5 秒音频：狗叫、有人喊 “stop!”，然后安静。真正有用的问题横跨多个轴：

- **Transcription / 转写。** “说了什么？”—— ASR 领域。
- **Semantic reasoning / 语义推理。** “这个人有危险吗？”—— 需要联合理解狗叫、喊声和沉默。
- **Music reasoning / 音乐推理。** “哪些乐器在演奏旋律？”
- **Long-audio retrieval / 长音频检索。** “这段 90 分钟讲座里，老师在哪里解释 gradient descent？”

能用一个 prompt 回答所有这些问题的单一模型，就是 **audio-language model**（LALM / ALM）。它不同于纯 ASR：LALMs 输出自由形式自然语言答案，而不只是 transcript。

## The Concept / 概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### The three-component template / 三组件模板

每个 2026 年 LALM 都有同一个骨架：

1. **Audio encoder。** Whisper encoder · BEATs · CLAP · WavLM · 或每个模型自定义 encoder。
2. **Projector。** Linear 或 MLP，把 audio-encoder features 桥接到 LLM 的 token embedding space。
3. **LLM。** 基于 Llama / Qwen / Gemma 的 decoder。接收交错的 text + audio tokens，生成文本。

训练：

- **Stage 1。** Freeze encoder + LLM；只在 ASR / captioning data 上训练 projector。
- **Stage 2。** 在 instruction-following audio tasks（QA、reasoning、music understanding）上做 full / LoRA fine-tune。
- **Stage 3 (optional)。** Voice-in / voice-out 会增加 speech decoder。Qwen2.5-Omni 和 AF3-Chat 会这样做。

### The 2026 model map / 2026 模型地图

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark reality check (2026) / Benchmark 现实检查（2026）

**MMAU-Pro。** 1800 QA pairs，覆盖 speech / sound / music / mixed。包含 multi-audio subset。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**multi-audio 列对所有模型都很难看。** 4-option multiple choice 的随机机会是 25%；多数模型就在这个附近。LALMs 仍然很难比较两段音频。

### Where LALMs are useful in 2026 / 2026 年 LALM 有用的地方

- **Call-center recordings 的 compliance audit。** “客服是否提到了必要披露？”
- **Accessibility / 无障碍。** 向 deaf users 描述 sound events，而不仅仅是 transcription。
- **Content moderation / 内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **Podcast / meeting chaptering。** 做 semantic summary，而不只是 speaker turns。
- **Music catalog analysis / 音乐库分析。** “找出所有带 B-section key change 的曲目。”

### Where they are NOT (yet) useful / 暂时还不适合的地方

- 细粒度 music theory（低于 chord-level）。
- 长对话中的 speaker-attributed reasoning（超过 10 分钟后退化）。
- Multi-audio comparison（22–26% 几乎只是略高于随机）。
- Real-time streaming reasoning（多数仍是 offline batch inference）。

## Build It / 动手构建

### Step 1: query Qwen2.5-Omni / 第 1 步：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2: the projector pattern / 第 2 步：projector pattern

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。Projector 通常是 1–3 层 linear layers。在 ASR pairs（audio → transcript）上训练它，就是 Stage-1 pretext task。

### Step 3: benchmarking MMAU / LongAudioBench / 第 3 步：benchmarking MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告 per-category（speech / sound / music / multi-audio）。Aggregate numbers 会掩盖模型在哪里失败。

## Use It / 应用它

| Task | 2026 pick |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni or GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 or 2 (music-specialized AF-CLAP) |
| Call-center audit | Gemini 2.5 Pro via API, with RAG over your policy docs |

## Pitfalls / 常见坑

- **Over-trust on multi-audio / 过度相信 multi-audio。** 如果你的任务需要“哪个 clip 有 X”，随机水平的表现是真实存在的。
- **Long-audio degradation / 长音频退化。** 超过 10 分钟后，多数模型的 speaker attribution 会崩。先做 diarize（第 6 课），再总结。
- **Hallucinations on silence / 静音幻觉。** 使用 Whisper encoder 的 LALMs 会继承 Whisper 风格问题。先 VAD-gate。
- **Benchmark cherry-picking / 挑 benchmark。** 厂商 blog post 会突出最佳类别。自己跑 MMAU-Pro multi-audio subset。

## Ship It / 交付它

保存为 `outputs/skill-alm-picker.md`。为给定 audio-understanding task 选择 LALM + benchmark subset + output-modality（text vs speech）。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`，观察 toy projector pattern + fake LALM routing，即 (audio-embedding, text-tokens) → output tokens。
2. **Medium / 中等。** 在 100 个 MMAU-Pro speech items 上打分 Qwen2.5-Omni-7B。与论文报告数字比较。
3. **Hard / 困难。** 构建一个最小 audio-captioning baseline：BEATs encoder + 2-layer projector + frozen Llama-3.2-1B。只在 AudioCaps 上 fine-tune projector。与 Clotho-AQA 上的 SALMONN 比较。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder。 |
| Projector | Adapter | 小型 MLP，把 audio features 映射进 LLM embedding space。 |
| MMAU | benchmark | 跨 speech、sound、music 的 10k audio-QA pairs。 |
| MMAU-Pro | 更难的 MMAU | 1800 个 multi-audio / reasoning-heavy questions。 |
| LongAudioBench | long-form eval | 带 semantic queries 的多分钟 clips。 |
| Voice-in / voice-out | speech-native | 模型摄入 speech 并输出 speech，不走 text detour。 |

## Further Reading / 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源 long-audio leader。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 live rankings。
