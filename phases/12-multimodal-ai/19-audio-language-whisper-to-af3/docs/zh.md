# Audio-Language Models: the Whisper to Audio Flamingo 3 Arc / 音频语言模型：从 Whisper 到 Audio Flamingo 3

> Whisper（Radford et al., 2022 年 12 月）基本解决了 speech recognition：680k 小时 weakly-supervised multilingual speech，一个简单 encoder-decoder transformer，以及一个让后续 ASR release 都必须引用的 benchmark。但识别不是推理。问“这段录音里有什么乐器”“说话人的情绪是什么”“第 3 分钟发生了什么”，需要 audio understanding，而不是 transcription。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3, 2025 年 7 月）逐步构建这个 stack：保留 Whisper-class encoders，接 Q-formers，在 audio-text instruction data 上训练，并加入 chain-of-thought reasoning。本课沿着这条弧线讲。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites / 前置知识：** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 从 waveform 计算 log-Mel spectrogram：windowing、FFT、filter banks、log transform。
- 比较 encoder 选择：Whisper encoder、BEATs、AF-Whisper hybrid。说明各自何时胜出。
- 构建 audio Q-former：N 个 learnable queries cross-attend 到 spectrogram patches。
- 解释 cascaded（Whisper-then-LLM）与 end-to-end audio-LLM training 的差异，以及为什么 end-to-end 更适合 reasoning scale。

## The Problem / 问题

Speech recognition 已被 Whisper 基本解决。Audio 的 OCR 已经 commodity。但“commodity”停留在 transcription。模型如果无法对听到的内容推理：timing、speakers、emotion、music structure、environmental sounds，transcript alone 就无法驱动产品功能。

三条明显路线：

1. Cascade：Whisper 转写，LLM 对 transcript 推理。适合纯 speech 场景。对 music、environmental audio、multi-speaker overlap、emotion 失败。

2. End-to-end audio-LLM：audio encoder 直接把 audio tokens 喂给 LLM，跳过 transcription。保留 acoustic information（emotion、speaker、environment）。需要新训练数据。

3. Hybrid：audio encoder + text decoder，既能 transcribe 也能 reason。Qwen-Audio 和 Audio Flamingo 选择这条路线。

## The Concept / 概念

### Log-Mel spectrogram: the input feature / Log-Mel spectrogram：输入特征

每个 audio encoder 都从同一特征开始：log-Mel spectrogram。

1. Resample 到 16 kHz。
2. 25ms windows、10ms hop 做 short-time Fourier transform。
3. 取 FFT result 的 magnitude。
4. 应用 Mel filter banks（通常 80 个 filters，在 0-8000 Hz 上 log-spaced），warp 到感知频率。
5. Log compress（log(1 + x)）以压缩 dynamic range。

结果是形状为 `(T, 80)` 的 2D array，其中 T 是 time frames 数。30 秒 clip，100 Hz frame rate 下是 `(3000, 80)`。

### Whisper's encoder / Whisper encoder

Whisper encoder 是一个 12-layer ViT-style transformer，把 log-Mel spectrogram 当作 time frames 序列处理。输出：每个 time frame 一个 hidden-state vector。

ASR 中，Whisper decoder 是 cross-attention transformer，在 encoder output 条件下生成 text tokens。标准 encoder-decoder。

对 ALMs（audio-LLMs），你想把 encoder output 作为另一个 LLM 的输入。模式是：Whisper encoder frozen，Q-former trainable，LLM frozen 或 tuned。

### BEATs and audio-specific encoders / BEATs 与音频专用 encoder

Whisper 训练数据以 speech 为主。对 music 和 environmental audio 较弱。

BEATs（Chen et al., 2022）是在 AudioSet 上训练的 self-supervised transformer。在同等参数量下，比 Whisper 更擅长 music 和 environmental sounds。

AF-Whisper（Audio Flamingo 3 的 hybrid）：concat Whisper + BEATs features 作为 audio input。Whisper 承载 linguistic signal，BEATs 承载 acoustic signal。

### Audio Q-former / Audio Q-former

模式与 BLIP-2 的 visual Q-former 相同。一组固定 learnable queries（常见 32 或 64）对 audio encoder 的 output frames 做 cross-attention。Queries 成为 LLM 消费的 audio tokens。

Training alignment stage：只训练 Q-former，在 audio-text pairs（AudioCaps、Clotho）上做 contrastive + captioning losses。Instruction stage：end-to-end，unfreeze LLM，在 instruction data 上训练。

### The arc — SALMONN, Qwen-Audio, AF3 / 演进线：SALMONN、Qwen-Audio、AF3

SALMONN（Tang et al., 2023）：Whisper + BEATs + Q-former + LLaMA。第一个有认真 reasoning ability 的 open audio-LLM。MMAU composite 约 0.55。

Qwen-Audio（Chu et al., 2023）：类似架构，但数据更丰富，面向 multi-turn dialogue 调优。MMAU 约 0.60。

LTU — Listen, Think, Understand（Gong et al., 2023）：显式 reasoning data，聚焦 audio clips 上的 chain-of-thought。更小但更专注。

Audio Flamingo 3（Goel et al., 2025 年 7 月）：当前 open SOTA。8B LLM backbone（Qwen2 7B）、Whisper-large encoder concat BEATs、64-query Q-former，在 1M+ audio-text instruction pairs 上训练。MMAU 0.72，在部分 sub-tasks 上追平 proprietary frontier。

AF3 还引入 on-demand chain-of-thought for audio：模型可选地先输出 thinking tokens（“let me identify the instruments first: ...”），再给 final answer。复杂 reasoning tasks 上开启 thinking 可提升 3-5 点。

### Cascaded vs end-to-end / 级联 vs 端到端

Cascaded pipeline：

1. Whisper 转写 audio → text。
2. LLM 基于 text 推理。

对 “summarize this podcast” 完全可用。对以下问题失败：

- “What's the mood of this song?”——mood 在声音中，不在文字里。
- “Who is speaking, Alice or Bob?”——需要 speaker identification。
- “At what second does the explosion happen?”——transcript 中丢失 temporal grounding。
- “Is this real or generated audio?”——deepfake detection 需要 acoustic features。

End-to-end 保留 acoustic signal。Qwen-Audio 和 AF3 原生处理 music、environment、emotion。

### 2026 production recipe / 2026 生产配方

新 audio-understanding 产品：

- Cascaded if：目标是 transcription，没有 music，也没有 emotion inference。
- AF3 / Qwen-Audio-family if：涉及 music、emotion、multi-speaker 或复杂 audio reasoning。

Cascaded 更便宜、更简单。End-to-end 更强。

### MMAU — the audio reasoning benchmark / MMAU：音频推理 benchmark

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年 audio reasoning benchmark：

- 10,000 个 audio-text QA pairs，覆盖 speech、music、environmental sounds。
- 涵盖 classification、temporal reasoning、causal reasoning、open-ended QA。
- 专门测试 cascaded pipelines 系统性遗漏的能力。

Open SOTA（AF3）约 0.72；proprietary frontier 约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的 open-vs-closed delta，说明 audio-LLMs 正在成熟。

## Build It / 动手构建

本课动手实现 log-Mel spectrogram 与 audio Q-former skeleton：先从 waveform 做窗口、DFT、Mel filter bank 和 log compression，再用 learnable queries 对 encoder frames 做 cross-attention，得到固定长度 audio tokens。

## Use It / 应用它

`code/main.py`：

- 用 stdlib 实现 log-Mel spectrogram：windowing、naive DFT、Mel filter-bank。
- Audio Q-former skeleton：给定 encoder output frames，计算 Q、K、V、attention，并输出 N tokens。
- 在 toy task 上比较 cascaded-vs-end-to-end。

## Ship It / 交付它

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定 audio task（transcription、music tagging、emotion inference、multi-speaker diarization、environment classification），它会在 cascaded、end-to-end AF3 或 hybrid 之间选择。

## Exercises / 练习

1. 对 16kHz、30 秒 clip、25ms window、10ms hop、80 Mel bins，计算 log-Mel spectrogram dimension。48kHz 时如何变化？

2. 为什么 Whisper 在 music 上表现较弱？BEATs 捕捉了 Whisper 不捕捉的哪些 audio features？

3. Audio Q-former 使用 64 queries vs 32。什么任务复杂度下 64 值得？32 对什么任务省 compute？

4. 阅读 AF3 Section 4 关于 on-demand thinking。提出三个 chain-of-thought 最有帮助的 audio tasks。

5. 用 AF3 输出实现一个 minimal diarization pipeline。如何标记 speaker changes？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Log-Mel spectrogram | “Mel features” | Mel filter banks 后的 log-magnitude 2D `(time, frequency)` array |
| Audio Q-former | “Audio Perceiver” | 从 audio encoder output 到固定长度 queries 的 cross-attention bottleneck，喂给 LLM |
| Cascaded | “ASR-then-LLM” | Whisper 先转写，text LLM 再推理；会丢 acoustic information |
| End-to-end | “Audio-LLM” | Audio features 通过 Q-former 直接进入 LLM；保留 acoustic signal |
| BEATs | “Audio AudioSet encoder” | 在 AudioSet 上训练的 SSL transformer；擅长 music + environmental sounds |
| MMAU | “Audio reasoning bench” | 覆盖 speech、music、environment 的 10k QA pairs；2024 eval standard |
| On-demand thinking | “Audio CoT” | 模型可选地先输出 reasoning tokens，再给 final answer，accuracy 提升 3-5 点 |

## Further Reading / 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
