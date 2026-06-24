# Omni Models: Qwen2.5-Omni and the Thinker-Talker Split / Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在 2024 年 5 月的产品 demo 之所以震撼，不是因为底层模型本身，而是产品形态：一个语音界面，用户说话，模型看着摄像头看到的内容，并在 250ms 内说话回应。Open ecosystem 在 2024 和 2025 年都在追赶这个产品表面。Qwen2.5-Omni（2025 年 3 月）是 reference open design：一个 Thinker（大型文本生成 transformer）加一个 Talker（并行语音生成 transformer），通过 streaming speech tokens 连接。Mini-Omni 简化了它，Moshi 追平 latency，GLM-4-Voice 扩展到中文。本课读取 Thinker-Talker architecture，以及让 streaming real-time dialogue 成立的 latency budget。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites / 前置知识：** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 把 inference pipeline 拆成 Thinker（text reasoning）和 Talker（speech synthesis），并解释为什么 parallel streaming 有效。
- 按组件计算 conversational interaction 的 time-to-first-audio-byte（TTFAB）预算。
- 描述 Thinker 内部跨 vision、audio、text 的 TMRoPE time-aligned position encoding。
- 说出三种 real-time conversational patterns：half-duplex、turn-taking、full-duplex。

## The Problem / 问题

实时语音助手必须快速完成很多事：

1. 听用户。实时 speech tokenization、voice activity detection（VAD）判断用户何时说完。
2. 可选地看。Camera input 以 2-4 FPS stream 进入 Thinker，与 audio 并行。
3. 思考。基于 conversation history 组织回应。
4. 说话。合成 audio tokens，decode 到 waveform，stream 到用户 speaker。

每一步都加 latency。要有对话感，总 round-trip 必须 < 500ms；低于这个阈值，用户才不会明显感到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都必须 stream。不能“攒够一批再 decode”。

## The Concept / 概念

### Thinker and Talker / Thinker 与 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B text-generating transformer。消费 interleaved text + image + audio tokens。输出表示“要说什么”的 text tokens。
- Talker：较小的 speech-generating transformer（200M-1B）。消费 Thinker 的 text output tokens 加 recent speech-context tokens。输出 discrete speech tokens（residual-VQ indices）。
- Speech decoder：streaming waveform decoder（SNAC、MoVQGAN family），实时把 speech tokens 转成 audio samples。

这种分离很重要。Thinker 需要大，才能有好 reasoning。Talker 可以小，因为它的任务局部：把 text 转成 speech tokens。更大的 Talker 不会更有表达力，只会更慢。

并行运行：

1. Thinker 输出 text token t_i。
2. Talker 通过 streaming 消费 t_i，并输出 speech tokens s_i, s_{i+1}, ..., s_{i+k}。
3. Speech decoder 边收到 speech tokens 边输出 audio samples。
4. 当 Thinker 到 text token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} stream 了 audio。

### TMRoPE — time-aligned multimodal positions / TMRoPE：时间对齐的多模态位置

Thinker 需要整合 image frames（例如 4 FPS 到达）、audio frames（50 frames/second 到达）和 conversation history text。朴素序列顺序（先所有 images、再所有 audio、再 text）会丢 temporal alignment。

TMRoPE 为每个 token 分配 absolute timestamp。Vision token at t=2.3s。Audio token at t=2.32s。用户说 “stop” 对应的 text token at t=2.35s。RoPE 按 timestamp 旋转 attention；模型把它们看成同一概念时刻附近的事件。

这就是 “he waved while saying hello” 能成立的基础：模型看到同一时刻的视频帧和音频。

### Streaming speech synthesis / 流式语音合成

Speech tokens 必须 stream。Mini-Omni（Xie & Wu, 2024）提出 “language models can hear, talk while thinking in streaming”：Thinker output tokens 与 Talker output tokens 交错出现在同一序列。Talker 在 Thinker commit 下一个 text token 时立刻触发。没有 batch boundaries。

Moshi（Défossez et al., 2024 年 10 月）是最快 open implementation。单 A100 上 160ms TTFAB。架构是一个 7B transformer，交替位置上输出 text 和 speech tokens，并用 “inner monologue” 分离 thinking stream 与 speaking stream。这本质上是把 Thinker + Talker 融合进一个模型，并通过细致训练控制。

### VAD and turn-taking / VAD 与轮次控制

Voice activity detection 在输入侧运行。两种模式：

- Half-duplex：用户说，模型听；模型说，用户听。通过 VAD silence detection（约 200ms）明确交接。
- Full-duplex：双方可同时说。模型可以 backchannel（“嗯嗯”）或打断。难得多。Moshi 支持。

Qwen2.5-Omni 默认支持 half-duplex，通过 silence threshold 做 turn-taking。Full-duplex 需要应用层处理。

### Qwen3-Omni (November 2025) / Qwen3-Omni（2025 年 11 月）

后继版本。Qwen3-80B Thinker、更大 Talker、改进 TMRoPE-v2。Latency 接近 GPT-4o 的 250ms。Open weights。在 OmniBench 上与 Gemini 2.0 Live 竞争。

### Production latency budget / 生产延迟预算

典型 streaming interaction：

- Mic -> audio tokens：40-80ms。
- Prefill（prompt + history）：7B 上 100-200ms，70B 上更多。
- First Thinker text token：40ms。
- Talker 处理第一个 text token：20ms。
- First speech tokens commit：40ms。
- Residual-VQ decode：30ms。
- Speech waveform decode：50-80ms。

总 TTFAB：7B 上 320-510ms；70B 上 600-900ms。Frontier quality 通常意味着 70B+，这就是 frontier latency gap 的来源。

### Token-rate math / Token 速率数学

16kHz speech、50 Hz base speech tokens 下，每秒输出需要 50 个 speech tokens。Talker 必须 ≥50 tok/s 才能跟上。在 H100 上典型 LLM throughput 是 30-80 tok/s，小型（200-300M）Talker 足够快；7B Talker 会落后。

这就是为什么存在小型专用 Talker，而不是“直接用主模型”。

## Build It / 动手构建

本课构建 streaming pipeline latency simulator 与 VAD loop。你会设置 Thinker/Talker token rates、mic/tokenizer/decoder latency、VAD silence threshold，然后计算 TTFAB 与 turn-taking 行为。

## Use It / 应用它

`code/main.py`：

- 用 mock token-emission rates 模拟 Thinker-Talker pipeline。
- 为可配置 model sizes 和 mic sample rates 计算 TTFAB。
- 用 VAD silence threshold 演示 half-duplex turn-taking。

## Ship It / 交付它

本课产出 `outputs/skill-omni-streaming-budget.md`。给定 real-time voice product 的 target TTFAB 与 feature set（vision-in、bilingual、full-duplex），它会在 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 之间选择，并 sizing Thinker/Talker。

## Exercises / 练习

1. 目标 TTFAB 是 300ms。使用 7B Thinker 与 300M Talker，写出每个组件的 latency。

2. Qwen2.5-Omni 使用 TMRoPE。描述用户在 t=1s 开始说话、摄像头在 t=1.2s 捕捉到手势时，模型看到什么。

3. Full-duplex support 要求模型边听边输出 audio。提出一种 training data format 来教会它。

4. 阅读 Moshi paper Section 4。描述 “inner monologue” separation，以及它为什么能避免 Thinker-Talker split。

5. 计算 throughput budget：16kHz speech、50 base-layer tokens/sec 时，Talker 必须多快才能跟上？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Thinker | “Reasoning brain” | 大型 text-generating transformer，决定要说什么 |
| Talker | “Speech-generating mouth” | 小型 transformer，从 Thinker text 生成 discrete speech tokens |
| TTFAB | “Latency budget” | Time-to-first-audio-byte：从用户语音结束到首个 audio sample 输出 |
| TMRoPE | “Time-aligned RoPE” | 使用 absolute timestamps 跨 vision、audio、text 编码位置 |
| Half-duplex | “Turn-taking” | 用户与模型轮流说话；VAD silence 判断用户完成 |
| Full-duplex | “Simultaneous” | 模型可同时听和说；支持 backchannel |
| Inner monologue | “Moshi separation” | 单模型设计中 thinking-stream 与 speaking-stream 交错分离 |

## Further Reading / 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
