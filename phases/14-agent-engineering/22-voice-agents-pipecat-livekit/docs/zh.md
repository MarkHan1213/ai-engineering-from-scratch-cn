# Voice Agents: Pipecat and LiveKit / 语音 Agent：Pipecat 与 LiveKit

> 到 2026 年，语音 Agent 已经是生产系统中的一类一等公民。Pipecat 提供基于 Python frame 的流水线（VAD → STT → LLM → TTS → transport）。LiveKit Agents 则通过 WebRTC 把 AI 模型连接到用户。高端生产栈的端到端延迟目标通常落在 450–600ms。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 01（Agent Loop）, 第 14 阶段 · 12（Workflow Patterns）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 Pipecat 的 frame-based pipeline：DOWNSTREAM（source→sink）和 UPSTREAM（control）。
- 说出典型语音流水线的阶段，以及 Pipecat 支持哪些 transports。
- 解释 LiveKit Agents 的两类语音 Agent（MultimodalAgent、VoicePipelineAgent）以及各自适用场景。
- 总结 2026 年生产语音 Agent 的延迟预期，以及这些预期如何影响架构选择。

## The Problem / 问题

语音 Agent 不是“文本循环外面接一个 TTS”。它的延迟预算非常残酷（约 600ms），partial audio 是常态，turn detection 本身就是一个模型问题，transport 也可能从 telephony SIP 到 WebRTC 不等。要么你构建 frame-based pipeline（Pipecat），要么你依赖一个平台（LiveKit）。

## The Concept / 概念

### Pipecat (pipecat-ai/pipecat)

- 面向 Python 的 frame-based pipeline framework。
- `Frame` → `FrameProcessor` 链。
- 两个流动方向：
  - **DOWNSTREAM** — source → sink（audio in, TTS out）。
  - **UPSTREAM** — feedback and control（cancellation、metrics、barge-in）。
- `PipelineTask` 通过事件（`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`）管理生命周期，并通过 observers 接入 metrics/tracing/RTVI。

典型流水线：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows 增加结构化对话能力（state machines）。Pipecat Cloud 是它的托管运行时。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 把 AI 模型连接到用户。
- 关键概念：`Agent`, `AgentSession`, `entrypoint`, `AgentServer`。
- 两类语音 Agent：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或同类能力直接处理音频。
  - **VoicePipelineAgent** — STT → LLM → TTS cascade；提供文本级控制。
- 通过 transformer model 做 semantic turn detection。
- 原生 MCP 集成。
- 通过 SIP 支持 telephony。
- 借助 LiveKit Inference 可无 API keys 使用 50+ models；再通过 plugins 接入 200+ models。

### Commercial platforms / 商业平台

Vapi（优化后的高端栈约 450–600ms）和 Retell（180 次测试调用中约 600ms end-to-end）构建在这类能力之上。当你想要托管语音栈，但没有 WebRTC 团队时，选平台更现实。

### Where this pattern goes wrong / 这种模式容易出错的地方

- **No barge-in handling.** 用户打断时，Agent 还在继续说。Pipecat 需要 UPSTREAM cancel frames，LiveKit 中也需要等价机制。
- **STT confidence ignored.** 低置信 transcript 被当成事实喂给 LLM。应按 confidence 做 gate，或者请求用户确认。
- **TTS mid-sentence cutoff.** pipeline 在一句话中途 cancel 时，TTS 必须知道该停止，否则会切出不完整音频。
- **Latency budget ignored.** 每个组件都会增加 50–200ms。发布前必须把整条链路加总。

### Typical 2026 latencies / 2026 年典型延迟

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

端到端 450–600ms 属于高端水平。800–1200ms 很常见。任何 > 1500ms 的体验都会像坏掉了一样。

## Build It / 动手构建

`code/main.py` 是一个 frame-based toy pipeline，包含：

- `Frame` 类型（audio, transcript, text, tts_audio, control）。
- 带 `process(frame)` 的 `Processor` interface。
- 一个五阶段流水线（VAD → STT → LLM → TTS → transport），用 scripted processors 实现。
- 一个 UPSTREAM cancel frame，用来演示 barge-in。

运行：

```
python3 code/main.py
```

trace 会展示正常流程，以及一次让 TTS 在 utterance 中途停止的 barge-in cancel。

## Use It / 应用它

- **Pipecat** 适合完全控制：custom processors、Python-first、pluggable providers。
- **LiveKit Agents** 适合 WebRTC-first 部署和 telephony。
- **Vapi / Retell** 适合没有 WebRTC 团队、想要 hosted voice agents 的场景。
- **OpenAI Realtime / Gemini Live** 适合直接 audio-in/audio-out（MultimodalAgent）。

## Ship It / 交付它

`outputs/skill-voice-pipeline.md` 会搭出一个 Pipecat 形态的语音流水线：VAD + STT + LLM + TTS + transport，并包含 barge-in handling。

## Exercises / 练习

1. 给你的 toy pipeline 增加 metrics observer：统计每个 stage 每秒处理的 frames。延迟在哪里累积？
2. 实现 confidence-gated STT：低于阈值时请求“could you repeat that?”
3. 增加 semantic turn detection：简单规则是，如果 transcript 以 “?” 结尾，就认为 turn 结束。
4. 阅读 Pipecat 的 transport docs。把 stdlib transport 替换为 SmallWebRTCTransport config（stub）。
5. 在同一个 query 上测量 OpenAI Realtime 与 STT+LLM+TTS cascade。文本级控制带来了多少延迟成本？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Frame | “Event” | pipeline 中带类型的数据单元（audio, transcript, text, control） |
| Processor | “Pipeline stage” | 带 process(frame) 的 handler |
| DOWNSTREAM | “Forward flow” | 从 source 到 sink：audio in, speech out |
| UPSTREAM | “Feedback flow” | 控制流：cancel, metrics, barge-in |
| VAD | “Voice activity detection” | 检测用户何时正在说话 |
| Semantic turn detection | “Smart end-of-turn” | 基于模型判断用户是否说完 |
| MultimodalAgent | “Direct audio agent” | audio in, audio out；中间没有 text |
| VoicePipelineAgent | “Cascade agent” | STT + LLM + TTS；提供文本级控制 |

## Further Reading / 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline, processors, transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitives
- [Vapi](https://vapi.ai/) — managed voice platform
- [Retell AI](https://www.retellai.com/) — managed voice, latency-benchmarked
