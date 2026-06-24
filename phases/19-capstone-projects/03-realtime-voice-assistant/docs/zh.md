# Capstone 03 — Real-Time Voice Assistant (ASR to LLM to TTS) / 实时语音助手（ASR 到 LLM 到 TTS）

> 一个“感觉对”的语音 Agent，需要端到端延迟低于 800ms，知道你什么时候说完，支持 barge-in，并且能在不中断语音体验的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这条线。它们的形态一致：streaming ASR、turn-detector、streaming LLM、streaming TTS，全都通过 WebRTC 串起来，并且每一跳都有激进的 latency budget。构建一个这样的系统，测量 WER、MOS 和 false-cutoff rate，再在 packet loss 下运行它。

**类型：** 综合项目
**语言：** Python（agent + pipeline）, TypeScript（web client）
**前置知识：** 第 06 阶段（speech and audio）, 第 07 阶段（transformers）, 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 14 阶段（agents）, 第 17 阶段（infrastructure）
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**时间：** 30 小时

## Learning Objectives / 学习目标

- 构建从 WebRTC audio in 到 ASR、LLM、TTS audio out 的端到端 streaming pipeline
- 组合 VAD 和 turn-detector，降低 false-cutoff，同时控制 first-audio-out latency
- 实现 barge-in、工具 side channel 和 backpressure 策略
- 量化 WER、MOS、false-cutoff rate、packet loss 稳定性和并发负载能力
- 交付一个可按业务域配置的 LiveKit / Pipecat 语音 Agent skill

## Problem / 问题

语音是 2025-2026 年增长最快的 AI UX 类别。技术上限几乎每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70，都让 sub-800ms first-audio-out 变得可达。但门槛不只是延迟，而是交互手感：不要打断用户、不要被错误打断、能从半句话插话中恢复，能在对话中调用工具且不让音频卡住，还要能承受抖动明显的移动网络。

把三个 REST call 简单拼起来无法达到这个体验。架构必须从头到尾都是 pipelined streaming。亲手构建后，失败模式会很清楚：VAD 针对电话音频调参后会被背景电视触发；turn-detector 等一个永远不会出现的标点；TTS 在发出首个音频块前 buffer 了 400ms。本 capstone 的任务，是在负载下逐个修掉这些问题，并发布 latency-and-quality report。

## Concept / 概念

pipeline 有五个 streaming stages：**audio in**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（Deepgram Nova-3 或 faster-whisper 的 streaming partial transcripts）、**turn detection**（VAD 加一个读取 partial transcripts 的小型 turn-detector model，用于判断表达是否完成）、**LLM**（一旦 turn 被判定完成就开始 streaming tokens）、**TTS**（在第一个 LLM token 后约 200ms 内开始 streaming audio out）。

三个横切关注点。**Barge-in**：用户在 Agent 说话时开始说话，TTS 立即取消，ASR 立刻接管。**Tool use**：对话中的 function calls（weather、calendar）必须在 side channel 上运行，不能阻塞音频；如果延迟超过 300ms，Agent 先补一句确认 token（“one second...”）。**Backpressure**：在 packet loss 下，partial transcripts 会被保留，VAD 提高 speech-gate threshold，Agent 避免盖过未确认的消息。

测量标准是定量的：15 dB SNR 的 Hamming VAD benchmark 上 WER 低于 8%；100 通测量电话的 first-audio-out p50 低于 800ms；false-cutoff rate 低于 3%；TTS MOS 高于 4.2；单台 g5.xlarge 支撑 50 路并发电话。这些数字就是交付物的一部分。

## Architecture / 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack / 技术栈

- Transport: LiveKit Agents 1.0（WebRTC）加 Twilio PSTN gateway；Pipecat 0.0.70 作为 alternate framework
- ASR: Deepgram Nova-3（streaming，sub-300ms first partial）或自托管 faster-whisper Whisper-v3-turbo
- VAD: Silero VAD v5 加 LiveKit turn-detector（读取 partial transcripts 的小 transformer）
- LLM: OpenAI GPT-4o-realtime（紧耦合集成）、Gemini 2.5 Flash Live，或 cascaded Claude Haiku 4.5（streaming completions，独立 audio path）
- TTS: Cartesia Sonic-2（最低 first-byte）、ElevenLabs Flash v3，或 self-host 的 open-source Orpheus
- Tools: weather / calendar / booking 通过 FastMCP side-channel；工具耗时 >300ms 时 Agent 预先发 filler
- Observability: OpenTelemetry voice spans，Langfuse voice traces with audio replay
- Deployment: self-hosted Whisper + Orpheus 使用单台 g5.xlarge（24GB VRAM）；追求最低延迟时使用 hosted APIs

## Build It / 动手构建

1. **WebRTC session.** 建一个 LiveKit room 和一个能 streaming microphone audio 的 web client。服务端挂一个加入 room 的 agent worker。

2. **ASR streaming.** 把 20ms PCM frames 喂给 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅 partial 和 final transcripts。记录每个 partial 的 latency。

3. **VAD and turn detector.** 在 frame stream 上跑 Silero VAD v5。speech-end event 出现时，用最新 partial transcript 触发 LiveKit turn-detector。只有当 VAD 判定 silence 达 500ms 且 turn-detector completion score > 0.6 时，才提交 “turn complete”。

4. **LLM stream.** turn complete 后，用 running conversation 和 final transcript 启动 LLM call。流式输出 tokens。第一个 token 到达时交给 TTS。

5. **TTS stream.** Cartesia Sonic-2 把音频 chunks 流回来。第一个 chunk 必须在第一个 LLM token 后 200ms 内离开服务器。把 chunks 发到 LiveKit room；client 通过 WebRTC jitter buffer 播放。

6. **Barge-in.** 当 VAD 在 TTS 播放时检测到新的用户语音，立即取消 TTS stream，丢弃剩余 LLM output，并重新 arm ASR。发布一个 `tts_canceled` span。

7. **Tool side channel.** 注册 weather 和 calendar function-calling tools。工具被调用时并发触发；如果 300ms 内没有返回，让 LLM 先输出 “one second, let me check” 作为 filler；工具返回后继续。

8. **Eval harness.** 录制 100 通电话。计算 WER（对 held-out transcript）、false-cutoff rate（用户还在说话时 TTS 被取消）、first-audio-out p50、TTS MOS（人工或 NISQA），以及 jitter-loss test（丢弃 3% packets）。

9. **Load test.** 用 synthetic caller 在单台 g5.xlarge 上驱动 50 路并发电话。测量持续 first-audio-out p95。

## Use It / 应用它

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Ship It / 交付它

`outputs/skill-voice-agent.md` 是交付物。给定一个 domain（customer support、scheduling 或 kiosk），它会搭起 LiveKit agent，并把 ASR/VAD/LLM/TTS pipeline 调到测量标准。评分标准：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | End-to-end latency | 100 通录音电话中 p50 first-audio-out 低于 800ms |
| 20 | Turn-taking quality | Hamming VAD benchmark 上 false-cutoff rate 低于 3% |
| 20 | Tool-use correctness | 对话中工具调用返回正确数据且不阻塞音频 |
| 20 | Reliability under packet loss | 注入 3% packet drop 后 WER 和 turn-taking 稳定性 |
| 15 | Eval harness completeness | 使用公开配置可复现的测量 |
| **100** | | |

## Exercises / 练习

1. 把 Deepgram Nova-3 换成 g5.xlarge 上的 faster-whisper v3 turbo。测量 latency 和 WER 差距。指出 CPU-vs-GPU 决策在哪里重要。

2. 添加 interruption-arbitration policy：当用户在工具调用期间 barge in，Agent 怎么办？比较三种策略（hard cancel、finish-tool-then-stop、queue next turn）。

3. 做 adversarial turn-detector test：让用户在句子中间长时间停顿。调 VAD silence threshold 和 turn-detector score threshold，让 false-cutoff 最低，同时不超过 900ms。

4. 通过 Twilio 在 PSTN 上部署同一 Agent。比较 PSTN first-audio-out 与 WebRTC。解释 jitter-buffer 和 codec 差异。

5. 为非英语语言（Japanese、Spanish）添加 voice activity detection。测量 Silero VAD v5 false-trigger rate 与 language-specific fine-tunes 的差异。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Turn detection | “End of utterance” | 给定 VAD silence 和 partial transcript，判断用户是否已经说完的 classifier |
| Barge-in | “Interruption handling” | VAD 检测到新用户语音时取消正在播放的 TTS |
| First-audio-out | “Latency” | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | “Speech gate” | 把 audio frames 分类为 speech 或 silence 的模型；Silero VAD v5 是 2026 默认选择 |
| Jitter buffer | “Audio smoothing” | 客户端短暂缓存 packets，用来吸收网络波动 |
| Filler | “Acknowledgment token” | 工具较慢时 Agent 发出的短确认语，用来避免沉默 |
| MOS | “Mean opinion score” | 感知语音质量评分；NISQA 是自动化 proxy |

## Further Reading / 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — reference WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — Python-first streaming agent 的替代框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — integrated speech models reference
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR reference
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD reference model
- [Cartesia Sonic-2](https://docs.cartesia.ai) — low-latency TTS reference
- [Retell AI architecture](https://docs.retellai.com) — production voice agent architecture
- [Vapi.ai production stack](https://docs.vapi.ai) — alternate production reference
