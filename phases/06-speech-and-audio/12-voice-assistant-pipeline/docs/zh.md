# Build a Voice Assistant Pipeline — The Phase 6 Capstone / 构建语音助手 Pipeline：Phase 6 Capstone

> 把 01–11 课的所有内容串起来。构建一个会听、会推理、会回应的 voice assistant。到 2026 年，这已经是解决了的工程问题，不再是研究问题；但集成细节决定它能不能上线。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 设计一个端到端 voice assistant pipeline，覆盖 capture、VAD、STT、LLM tools、TTS、playback 和 interruption
- 为每个组件设置 latency、quality、privacy 与 safety 目标
- 用 stub modules 理解 streaming STT → LLM → TTS 和 tool calling 的控制流
- 识别 first-word clipping、mid-response interrupt confusion、silence hallucination 和 PII logging 等上线风险

## The Problem / 问题

构建一个端到端 assistant：

1. 捕获麦克风输入（16 kHz mono）。
2. 检测用户语音开始/结束。
3. 流式转写。
4. 把 transcript 传给一个能 call tools 的 LLM（timer、weather、calendar）。
5. 把 LLM text 流式传给 TTS。
6. 把音频播放给用户。
7. 如果用户在回应中途打断，就停止。

Latency target：在 laptop CPU 上，用户说完之后 800 ms 内输出第一个 TTS audio byte。Quality target：不漏词，不在静音上 hallucinated subtitles，不泄漏 voice cloning，不让 prompt injection 成功。

## The Concept / 概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### The seven components / 七个组件

1. **Audio capture。** Mic → 16 kHz mono → 20 ms chunks。Python 中通常用 `sounddevice`，生产中用 native AudioUnit/ALSA/WASAPI。
2. **VAD (Lesson 11)。** Silero VAD @ threshold 0.5，min speech 250 ms，silence hang-over 500 ms。发出 “start” 和 “end” 信号。
3. **Streaming STT (Lesson 4-5)。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。输出 partial + final transcripts。
4. **LLM with tool calling。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。为 tools 定义 JSON schema。流式输出 tokens。
5. **Streaming TTS (Lesson 7)。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 LLM 输出 20 个 tokens 后启动 TTS。
6. **Playback。** Speaker out；低带宽网络下用 opus-encode。
7. **Interruption handler。** 如果 VAD 在 TTS playback 中触发，就停止 playback、取消 LLM、重新开始 STT。

### The three failure modes you will hit / 你一定会遇到的三种失败模式

1. **First-word clip / 第一个词被截掉。** VAD 稍晚启动。用户的 “hey” 丢了。Start threshold 用 0.3，而不是 0.5。
2. **Mid-response interrupt confusion / 回答中途打断混乱。** 用户打断后 LLM 还在生成；assistant 盖过用户说话。把 VAD 接到 cancel-LLM。
3. **Silence hallucination / 静音幻觉。** Whisper 在 silent warm-up frames 上输出 “Thanks for watching”。一定要 VAD-gate。

### 2026 production reference stacks / 2026 生产参考 stack

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026 行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，第 15 课 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 最快上线；customization 受限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## Build It / 动手构建

### Step 1: mic capture with chunking (pseudocode) / 第 1 步：带 chunking 的 mic capture（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2: VAD-gated turn capture / 第 2 步：VAD-gated turn capture

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3: streaming STT → LLM → TTS / 第 3 步：streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4: tool calling inside the LLM loop / 第 4 步：LLM loop 内 tool calling

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5: interruption handling / 第 5 步：interruption handling

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Use It / 应用它

查看 `code/main.py` 中的 runnable simulation。它会用 stub models 串起所有七个组件，让你即使没有硬件也能看清 pipeline shape。真实实现时，用这些替换 stubs：

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` or `openai-whisper`
- `openai` (`gpt-4o`) or `anthropic`
- `kokoro` or `cartesia`
- `sounddevice` for I/O

## Pitfalls / 常见坑

- **Logging PII forever / 永久记录 PII。** 在多数司法辖区，full-turn audio 都是 PII。保留 30 天，静态加密。
- **No barge-in / 没有 barge-in。** 用户会打断。你的 assistant 必须停止说话。
- **TTS that blocks / 阻塞式 TTS。** Synchronous TTS 会阻塞 event loop。使用 async 或独立线程。
- **No tool-call error handling / 没有 tool-call 错误处理。** Tools 会失败。LLM 必须拿到 error + retry once，然后优雅降级。
- **Overzealous hallucination filters / 过度激进的 hallucination filters。** 过滤过度，assistant 会反复说 “I can't help with that.” 过滤不足，它会什么都说。在 held-out set 上校准。
- **No wake-word option / 没有 wake-word 选项。** Always-listening 是隐私风险。加入 wake-word gate（Porcupine 或 openWakeWord）。

## Ship It / 交付它

保存为 `outputs/skill-voice-assistant-architect.md`。给定 budget + scale + language + compliance constraints，产出完整 stack spec。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会用 stub modules 模拟一个完整 turn，并打印 per-stage latency。
2. **Medium / 中等。** 用真实 Whisper 模型替换 STT stub，输入一段预录 `.wav`。测量 WER 和 end-to-end latency。
3. **Hard / 困难。** 添加 tool calling：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 通过 tools 路由，并验证当用户说 “set a 5 minute timer” 时正确函数被触发，spoken reply 会确认。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Turn | 用户 + assistant 的 round-trip | 一个 VAD-bounded user speech + 一个 LLM-TTS response。 |
| Barge-in | interruption | 用户在 assistant 说话时开口；assistant 停止。 |
| Wake word | "Hey assistant" | 短 keyword detector；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | turn ending | VAD + min-silence 判定用户已经说完。 |
| Pre-roll | pre-speech buffer | 在 VAD 触发前保留 200–400 ms 音频，避免 first-word clip。 |
| Tool call | function invocation | LLM 发出 JSON；runtime dispatch；result 回流到 loop。 |

## Further Reading / 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — managed voice-native path。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference（第 15 课）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling。
