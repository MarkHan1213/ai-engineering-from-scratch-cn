# Real-Time Audio Processing / 实时音频处理

> Batch pipelines 处理一个文件。Real-time pipelines 必须在下一个 20 ms 到来之前处理完当前 20 ms。每个 conversational AI、broadcast studio 和 telephony bot 都被这个 latency budget 决定生死。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 分解 real-time audio pipeline 的 latency budget，并理解 frame、chunk、window 的节奏约束
- 实现 ring buffer、energy VAD、streaming ASR skeleton 和 interruption handler
- 解释 VAD、jitter buffer、AEC、barge-in、WebRTC Opus transport 在语音系统中的作用
- 识别线程争用、重采样延迟、TTS priming 和 echo cancellation 等实时系统风险

## The Problem / 问题

你想做一个感觉“活着”的 voice assistant。人类对话中的话轮转换延迟大约是 230 ms（从静音到回应）。超过 500 ms 会显得机械；超过 1500 ms 会像坏了一样。2026 年，一个完整 **hear → understand → respond → speak** loop 的预算是：

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi（Kyutai, 2024）实现了 200 ms full-duplex。GPT-4o-realtime（2024）约为 320 ms。2022 年的 cascaded pipelines 上线时常常是 2500 ms。10× 改进来自三项技术：(1) streaming everywhere，(2) 使用 partial results 的 asynchronous pipelining，(3) interruptible generation。

## The Concept / 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window。** Real-time audio 以固定大小 block 流动。常见选择是 20 ms（16 kHz 下 320 samples）。下游所有部分都必须跟上这个 cadence。

**Ring buffer / 环形缓冲区。** 固定大小的 circular buffer。Producer thread 写入新 frames，consumer thread 读取。它避免 hot path 上的 allocations。Size ≈ maximum-latency × sample-rate；2 秒 16 kHz ring = 32,000 samples。

**VAD (Voice Activity Detection)。** 没人说话时 gate downstream work。Silero VAD 4.0（2024）在 CPU 上处理每个 30 ms frame 小于 1 ms。`webrtcvad` 是更老的替代方案。

**Streaming ASR。** 随音频到达输出 partial transcripts 的模型。Parakeet-CTC-0.6B streaming mode（NeMo, 2024）在 320 ms latency 下达到 2–5% WER。Whisper-Streaming（Macháček et al., 2023）把 Whisper 分块，做 ~2 s latency 的 near-streaming。

**Interruption / 打断。** 当 assistant 正在说话而用户开口时，你必须：(a) 检测 barge-in，(b) 停止 TTS，(c) 丢弃剩余 LLM output。全部要在 100 ms 内完成，否则用户会觉得 assistant 听不见。

**WebRTC Opus transport。** 20 ms frames，48 kHz，自适应 bitrate 8–128 kbps。浏览器和移动端标准。LiveKit、Daily.co、Pion 是 2026 年构建 voice apps 的 stack。

**Jitter buffer / 抖动缓冲。** 网络 packets 会乱序或迟到。Jitter buffer 做重排和平滑；太小会有 audible gaps，太大则增加 latency。典型值 60–80 ms。

### Common gotchas / 常见问题

- **Thread contention / 线程争用。** Python 的 GIL + 重模型可能饿死 audio thread。使用 C-callback audio library（sounddevice、PortAudio），并让 Python 离开 hot path。
- **Sample-rate conversion latency / 采样率转换延迟。** Pipeline 内重采样会增加 5–20 ms。要么 upfront resample，要么使用 zero-latency resampler（PolyPhase、`soxr_hq`）。
- **TTS priming / TTS 预热。** 即使 Kokoro 这样的快速 TTS，首个请求也有 100–200 ms warm-up。缓存模型，并在第一个真实 turn 前用 dummy run 预热。
- **Echo cancellation / 回声消除。** 没有 AEC，TTS output 会重新进入麦克风，并触发 ASR 识别 bot 自己的声音。WebRTC AEC3 是开源默认选择。

```figure
nyquist-aliasing
```

## Build It / 动手构建

### Step 1: ring buffer / 第 1 步：ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity 决定最大 buffering latency。16 kHz 下 32,000 samples = 2 s。

### Step 2: VAD gate / 第 2 步：VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3: streaming ASR / 第 3 步：streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4: interruption handler / 第 4 步：interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键是 async I/O 和 cancellable TTS streaming。对 audio track 调用 WebRTC peerconnection.stop() 是 canonical way。

## Use It / 应用它

2026 年的 stack：

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Pitfalls / 常见坑

- **Buffering 500 ms to be safe / 为了保险缓冲 500 ms。** Buffer 本身就是 latency floor。把它缩小。
- **Not pinning threads / 没有固定线程优先级。** Audio callback 跑在比 UI 更低优先级的线程上，负载下就会 glitch。
- **TTS chunks too small / TTS chunks 太小。** 小于 200 ms 的 chunks 会让 vocoder artifacts 可闻。320 ms chunks 是 sweet spot。
- **No jitter buffer / 没有 jitter buffer。** 真实网络有 jitter；没有 smoothing 就会 pops。
- **Single-shot error handling / 一次性错误处理。** Audio pipelines 必须 crash-proof。一个 exception 就会杀掉 session。

## Ship It / 交付它

保存为 `outputs/skill-realtime-designer.md`。设计一个 real-time audio pipeline，并给出每个 stage 的具体 latency budget。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会模拟 ring buffer + energy VAD，并打印 fake 10-second stream 的 stage latencies。
2. **Medium / 中等。** 使用 `sounddevice`，构建一个 passthrough loop，以 20 ms frames 处理麦克风，并打印每个 frame 的 VAD state。
3. **Hard / 困难。** 用 `aiortc` 构建 full duplex echo test：browser → WebRTC → Python → WebRTC → browser。用 1 kHz pulse 测量 glass-to-glass latency。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Ring buffer | circular queue | 固定大小、lock-free（或 SPSC-locked）的 audio frames FIFO。 |
| VAD | silence gate | 标记 speech vs non-speech 的模型或 heuristic。 |
| Streaming ASR | real-time STT | 音频到达时输出 partial text；bounded lookahead。 |
| Jitter buffer | network smoother | 重排 out-of-order packets 的 queue；典型 60–80 ms。 |
| AEC | echo cancellation | 减去 speaker-to-mic feedback path。 |
| Barge-in | user interrupt | 用户在 TTS 中途说话；系统必须取消 playback。 |
| Full duplex | 双向同时 | 用户和 bot 可以同时说话；Moshi 是 full duplex。 |

## Further Reading / 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产 audio agent orchestration。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源 echo cancellation。
