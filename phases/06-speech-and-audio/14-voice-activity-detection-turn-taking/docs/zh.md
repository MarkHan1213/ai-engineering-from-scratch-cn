# Voice Activity Detection & Turn-Taking — Silero, Cobra, and the Flush Trick / 语音活动检测与轮次判断：Silero、Cobra 与 Flush Trick

> 每个 voice agent 都由两个决策决定成败：用户现在是否在说话，以及用户是否说完了。VAD 回答第一个。Turn-detection（VAD + silence-hangover + semantic endpoint model）回答第二个。任一做错，assistant 要么打断用户，要么永远不闭嘴。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分 per-frame VAD、onset detection 和 end-pointing / turn-end detection
- 理解 energy gate、Silero VAD、semantic turn detector 的三级 cascade
- 实现 energy VAD、Silero VAD 调用、turn-end state machine 和 flush trick skeleton
- 为 voice agent 调整 threshold、minimum speech、silence hangover、pre-roll 和 semantic endpointing

## The Problem / 问题

Voice agent 在每个 20 ms chunk 上都要做三种不同决策：

1. **Is this frame speech? / 这一帧是不是语音？** —— VAD。二分类，per-frame。
2. **Has the user started a new utterance? / 用户是否开始了新的 utterance？** —— onset detection。
3. **Has the user finished? / 用户是否说完？** —— end-pointing（turn-end）。

朴素答案（energy threshold）在任何噪声下都会失败：交通声、键盘声、人群噪声。2026 年答案是：Silero VAD（开放、deep-learned）+ turn-detection model（semantic endpointing）+ 基于 VAD 校准的 silence hangover。

## The Concept / 概念

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### The three-tier VAD cascade / 三级 VAD cascade

**Tier 1: energy gate / 能量门。** 最便宜。对 RMS 做 -40 dBFS 阈值判断。可以过滤明显静音，但阈值以上的任何噪声都会触发。

**Tier 2: Silero VAD**（2020–2026, MIT）。1M parameters。在 6000+ languages 上训练。单 CPU 线程上每个 30 ms chunk 约 1 ms。5% FPR 下 TPR 87.7%。开源默认选择。

**Tier 3: semantic turn detector / 语义话轮结束检测器。** LiveKit 的 turn-detection model（2024–2026）或你自己的小分类器。区分“句中暂停”和“说完了”。它使用 linguistic context（intonation + recent words），而不只是 silence。

### Key parameters and their defaults / 关键参数及默认值

- **Threshold / 阈值。** Silero 输出概率；在 &gt; 0.5（默认）或 &gt; 0.3（敏感）时分类为 speech。更低 threshold = 更少 first-word clips，更多 false positives。
- **Minimum speech duration / 最短语音时长。** 拒绝短于 250 ms 的 speech，通常是 coughs 或 chair noise。
- **Silence hangover (end-pointing) / 静音挂起。** VAD 回到 0 后，等待 500–800 ms 再声明 end-of-turn。太短会打断用户；太长会显得迟钝。
- **Pre-roll buffer / 预滚缓冲。** 在 VAD 触发前保留 300–500 ms 音频。避免 “hey” 被截掉。

### The flush trick (Kyutai 2025) / Flush trick（Kyutai 2025）

Streaming STT models 有 look-ahead delay（Kyutai STT-1B 是 500 ms，STT-2.6B 是 2.5 s）。通常你需要在 end-of-speech 后等这么久才能拿到 transcript。Flush trick：当 VAD 触发 end-of-speech，**向 STT 发送 flush signal**，强制立即输出。STT 以约 4× realtime 处理，所以 500 ms buffer 约 125 ms 就能完成。

端到端：125 ms VAD + flush STT = conversational latency。

### 2026 VAD comparison / 2026 VAD 对比

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确默认选择。Cobra 是 compliance / accuracy upgrade。Energy-only VAD 不应出现在 2026 年生产系统里。

## Build It / 动手构建

### Step 1: the energy gate / 第 1 步：energy gate

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2: Silero VAD in Python / 第 2 步：Python 中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3: turn-end state machine / 第 3 步：turn-end state machine

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4: the flush trick skeleton / 第 4 步：flush trick skeleton

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

要让它工作，STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush。Whisper streaming 不支持，它是 block-based，总是要等 chunks。

## Use It / 应用它

| Situation | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (phone) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Need turn-ending quality | Silero + LiveKit turn-detector layered |

经验法则：除非真的没有其他选择，否则不要上线 energy-only VAD。

## Pitfalls / 常见坑

- **Fixed threshold / 固定阈值。** 安静环境可用，噪声环境失败。要么在设备上校准，要么切到 Silero。
- **Too-short silence hangover / silence hangover 太短。** Agent 会在句中打断。500–800 ms 是 conversational speech 的 sweet spot。
- **Too-long hangover / hangover 太长。** 感觉迟钝。对目标用户做 A/B test。
- **No pre-roll buffer / 没有 pre-roll buffer。** 用户音频前 200–300 ms 丢失。永远保留 rolling pre-roll。
- **Ignoring semantic endpointing / 忽略语义 endpointing。** “Hmm, let me think...” 包含长停顿。用户很讨厌被打断思路。使用 LiveKit 的 turn-detector 或类似模型。

## Ship It / 交付它

保存为 `outputs/skill-vad-tuner.md`。为 workload 选择 VAD model、threshold、hangover、pre-roll 和 turn-detection strategy。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会模拟 speech + silence + speech + coughs 序列，并测试三层 VAD。
2. **Medium / 中等。** 安装 `silero-vad`，处理一段 5 分钟录音，调整 threshold，让 first-word clips 和 false triggers 同时最小化。报告 precision/recall。
3. **Hard / 困难。** 构建一个 mini turn-detector：Silero VAD + 最近 10 个 words embeddings 上的 3-layer MLP（用 sentence-transformers）。在手工标注的 turn-end dataset 上训练。F1 比 Silero-only 高 10%。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| VAD | Voice detector | 二分类 per-frame：这是不是 speech？ |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint。 |
| Silence hangover | Wait-after-speech | 声明 turn end 前等待的时间；500–800 ms。 |
| Pre-roll | Pre-speech buffer | 在 VAD 触发前保留 300–500 ms audio。 |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms，而不是 500 ms delay。 |
| Semantic endpoint | “他们是真的停了吗？” | 看 words 而不是只看 silence 的 ML classifier。 |
| TPR @ FPR 5% | ROC point | 标准 VAD benchmark；Silero 87.7%，WebRTC 50%。 |

## Further Reading / 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考级 open VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — sub-200 ms engineering trick。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的 semantic endpointing。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — legacy baseline。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization-grade segmentation。
