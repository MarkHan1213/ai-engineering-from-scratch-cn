# Streaming Speech-to-Speech — Moshi, Hibiki, and Full-Duplex Dialogue / 流式语音到语音：Moshi、Hibiki 与全双工对话

> 2024–2026 年重塑了 voice AI。Moshi 用一个模型同时听和说，延迟 200 ms。Hibiki 逐 chunk 做 speech-to-speech translation。两者都放弃 ASR → LLM → TTS pipeline，改用 Mimi codec tokens 上的统一 full-duplex architecture。这是新的参考设计。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么 cascaded voice pipeline 存在 300–500 ms 的结构性 latency floor
- 理解 Moshi 的双 Mimi stream、Temporal Transformer、inner monologue text stream 和 Depth Transformer
- 描述 Hibiki 的 streaming speech-to-speech translation 思路及 Hibiki-Zero 的数据要求变化
- 判断什么时候选择 full-duplex architecture，什么时候仍应使用传统 pipeline

## The Problem / 问题

用第 11 + 12 课搭建的每个 voice agent，都有一个约 300–500 ms 的基础延迟下限：VAD 触发，STT 处理，LLM 推理，TTS 生成。每个阶段都有自己的最小延迟。你可以调优和并行化，但 pipeline shape 本身会封顶。

Moshi（Kyutai, 2024–2026）问了另一个问题：如果没有 pipeline 呢？如果一个模型直接吃 audio in、发 audio out，连续运行，把 text 作为中间 “inner monologue” 而不是必经阶段呢？

答案是 **full-duplex speech-to-speech**。理论延迟 160 ms（80 ms Mimi frame + 80 ms acoustic delay）。实际在单张 L4 GPU 上约 200 ms。这是最佳 pipelined voice agent 延迟的一半。

## The Concept / 概念

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### The Moshi architecture / Moshi 架构

**Inputs / 输入。** 两条 Mimi codec streams，均为 12.5 Hz × 8 codebooks：

- Stream 1：user audio（Mimi-encoded，持续到达）
- Stream 2：Moshi 自己的 audio（由 Moshi 生成）

**The transformer / Transformer。** 一个 7B-parameter Temporal Transformer 处理两条 streams 和一条 text “inner monologue” stream。在每个 80 ms step，它：

1. 消费最新 user Mimi tokens（8 codebooks）。
2. 消费最近的 Moshi Mimi tokens（8 codebooks，作为已生成内容）。
3. 生成下一个 Moshi text token（inner monologue）。
4. 通过一个小 Depth Transformer 生成下一个 Moshi Mimi tokens（8 codebooks）。

三条 streams：user audio、Moshi audio、Moshi text 并行运行。Moshi 可以边说边听用户；用户打断时可以中止自身输出；也可以插入 backchannel（如 “mhm”）而不打断主发话。

**The depth transformer / Depth Transformer。** 在一个 frame 内，8 个 codebooks 不是并行预测的，它们之间存在 inter-codebook dependencies。一个小型 2-layer “depth transformer” 会在 80 ms 内顺序预测它们。这是 AR codec LMs 的标准 factorization（VALL-E、VibeVoice 也使用）。

### Why inner-monologue text helps / 为什么 inner-monologue text 有帮助

没有显式文本时，模型必须在 acoustic stream 中隐式建模语言。Moshi 的洞察是：强制它在输出音频的同时输出 text tokens。Text stream 本质上是 Moshi 正在说的话的 transcript。它提高 semantic coherence，让替换 language model head 更容易，并免费给你 transcripts。

### Hibiki: streaming speech-to-speech translation / Hibiki：流式 speech-to-speech translation

同样架构，训练在 translation pairs 上。Source audio in，target-language audio out，连续执行。Hibiki-Zero（2026 年 2 月）不再需要 word-level aligned training data，而是用 sentence-level data + GRPO reinforcement learning 做 latency optimization。

初始支持四种 language pairs；适配新语言约需 1000 小时数据。

### The broader Kyutai stack (2026) / 更完整的 Kyutai stack（2026）

- **Moshi** —— full-duplex dialogue（French first，English well-supported）
- **Hibiki / Hibiki-Zero** —— simultaneous speech translation
- **Kyutai STT** —— streaming ASR（500 ms 或 2.5 s look-ahead）
- **Kyutai Pocket TTS** —— 100M-param TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** —— 在 public servers 上组合这些能力的完整 pipeline

L40S GPU 上吞吐：64 concurrent sessions at 3× real-time。

### Sesame CSM — the cousin / Sesame CSM：近亲

Sesame CSM（2025）使用类似想法：Llama-3 backbone + Mimi codec head。但 CSM 是单向的（接收 context + text，生成 speech），而不是 full-duplex。它是市场上最好的 “voice presence” TTS；但不等同于 Moshi 的 full-duplex capability。

### 2026 performance numbers / 2026 性能数字

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

## Build It / 动手构建

### Step 1: the interface / 第 1 步：interface

Moshi 暴露一个 WebSocket server，接收 80 ms chunks 的 Mimi-encoded audio，并返回 80 ms chunks 的 Mimi-encoded audio。双向。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Step 2: the full-duplex loop / 第 2 步：full-duplex loop

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准 transport。

### Step 3: the training objective (conceptual) / 第 3 步：training objective（概念）

对每个 80 ms frame `t`：

- Input：`user_mimi[0..t]`，`moshi_mimi[0..t-1]`，`moshi_text[0..t-1]`
- Predict：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

Text 先于 audio 预测（inner monologue）；audio 在 depth transformer 内按 codebook 顺序预测。

### Step 4: where Moshi wins and where it doesn't / 第 4 步：Moshi 赢在哪里，不赢在哪里

Moshi 赢在：

- 便宜硬件上 sub-250 ms end-to-end。
- 自然 back-channels 和 interruptions。
- 不需要 pipeline glue code。

Moshi 不赢在：

- Tool calling（没有为此训练；需要单独 LLM path）。
- Long reasoning（Moshi 是 8B-ish dialogue model，不是 Claude/GPT-4）。
- 小众主题事实准确性。
- 大多数生产企业用例（2026 年仍使用 pipelines）。

## Use It / 应用它

| Situation | Pick |
|-----------|------|
| Lowest-latency voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent with tools | Pipeline (Lesson 12), not Moshi |
| Custom-voice TTS in context | Sesame CSM |
| Speech-to-speech, any languages | GPT-4o Realtime or Gemini 2.5 Live (commercial) |

## Pitfalls / 常见坑

- **Limited tool calling / tool calling 受限。** Moshi 是 dialogue model，不是 agent framework。需要和 pipeline 组合处理 tools。
- **Specific-voice conditioning / 指定声音条件化。** Moshi 使用单一训练 persona；cloning 是单独训练过程。
- **Language coverage / 语言覆盖。** French + English 很强；其他有限。Hibiki-Zero 有帮助，但仍需要 training data。
- **Resource cost / 资源成本。** 一个完整 Moshi session 占用一个 GPU slot；不是便宜的 shared-tenant deploy pattern。

## Ship It / 交付它

保存为 `outputs/skill-duplex-pipeline.md`。为 voice-agent workload 选择 pipeline 或 full-duplex architecture，并说明理由。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会符号化模拟 two-stream + inner-monologue architecture。
2. **Medium / 中等。** 从 HuggingFace 拉取 Moshi，运行 server，测试一次对话。测量从 end-of-user-speech 到 start-of-Moshi-response 的 wall-clock latency。
3. **Hard / 困难。** 把你的第 12 课 pipeline agent 和 Moshi 在 20 条匹配 test utterances 上比较 P50 latency。写清楚 pipeline 在什么情况下仍然 architecturally wins。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Full-duplex | hear-and-speak at once | 同一个模型上两条 audio streams 同时活跃。 |
| Inner monologue | 模型的 text stream | Moshi 在输出音频的同时发出 text tokens。 |
| Depth transformer | inter-codebook predictor | 在一个 80 ms frame 内预测 8 个 codebooks 的小 transformer。 |
| Mimi | Kyutai 的 codec | 12.5 Hz × 8 codebooks；semantic+acoustic；支撑 Moshi。 |
| Streaming S2S | audio → audio live | 逐 chunk 翻译/对话，没有 pipeline stages。 |
| Back-channeling | “Mhm” reactions | Moshi 可以发出小的应答声，不破坏自己的 turn。 |

## Further Reading / 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 不需要 aligned data 的 streaming translation。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM spec。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — install + server。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业 peer。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS framework。
