# MIO and Any-to-Any Streaming Multimodal Models / MIO 与 Any-to-Any 流式多模态模型

> GPT-4o 交付了多数 open models 还无法复制的产品形态：一个 agent 能听语音、看视频，并实时说话回应。到 2024 年末，open ecosystem 的答案是 MIO（Wang et al., 2024 年 9 月）。MIO tokenize text、image、speech 和 music，在 interleaved sequences 上训练一个 causal transformer，并支持任意 modality 到任意 modality 的生成。AnyGPT（Zhan et al., 2024 年 2 月）是 proof of concept；MIO 是 scale-up；Unified-IO 2（Allen AI, 2023 年 12 月）是带 vision + action grounding 的近亲。本课读取 any-to-any pattern：四个 tokenizer，一个 transformer，面向 streaming 的 decode。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites / 前置知识：** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 设计一个 shared vocabulary，让 text、image、speech、music tokens 不发生 collision。
- 从 compression + reconstruction trade-off 比较 SEED-Tokenizer（images）和 SpeechTokenizer residual-VQ（speech）。
- 解释构建 any-to-any generation 的四阶段 curriculum。
- 说出三个 open any-to-any recipes 及其主要 trade-offs：MIO、AnyGPT、Unified-IO 2。

## The Problem / 问题

统一多模态模型很容易宣称，真正 scale 起来很难。2024 年前的大多数 “any-to-any” 系统是 pipeline：vision model → text representation → speech model → audio。每一跳都会丢信息、增加 latency，并让训练复杂化。GPT-4o 的 demo video 展示了 subsecond response 的单模型替代方案；open systems 落后数月。

工程挑战包括：

- 每个 modality 都需要 tokenizer，压缩要足够近似无损以便重建，并且 token 速率要让 transformer 消费得动。
- 单一 vocabulary 必须为 text（32k+）、image（16k+）、speech（4k+）、music（8k+）分配空间。至少四万多个 entries。
- 训练数据必须覆盖每种 input-output pair（text→image、image→speech、speech→image 等），否则模型必须学会组合。
- 推理必须足够快地 stream output tokens，达到对话 latency（<500ms time-to-first-audio-byte）。

## The Concept / 概念

### Four tokenizers for four modalities / 四个模态的四个 tokenizer

MIO 的 tokenizer stack：

- Text：标准 BPE，vocab 约 32000。
- Image：SEED-Tokenizer（2023）—— quantized VAE with discrete codebook，4096 entries，每张图 32x32 tokens。
- Speech：SpeechTokenizer residual-VQ（2023）—— 把 16kHz waveform 编码到 8 个 hierarchical codebooks；第一层是 coarse content，后续层加入 prosody 和 speaker identity。
- Music：类似 residual-VQ（Meta MusicGen / Encodec family），4-8 个 codebooks。

每个 modality 都产出 integer tokens。Tokens 在 shared vocabulary 中使用互不重叠的 ID ranges：

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

总计约 48k vocabulary。Input embedding 与 output projection 覆盖全部。

### Streaming decode / 流式解码

Speech generation 使用 residual-VQ。Transformer 预测 base（layer 0）speech tokens；parallel-decoded residual quantizer 预测后续 layers。每个 layer 0 token 大约对应 16kHz 音频的 50ms。

Streaming pattern：

1. 用户对麦克风说话；real-time audio tokenizer 每 50ms 输出 speech tokens。
2. MIO 边到达边消费 tokens（prompt prefill + incremental forward）。
3. Output tokens 流式生成；parallel speech decoder 以约 50-150ms latency 把它们转成 audio samples。
4. Time-to-first-audio-byte：MIO 论文约 300-500ms，接近 GPT-4o 的约 250ms。

Mini-Omni（arXiv:2408.16725）、GLM-4-Voice（arXiv:2412.02612）和 Moshi（arXiv:2410.00037）是互补的 streaming speech-LLM 设计。Moshi 尤其快，单 GPU 上 160ms round-trip。

### Four-stage curriculum / 四阶段 curriculum

MIO 的训练 curriculum：

1. Stage 1 — alignment。大规模 modality-pair corpora：text-image、text-speech、text-music。每个 pair 使用自己的 token vocabulary segment。训练 shared vocabulary。
2. Stage 2 — interleaved。多模态 interleaved documents（带图片和视频的博客、带 transcript 的播客等）。训练 cross-modality context。
3. Stage 3 — speech-enhanced。额外音频数据，提升 speech quality 且不丢 text capability。
4. Stage 4 — SFT。跨模态 instruction tuning：VQA、captioning、narration、speech-to-speech dialogue。

缺一个 stage 会损伤特定能力：跳过 stage 2，模型会丢 cross-modality context；跳过 stage 3，speech 很差。

### Chain-of-visual-thought / 视觉思维链

MIO 引入 chain-of-visual-thought：模型输出 intermediate image tokens 作为 reasoning step。对 “is the cat climbing a tree?”，模型会：

1. 输出 `<image>` tokens，渲染场景（来自输入图或 sketch）。
2. 输出文本分析 sketch。
3. 输出 final answer。

这个中间图像像 scratchpad。Spatial-reasoning tasks 上 benchmark 会提升。思想类似 text reasoning 的 chain-of-thought。

### Competitors in any-to-any / Any-to-any 竞争路线

- AnyGPT（arXiv:2402.12226）：4 modalities（text、image、speech、music），设计相似。
- Unified-IO 2（arXiv:2312.17172）：加入 vision action outputs、depth、normals。任务更多，规模更小。
- NExT-GPT（arXiv:2309.05519）：LLM + modality-specific diffusion decoders。不是 single-model approach。
- CoDi（arXiv:2305.11846）：composable diffusion；通过 shared latent 做 any-to-any。

MIO 最接近 pure-token any-to-any。AnyGPT 是它的概念祖先。

### Latency budget / 延迟预算

对话产品中，每个组件都消耗 latency：

- Mic to audio tokens：约 50ms。
- Prefill（audio tokens + history）：8B 模型上约 100ms。
- First output token：约 50ms。
- Parallel residual-VQ + speech decoder：约 100-150ms。

总 time-to-first-audio-byte：最少约 300ms。GPT-4o 声称约 250ms。Moshi 声称 160ms。MIO/AnyGPT 在公开 benchmark 中通常是 400-600ms。

### Why any-to-any stays hard / 为什么 any-to-any 仍然难

即使到 2026 年，open any-to-any models 在两个轴上仍落后于 closed models：

- Speech quality。Residual-VQ tokenizer 有损；对话语音比 ElevenLabs-class voices 更机械。
- Cross-modality reasoning。让模型 “sing about what you see” 仍比纯视觉任务更容易失败。

这些仍是开放研究问题。Qwen3-Omni（Lesson 12.20）是 2025 年最先进的 open attempt。

## Build It / 动手构建

本课构建四模态 vocabulary allocator 和 streaming decode simulator。你会为不同 token range 分配 ID，模拟 speech token 到 audio 的流式路径，并用组件级 latency 估算 time-to-first-audio-byte。

## Use It / 应用它

`code/main.py`：

- 定义四模态 vocabulary allocation 并打印出来。
- 通过 tokenizer router 路由一组 multimodal inputs（text、image、audio-clip、music）。
- 为 text-to-speech response 模拟 streaming decode，并累计 latency。
- 给定 encoder、prefill 和 decoder latencies，计算 expected time-to-first-audio-byte。

## Ship It / 交付它

本课产出 `outputs/skill-any-to-any-pipeline-auditor.md`。给定 conversational product spec（modalities in、modalities out、latency target），它会审计 MIO-family design choices 并计算 latency budget。

## Exercises / 练习

1. 产品接受 speech input 并返回 speech output。端到端 latency budget target 应该是多少？列出消耗时间的组件。

2. SpeechTokenizer residual-VQ 使用 8 个 codebooks。解释为什么需要 parallel-decoding residual levels（相对于 sequential），以及它节省了什么 latency。

3. Vocabulary 有 32k text + 4k image + 4k speech。加入 8k music 和约 10 个 separators。hidden dim 4096 下 embedding matrix 的参数成本是多少？

4. Chain-of-visual-thought 会输出 intermediate image。哪些问题受益？哪些会被额外 tokens 伤害？

5. 阅读 Moshi（arXiv:2410.00037）。描述它的 “inner monologue” 技术，并与 MIO 的 chain-of-visual-thought 对比。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Any-to-any | “Multimodal in/out” | 单个模型接受并输出 text、image、speech、music 的任意方向 |
| Residual-VQ | “Speech tokenizer stack” | 多 codebook tokenization，每层增加信息；base layer 是 content，后续层是 prosody |
| SEED-Tokenizer | “Image codes” | MIO 使用的 4096-entry codebook 离散图像 tokenizer |
| Chain-of-visual-thought | “Visual scratchpad” | 模型在 final answer 前生成 intermediate image 作为 reasoning step |
| Time-to-first-audio-byte | “TTFAB” | 从用户语音到首个音频输出的 latency；<500ms 才有对话感 |
| Four-stage curriculum | “Training recipe” | Alignment -> interleaved -> speech-enhanced -> SFT，按此顺序 |

## Further Reading / 延伸阅读

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
