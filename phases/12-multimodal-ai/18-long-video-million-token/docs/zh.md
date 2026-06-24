# Long-Video Understanding at Million-Token Context / 百万 Token 上下文中的长视频理解

> 1 小时 4K 视频，24 FPS，patch 并 embedding 后会产生约 6000 万 tokens。2 小时 podcast 的 transcript 是 30,000 tokens。一整部 Blu-ray feature film，即使用激进 pooling 压缩，也有数十万 tokens。Google Gemini 1.5（2024 年 3 月）用 10-million-token context 打开这个时代，能在小时级视频中可靠做 needle-in-a-haystack recall。LWM（Liu et al., 2024 年 2 月）展示了 ring attention 的 scaling path。LongVILA 和 Video-XL 进一步扩大 ingestion。VideoAgent 用 agentic retrieval 替代 raw context。每条路线都在 compute、recall 和工程复杂度之间做不同取舍。本课并排读取它们。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites / 前置知识：** Phase 12 · 17 (video temporal tokens)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 计算不同 FPS 和 pooling 下 long-form video 的 total visual-token counts。
- 解释三条 scaling paths：brute context（Gemini 1.5）、ring attention（LWM）、token compression（LongVILA / Video-XL）。
- 比较 raw-context video VLM 与 agentic-retrieval video VLM（VideoAgent）的 accuracy 和 latency。
- 为 30 分钟视频设计 needle-in-a-haystack test，并测量特定分钟处的 recall。

## The Problem / 问题

Qwen2.5-VL 尺寸的单帧 patch 在 384 native resolution 下约 729 tokens。3x3 pooling 后是 81 tokens/frame。30 分钟 clip @ 1 FPS = 1800 frames = 145,800 tokens。2025 年 open VLM 能做，但很紧。2 FPS 就是 291,600 tokens，只有最大 contexts 才放得下。

2 小时电影 @ 1 FPS 是 583k tokens。超过大多数 2026 open models；需要 Gemini 2.5 Pro，或更激进 pooling。

三条 scaling paths 因此出现。

## The Concept / 概念

### Path 1: Brute context (Gemini 1.5, Claude Opus) / 路线 1：Brute context

把硬件堆上去。把 context scale 到百万 tokens，一次 forward 处理全部。

Gemini 1.5 Pro 以 1M tokens 发布；Gemini 1.5 Ultra 到 10M；Gemini 2.5 Pro 在 2026 年能可靠处理数小时视频。论文（arXiv:2403.05530）记录了在约 9.5M tokens 内 99.7% 的 needle-in-a-haystack recall。

工程上是自定义 attention implementation，带 memory hierarchy（local + global + sparse）和面向 long-context efficiency 的 MoE expert routing。细节未完全公开，不开源。

### Path 2: Ring attention (LWM, LongVILA) / 路线 2：Ring attention

Ring attention 把长序列分布到多个 devices，每个 device 持有一个 chunk。完整序列上的 attention 通过 ring pattern 完成：每个 device 把自己的 chunk 发给下一个，计算 partial attention 并聚合。

LWM（Liu et al., 2024）用这种方式训练了 1M-token context model。训练 compute 随 context 近似线性增长，而不是二次：attention 的二次代价在 ring devices 上被摊开。

LongVILA（arXiv:2408.10188）把这个模式适配到 VLM。1400-frame videos × 192 tokens/frame = 268k context，用 8-way parallelism 上的 ring attention 训练。

### Path 3: Token compression (Video-XL, LongVA) / 路线 3：Token compression

比 brute context 便宜：在 LLM 看到序列前大幅压缩。

Video-XL（arXiv:2409.14485）使用 visual summary token：每个 N 帧 clip 产生一个 “summary” token，attend 到 N 帧。推理时 LLM 只看到每个 clip 一个 summary token，context 大幅缩小。

LongVA 用 “long context transfer” 技术，把 LLM context 从 200k 扩到 2M。先在 long-context text 上训练，再通过 shared representation 迁移到 long-context video。

Token compression 用特定 timestamp recall 换 scalability。模型通常知道发生了什么，但会错过精确帧。

### Path 4: Agentic retrieval (VideoAgent) / 路线 4：Agentic retrieval

不要把完整视频喂给 LLM。把视频当数据库，让 LLM 查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取问题。
2. LLM 向 retrieval tool 请求相关 clips（“show me segments with a cat”）。
3. Tool 返回匹配 clip timestamps。
4. LLM 通过 VLM 读取这些 clips。
5. LLM 合成答案，或继续发 follow-up queries。

这是 LLM-as-agent pattern 应用于长视频。推理更便宜（只编码相关 clips），工程更难（retrieval quality 成为瓶颈）。

### Needle-in-a-haystack benchmarks / Needle-in-a-haystack 基准

标准 long-context test：在视频随机位置插入唯一 visual 或 textual marker，然后问一个必须回忆它的问题。

Metric：随视频长度和 marker position 变化的 Recall@k。

Gemini 2.5 Pro 在 90 分钟视频内 recall >99%。Open 72B models（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟约 85-90%，超过 60 分钟下降。

如果 retrieval tool 足够好，VideoAgent 在 2+ 小时视频上能追平或超过 raw-context models，因为 tool 能命中 needle。

### Which path to pick / 如何选择

15 分钟 clip、frontier accuracy：open 72B + native context 通常可行。选 Qwen2.5-VL-72B。

30 分钟到 1 小时内容：open 选 LongVILA 或 Video-XL；closed 选 Gemini 2.5 Pro。质量门槛很高时，frontier 会走 closed。

2+ 小时内容：VideoAgent 或类似 retrieval patterns。或者先 chunk-level summarize，再喂 hierarchical summaries。

### 2026 production pattern / 2026 生产模式

实践中，生产 long-video pipelines 多为 hybrid：

1. 对完整视频运行 dynamic-FPS sampling + aggressive pooling（得到 100k-token global representation）。
2. 送入 72B VLM 生成 global summary。
3. 如果用户问细节问题，用 summary 作为 index 运行 agentic retrieval。

这结合了 brute-context 的全局理解与 retrieval 的局部细节。

## Build It / 动手构建

本课构建 token budget calculator、needle-in-a-haystack simulator 和 agentic retrieval router。你会先算出长视频的 token 规模，再模拟 marker recall，最后让 router 决定是否需要检索局部 clips。

## Use It / 应用它

`code/main.py`：

- 计算 1 分钟到 3 小时视频在不同 FPS + pooling 下的 token budgets。
- 模拟 needle-in-a-haystack run：在随机 timestamp 注入 marker，提问并打 recall 分。
- 包含 agentic-retrieval router simulator，选择要喂给下游 VLM 的特定 clips。

运行 budget table，感受 scale gap。

## Ship It / 交付它

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定 video duration 和 query complexity，它会在 brute-context、compression 和 agentic retrieval 之间选择，并计算 latency + quality expectations。

## Exercises / 练习

1. 45 分钟 lecture @ 1 FPS、81 tokens/frame。总 tokens 是多少？能放进哪些模型 context？

2. 设计一个 needle-in-a-haystack test：你会在哪一分钟注入 marker？确切 query format 是什么？

3. 比较 1 小时视频上的 brute-context Qwen2.5-VL-72B（80k context）与 VideoAgent（Claude 3.5 + retrieval）。哪个 recall 更好？哪个 latency 更好？

4. Ring attention 的 memory cost 随 sequence length 和 device count 线性增长。解释原因，以及如果去掉 ring-rotation phase 会失败在哪里。

5. 阅读 Gemini 1.5 Section 5 关于 needle-in-a-haystack。论文在 1M vs 10M token boundary 上发现了什么 recall 现象？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Brute context | “Just more tokens” | 把 LLM context scale 到百万 tokens；一次处理全部内容 |
| Ring attention | “LWM-style parallel” | 分布式 attention pattern，每个 device 持有 chunk 并轮转 |
| Token compression | “Summary tokens” | 在 LLM 前通过 learned compressor 减少每个 clip 的 tokens |
| Needle-in-haystack | “NIH test” | 在随机位置插入唯一 marker，并在测试时要求模型回忆 |
| Agentic retrieval | “LLM as query planner” | LLM 请求 retrieval tool 返回相关 clips，再通过 VLM 读取并合成答案 |
| VideoAgent | “Retrieval pattern for video” | Canonical agentic-retrieval design：question -> tool -> clip -> answer |

## Further Reading / 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
