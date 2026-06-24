# Video-Language Models: Temporal Tokens and Grounding / 视频语言模型：时间 Token 与 Grounding

> 视频不是一叠照片。一个 5 秒 clip 有 causal ordering、action verbs 和 event timing，这些是图像模型无法表示的。Video-LLaMA（Zhang et al., 2023 年 6 月）发布了第一个带 audio-visual grounding 的 open video-LLM。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与 frontier proprietary models 的差距。每个系统以不同方式解决 temporal tokens：per-clip Q-former、per-frame concat-pool、per-token TMRoPE。本课读取这些模式，构建 uniform-vs-dynamic frame sampler，并在 temporal grounding tasks 上评估。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites / 前置知识：** Phase 12 · 08 (LLaVA-OneVision)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 解释 temporal positional encoding 为什么能独立于 vision encoder 改变 video VLM performance。
- 比较 uniform、dynamic-FPS 和 event-driven frame sampling 在 tokens-per-second 与 grounding accuracy 上的取舍。
- 描述 Q-former-per-clip（Video-LLaMA）、pooled-per-frame（Video-LLaVA）和 M-RoPE-per-token（Qwen2.5-VL）设计。
- 说出四个视频 benchmark：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## The Problem / 问题

1 分钟 30 FPS 视频有 1800 帧。每帧 196 visual tokens（ViT-B @ 224）时，总计 352k tokens，超过 2024 年大多数 LLM context。

三种 reduction strategies：

1. Subsample frames（按内容 1-8 FPS）。
2. 对每帧 patch tokens 做 aggressive pooling（3x3 或 4x4 bilinear pool）。
3. 用 Q-former 压缩，一个 16-frame clip 输出 64 tokens。

每种取舍不同。Subsampling 丢 temporal detail。Pooling 丢 spatial detail。Q-former 两者都丢一点，但省 tokens。

Temporal position encoding 是另一条轴：模型如何知道 frame 5 在 frame 6 之前？选择包括简单 1D temporal RoPE（Video-LLaMA）、learned temporal embeddings（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整 3D）。

## The Concept / 概念

### Video-LLaMA: Q-former per clip + audio branch / Video-LLaMA：每 clip 一个 Q-former + audio branch

Video-LLaMA（2023）是第一个 open video-LLM。架构：

- 16-frame clips @ 2 FPS（覆盖 8 秒）。
- Per-frame ViT features -> Video Q-former，对 16 帧做 cross-attention -> 32 learnable queries -> LLM。
- 并行 audio branch：waveform -> ImageBind audio encoder -> Audio Q-former -> 32 queries -> LLM。

优势：audio-visual joint reasoning。弱点：固定 clip length，不支持任意 time grounding。

### VideoChat and Video-LLaVA / VideoChat 与 Video-LLaVA

VideoChat 保留 Video-LLaMA 思路，但去掉 audio 并简化。Video-LLaVA（Lin et al., 2023）在 image 和 video frames 上训练单个 visual encoder（“alignment before projection”），得到统一 representation。两者都是 frozen-CLIP-encoder + MLP + LLM。

两者都不处理长视频。它们是 8-16 frame systems。

### Qwen2.5-VL and TMRoPE / Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入 TMRoPE（Temporal-Modality Rotary Position Embedding）。每个 patch token 携带 `(t, h, w)` position，其中 t 是实际 timestamp，而不是 frame index。

相比简单 temporal embedding，关键差异是：

- Absolute time，不是 index。模型看到的是 “at 4.2 seconds”，不是 “at frame 15”。
- Per-token rotation，不是 per-clip。每个 visual token 按自己的 timestamp 独立旋转。
- Compatible with dynamic FPS。如果这里 2 FPS、那里 4 FPS，TMRoPE 原生处理不均匀间隔。

TMRoPE 使 “at what second does the cat jump?” 这类 query 可行。模型可以输出 “at 4.2 seconds”。Video-LLaMA 只能说 “early in the clip”。

### Frame sampling strategies / 帧采样策略

Uniform：在时长内均匀采样 N 帧。简单，但会错过 motion peaks。

Dynamic FPS：根据 motion intensity 自适应采样。Optical flow 或 frame differencing 会为 high-motion segments 选择更密采样。Qwen2.5-VL 训练了这种方式。

Event-driven：运行轻量 detector，在 action 发生处多采样。VideoAgent 使用。

Keyframe + context：在 shot boundaries 采样，再加相邻帧。适合 cinematic content。

### Pooling per frame / 每帧 pooling

1 FPS、每帧 576 tokens 时，5 分钟 clip 是 172,800 tokens。Qwen2.5-VL-72B 的 128k context 勉强可做，但昂贵。

3x3 bilinear pool 降到 64 tokens/frame，5 分钟是 19,200 tokens。这是多数任务的最佳折中点。

Agent workflows 中空间细节不那么重要，可以更激进 pooling（6x6 -> 16 tokens/frame）。

### The four video benchmarks / 四个视频 benchmark

- VideoMME：综合视频理解，覆盖 short + medium + long。
- TempCompass：细粒度 temporal reasoning，强调 “before” / “after”。
- EgoSchema：长程第一人称视频。
- Video-MMMU：多学科多模态视频问题。

完整 video-VLM evaluation 应覆盖四者。它们强调不同轴：TempCompass 几乎全是 ordering；EgoSchema 是 3+ 分钟 reasoning；VideoMME 覆盖多种时长。

### Grounding output formats / Grounding 输出格式

Temporal grounding 的输出格式：

- Free text：“The cat jumps around the 4-second mark.” 易读但不精确。
- Structured JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这种格式。
- Token-based：答案中 interleaved 特殊 `<time>4.1</time>` tokens。Qwen2.5-VL 的内部格式。

Token-based 对下游最准确。Qwen2.5-VL 的 JSON output format 可以直接解析。

### 2026 best practice / 2026 最佳实践

2026 年 video VLM：

- Encoder：SigLIP 2 + M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- Frame sampling：dynamic FPS（按 motion 1-4）并设置 max-frame cap。
- Per-frame pooling：3x3 bilinear。
- Output：带 time + event fields 的 structured JSON。
- Benchmarks：general 用 VideoMME + TempCompass；long-horizon 用 EgoSchema。

## Build It / 动手构建

本课构建 frame sampler 与 temporal-grounding evaluator：先比较 uniform 与 dynamic-FPS 的 token 数，再用事件 timestamp 的容忍窗口评估输出是否命中。你会把采样策略、pooling 和 structured output 连接起来。

## Use It / 应用它

`code/main.py` 包含：

- Uniform 与 dynamic-FPS frame samplers。
- Toy temporal-grounding evaluator：给定 “ground truth” event time T 和 model output，在容忍范围内打分。
- 对比 Video-LLaMA（16 frames, Q-former）、Video-LLaVA（8 frames, MLP）、Qwen2.5-VL（dynamic FPS + TMRoPE）。

## Ship It / 交付它

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定 video task（monitoring、action recognition、temporal grounding、summarization），它会选择 frame sampler、pooling factor、output format 和 expected accuracy tier。

## Exercises / 练习

1. 对 3 分钟 cooking demo，选择 uniform 或 dynamic FPS。用 token count 说明理由。

2. TMRoPE 相比简单 temporal embedding table 具体增加了什么能力？

3. 写一个 VLM 可学习输出的 temporal grounding JSON schema，包含 error cases。

4. 阅读 Video-LLaVA Section 3 的 “Alignment Before Projection”。为什么它优于训练分离的 image 和 video encoders？

5. 根据 2026 年 VideoMME leaderboard，top open model 与 top proprietary model 差距是多少？其中多少可归因于 temporal encoding，多少可归因于 base LLM scale？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Temporal grounding | “Time-localized answers” | VLM 输出事件发生的具体 timestamp range |
| TMRoPE | “Time-Multimodal RoPE” | 使用 absolute timestamps 的 3D rotary position，Qwen2.5-VL 使用 |
| Dynamic FPS | “Motion-aware sampling” | 高 motion segment 多采样，静态 segment 少采样 |
| Frame pooling | “Spatial compress per frame” | 在进入 LLM 前用 bilinear interpolation 减少每帧 patches |
| Video Q-former | “Clip compressor” | 把 N 帧映射到 K 个 learnable queries 的 cross-attention bottleneck |
| VideoMME | “Video bench” | 综合 short/medium/long video benchmark，2500+ samples |

## Further Reading / 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
