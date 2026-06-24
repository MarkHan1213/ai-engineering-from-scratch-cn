# LLaVA-OneVision: Single-Image, Multi-Image, Video in One Model / LLaVA-OneVision：一个模型统一单图、多图与视频

> LLaVA-OneVision（Li et al., 2024 年 8 月）之前，open-VLM 世界有几条分离 lineage：单图的 LLaVA-1.5，多图的 Mantis 和 VILA，视频的 Video-LLaVA 与 Video-LLaMA。它们各赢自己的 benchmark，也各自在其他场景失败。LLaVA-OneVision 认为，一个 curriculum 可以训练单个模型压过三类 specialist，而且 emergent task-transfer effects（单图技能迁移到视频，多图推理迁移到单图）会超过 specialist 之和。Recipe 很简单：跨场景保持视觉 token budget 近似恒定，再显式地从 single-image 走到 OneVision（multi-image）再到 video。本课读取 budget、curriculum 和 emergent behaviors。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites / 前置知识：** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 设计一个 visual-token budget，使 single-image、multi-image 和 video 输入都能保持恒定总量。
- 排列一个 training curriculum，把 single-image skills 迁移到 video，同时避免 catastrophic forgetting。
- 解释为什么在 curriculum 正确时，单个模型会在相同参数量下击败 specialists。
- 说出 LLaVA-OneVision 报告的三个 emergent capabilities：multi-camera reasoning、set-of-mark prompting、iPhone-screenshot agent。

## The Problem / 问题

Image、multi-image 和 video 对模型施加不同压力。

Single-image 需要高 resolution tokens（AnyRes，约 2880 visual tokens）来捕捉 OCR 和细节。每个样本预算：1 张图，2880 tokens。

Multi-image 需要几张中等 resolution 的图（每张约 576 tokens），让跨图推理能放进 context。每个样本预算：4-8 张图，每张 576，总计 2300-4600 tokens。

Video 需要许多低 resolution frame（pooling 后约 196 tokens/frame）捕捉时间动态。每个样本预算：8-32 frames，每帧 196，总计 1600-6200 tokens。

如果训练 separate models，你只选一个 budget。如果训练一个模型，budget 必须能在各场景间合理 scale，且不能炸掉 context。

OneVision 之前的默认答案是“训练一个场景，忽略其他”。Video-LLaVA 把 video retrofit 到 image model 上，需要额外 training stages。LLaVA-NeXT 通过 tiling 加了 multi-image support。没有一个能干净覆盖三者。

## The Concept / 概念

### The OneVision token budget / OneVision token 预算

LLaVA-OneVision 选择每个样本约 3000-4000 visual tokens 的统一预算，并按场景不同分配：

- Single image：AnyRes-9（3x3 tiles + thumbnail），每个 tile @ 384 有 729 patches，激进 2x2 bilinear pooling → 每 tile 182。总计：9 * 182 + 182 = 1820 tokens。或者 AnyRes-4，每 tile 729 → 2916 + 729。
- Multi-image：每张图中等 resolution（384，无 tiling），729 tokens，不 pooling。6 张图 → 4374 tokens。
- Video：32 frames @ 384 resolution，激进 3x3 bilinear pool → 81 tokens/frame。总计：32 * 81 = 2592 tokens。

分配方式保持总 token 数大致恒定。LLM 不会看到炸掉 context 的 batch。Encoder 在不同场景产生不同几何形态，但 LLM 消费相同预算。

### The three-stage curriculum / 三阶段 curriculum

LLaVA-OneVision 分三阶段训练：

1. Single-image SFT（stage SI）。所有数据都是 single-image-plus-text。训练高分辨率 AnyRes 输入。它教会 perception、OCR 和细粒度理解。使用 LLaVA-NeXT data 加 OneVision-specific single-image data。
2. OneVision SFT（stage OV）。混合 single-image + multi-image + video（uniformly sampled frames）。在 unified token budget 上训练。它教模型处理 heterogeneous batch shapes。没有 weight reset，从 stage SI 继续。
3. Task transfer（stage TT）。继续使用目标 task mix，通常根据产品更偏 multi-image 或 video。可选 deployment fine-tune。

关键是 curriculum order。Video-first 或 multi-image-first 会比 single-image-first 带来更差 image performance，即使数据相同。论文明确做了 ablation。

### Why curriculum works / 为什么 curriculum 有效

Single-image training 建立 perceptual base。Patch tokens 携带细粒度视觉特征，LLM 学会把它们与文本整合。Multi-image 和 video 引入结构挑战（哪张图是哪张、什么先发生），如果没有强 perception base，就很难学好。

如果从头把所有场景混在一起训练，模型会 underfit perception（每个 batch 的 single-image 数据有限）并 overfit structure（multi-image / video 数据太多）。结果是：会遵循跨图推理套路，但视觉很浅。

Curriculum ordering 先给 stage SI 的 perception strength，再通过 stage OV 加入 compositional/temporal reasoning，且不丢前者。

### Emergent cross-scenario skills / 跨场景涌现技能

LLaVA-OneVision 论文报告了三个 emergent capabilities：

1. Multi-camera reasoning。模型分别在 multi-image + video 上训练；推理时要求它理解 multi-camera driving scene。尽管训练中没有完全相同格式，模型仍能整合视角。
2. Set-of-mark prompting。用户用数字标记图像中的物体，模型回答“mark 3 相对 mark 7 在做什么”。训练中没有 marks 或 annotation；能力来自 spatial grounding + multi-image reference 的组合。
3. iPhone-screenshot agent。用户给 iPhone 屏幕截图，要求规划下一次点击。训练包含 UI screenshots、用户 workflow 视频和 before/after 多图对，泛化到 agent use case。

这些不是显式训练任务，而是 curriculum compositional structure 的产物。

### Visual-token pooling / 视觉 token pooling

Token budget 需要 pooling。OneVision 在 2D patch grid 上做 bilinear interpolation：24x24 = 576 patches 变成 12x12 = 144（2x factor）或 8x8 = 64（3x factor）。Pooling 在 patch-grid space 中完成，而不是 token space，以保留 locality。

每个场景选什么 pooling factor 也是 hyperparameter。少 pooling = 更多 tokens = 更丰富 representation。多 pooling = 更少 tokens = 能容纳更多 frames / images。

### LLaVA-OneVision-1.5 / LLaVA-OneVision-1.5

2025 年 follow-up（LLaVA-OneVision-1.5, arXiv 2509.23661）是 training data、model weights 和 code 都 “fully open” 的版本。在某些 benchmark 上缩小了 proprietary gap，并把 recipe 民主化。Curriculum 相同，数据更多，base LLM 更好。没有架构变化。

### Contrast with Qwen2.5-VL / 与 Qwen2.5-VL 对比

Qwen2.5-VL（Lesson 12.09）选择不同。它使用 M-RoPE 和 dynamic FPS，而不是固定 pooling。它的 budget 随输入 scale：1 分钟视频比 5 秒视频用更多 tokens。LLaVA-OneVision 固定 budget，通过 pooling scale。两者都有效；前者更可配置，后者更可预测。

## Build It / 动手构建

本课构建一个 OneVision-style budget solver：给定每样本 token budget 和场景占比，自动分配 single-image 的 AnyRes factor、多图的 per-image resolution、视频的 frame count 与 pooling factor，并输出一个 stage-by-stage curriculum。

## Use It / 应用它

`code/main.py` 是 OneVision-style VLM 的 curriculum 和 budget planner。给定每样本 token budget 与目标 scenario mix（例如 40% single-image、30% multi-image、30% video），它会：

- 为每个场景分配 resolution、pooling factor 和 frames。
- 检查每个场景是否落在 shared budget 内。
- 报告 expected token count、LLM FLOPs，以及哪些场景 under-tokenized。
- 打印逐阶段 training schedule。

用它规划 OneVision fine-tune，或 sanity-check VLM deployment 的 per-request cost。

## Ship It / 交付它

本课产出 `outputs/skill-onevision-budget-planner.md`。给定 target task distribution 和 per-sample budget，它会输出 AnyRes factor、per-frame pooling、video frame count 和 curriculum stage weights。训练或微调 unified-scenario VLM 时都可以使用。

## Exercises / 练习

1. 产品支持 80% single-image、10% multi-image（2-4 images）、10% video（8-16 frames）。设计 token budget。由于 heavy multi-image 少，省下来的额外预算应该放在哪里？

2. 阅读 LLaVA-OneVision Section 4.3（emergent capabilities）。提出一个论文没有报告、但该 curriculum 可能解锁的第四种 emergent skill。

3. 交换 curriculum order：先 multi-image，再 single-image，再 video。预测哪些 benchmarks 会下降，为什么？

4. 论文报告视频 benchmark 只用每样本 8 帧训练。这能泛化到推理时 30 秒视频吗？最先出问题的是 token budget 还是 temporal reasoning？

5. 把 24x24 patches bilinear pooling 到 12x12 是每个维度 4x reduction。用 stdlib Python 实现 pooling，并验证每个 2x2 block 的 mean 与 bilinear output 匹配。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| OneVision scenario | “Single-image, multi-image, or video” | 统一 VLM 处理的三种 input shape；预算在三者间保持恒定 |
| Token budget | “How many tokens per sample” | 训练/推理样本中 LLM 看到的 visual tokens 总数，通常 3000-4000 |
| Curriculum | “Training order” | 为 emergent transfer 选择的 stage order（single-image → multi-image → video） |
| Bilinear pooling | “Token shrink” | 对 patch grid（2D）做 bilinear interpolation，减少 token count 同时保留 locality |
| Emergent skill | “Not trained, still works” | 推理时出现、但训练数据中没有匹配任务的能力，来自 curriculum composition |
| AnyRes-k | “k-tile setup” | k 个固定 resolution sub-tiles 加一个 thumbnail，常见 k ∈ {4, 9} |
| Task transfer | “Cross-scenario generalization” | single-image 学到的技能通过共享 backbone 迁移到 video（反之亦然） |

## Further Reading / 延伸阅读

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
