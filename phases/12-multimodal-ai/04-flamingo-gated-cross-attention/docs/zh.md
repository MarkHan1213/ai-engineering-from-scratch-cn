# Flamingo and Gated Cross-Attention for Few-Shot VLMs / Flamingo 与用于 Few-Shot VLM 的 Gated Cross-Attention

> DeepMind 的 Flamingo（2022）先于所有人做成两件事。它证明单个模型可以处理任意 interleaved 的 images、videos 和 text 序列；也证明 VLM 可以 in-context learning：给一个包含三个示例 `(image, caption)` pair 的 few-shot prompt，模型无需梯度更新就能为新图生成 caption。机制是 gated cross-attention layers：把它们插入 frozen LLM 的已有层之间，并用从 0 开始的可学习 tanh gate，确保初始化时保留 LLM 的文本能力。本课讲 Flamingo 的 Perceiver resampler 与 gated cross-attention 架构，它们是 Gemini interleaved inputs 和 Idefics2 visual tokens 的祖先。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites / 前置知识：** Phase 12 · 03 (BLIP-2 Q-Former)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 解释 gated cross-attention 如何通过 tanh(gate) = 0 在初始化时保留 frozen LLM 的文本能力。
- 走通 Perceiver resampler：N 个 image patches → K 个固定 “latent” queries via cross-attention。
- 描述 Flamingo 如何用尊重图像位置的 causal masking 处理 interleaved image-text sequences。
- 复现一个 few-shot multimodal prompt 结构：3 个 image-caption examples，然后是 query image。

## The Problem / 问题

BLIP-2 把 32 个 visual tokens 喂进 frozen LLM 的输入层。单图 prompt 很有效。但如果你想输入多张图，并和文本交错，例如“这是 image A，caption it；这是 image B，caption it；现在这是 image C，caption it”怎么办？LLM 的 self-attention 需要在一个流里处理 image tokens 与 text tokens，而且哪些位置能 attend 到哪些图会变得麻烦。

Flamingo 的答案是：不要改变 LLM 的 input stream。在已有 LLM blocks 之间插入额外 cross-attention layers。Text tokens 仍然像过去一样通过 LLM 的 causal self-attention。在每几个 LLM blocks 之间，text tokens 还会通过新的 gated layer cross-attend 到 image features。Gate 初始化为 0，意味着 step zero 时新层是 no-op，模型表现与 pretrained LLM 完全一致。训练推进后 gate 打开，视觉信息开始流动。

Flamingo 回答的第二个问题是：prompt 中图像数量可变（0、1 或多张）时怎么办？Perceiver resampler 是一个小 cross-attention module，它接收任意数量的 patches，输出固定数量的 visual latent tokens。无论 prompt 中有多少图像，LLM cross-attention layer 看到的 shape 都相同。

## The Concept / 概念

### The frozen LLM / 冻结的 LLM

Flamingo 从 frozen Chinchilla 70B LLM 开始。70B 权重全部不动。已有 text self-attention 和 FFN 正常运行。

### Perceiver resampler / Perceiver 重采样器

对 prompt 中每张图，ViT 输出 N 个 patch tokens。Perceiver resampler 有 K 个固定可学习 latents（Flamingo 使用 K=64）。每个 resampler block 包含两个子步骤：

1. Cross-attention：K 个 latents attend 到 N 个 patch tokens（Q 来自 latents，K/V 来自 patches）。
2. Latents 内部的 self-attention + FFN。

经过 6 个 resampler blocks 后，输出 K=64 个 dim 1024 的 visual tokens，不管 ViT 产生多少 patch。224x224 图像（196 patches）和 480x480 图像（900 patches）都输出 64 个 resampler tokens。

对视频，resampler 按时间应用：每帧 patches 产生 64 个 latents，temporal positional encoding 让模型区分 t=0 与 t=N。完整视频变成 T * 64 个 visual tokens。

### Gated cross-attention / 带门控的 cross-attention

在 frozen LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的 gated cross-attention block：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是初始化为 0 的可学习标量。
- `tanh(0) = 0`，所以初始化时 gated branch 贡献为 0。
- 当 `alpha` 偏离 0，cross-attention contribution 会平滑增大。
- residual connection 表示即使 gate 完全打开，也不会覆盖 LLM 的文本 representation，而是在其上添加视觉信息。

这是 Flamingo 最重要的设计：visual conditioning 是 additive、gated，并且初始化为 0。Step 0 的 Flamingo 在 text-only inputs 上就是一个完美的 Chinchilla 70B。

### Masked cross-attention for interleaved inputs / 面向 interleaved input 的 masked cross-attention

在类似 “<image A> caption A <image B> caption B <image C> ?” 的 prompt 中，每个 text token 只应该看到序列中它之前出现的图像。Cross-attention mask 强制执行：位置 `t` 的 text token 只 attend 到 image index `i < i_t` 的 image resampler tokens，其中 `i_t` 是位置 `t` 之前最近的图像 index。“只看最近一张 preceding image”与“看所有 preceding images”都是合理选择；Flamingo 选择前者。

### In-context few-shot learning / 上下文 few-shot learning

Flamingo prompt 看起来像：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到 completion pattern 后，会输出 “bird”（或 image3 实际内容）。没有梯度步骤。Frozen LLM 的 in-context learning 能力通过 gated cross-attention 延续下来，这是论文的 punchline，也是它的重要性所在。

### Training data / 训练数据

Flamingo 使用三类数据训练：

1. MultiModal MassiveWeb (M3W)：43M 个带 interleaved images/text 的网页，重建阅读顺序。
2. Image-Text Pairs（ALIGN + LTIP）：4.4B pairs。
3. Video-Text Pairs（VTP）：27M 个短视频 clips。

OBELICS（2023）是 interleaved web corpus 的 open reproduction，Idefics、Idefics2 和大多数 open “Flamingo-like” models 都在它上面训练。

### OpenFlamingo and Otter / OpenFlamingo 与 Otter

OpenFlamingo（2023）是 open reproduction。架构相同（Perceiver resampler + frozen LLaMA 或 MPT 上的 gated cross-attention）。Checkpoints 有 3B、4B、9B。由于 base LLM 更小、数据更少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 上用 MIMIC-IT（multimodal instructions 数据集）做 instruction tuning，说明 gated cross-attention 同样适合 instruction following。

### The descendants / 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的 gated cross-attention lineage，逐步简化（Idefics2 去掉 resampler，改用 direct patch tokens + adaptive pooling）。
- Flamingo-to-Chameleon transition：到 2024 年，许多团队转向 early-fusion（Lesson 12.11）；但在必须冻结 backbone 的生产场景，Flamingo-style gated cross-attention 仍然存在。
- Gemini 的 interleaved input：概念上继承 Flamingo 的 interleaved-format 灵活性，尽管确切机制是 proprietary。

### Comparison to BLIP-2 / 与 BLIP-2 对比

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Q-Former once at input | Gated cross-attention at every M layers |
| Visual tokens | 32 per image | 64 per image per cross-attn layer |
| Frozen LLM | Yes | Yes |
| Few-shot in-context | Weak | Strong — the paper's centerpiece |
| Interleaved inputs | No native support | Yes, the design target |
| Training data | 130M pairs | 1.3B pairs + 43M interleaved pages |
| Parameter count | 188M trained | ~10B trained (cross-attn layers) |
| Compute | Days on 8 A100s | Weeks on thousands of TPUv4 |

预算内做单图 VQA 选 BLIP-2。需要 interleaved、few-shot 或 multi-image reasoning，选 Flamingo/Idefics2。

## Build It / 动手构建

本课动手部分聚焦两个 Flamingo 原语：Perceiver resampler 把可变 patch 序列压成固定 latent tokens；gated cross-attention 用 `tanh(alpha)` 控制视觉分支从 no-op 平滑打开。再加上 interleaved attention mask，你就能看到 few-shot 多模态 prompt 如何保持文本 causal 语义。

## Use It / 应用它

`code/main.py` 演示：

1. 在 36 个 fake patch tokens 上运行 Perceiver resampler，并使用 8 个 learnable latents（pure Python cross-attention）。
2. 一个 gated cross-attention step：`alpha = 0` 时输出等于输入（LLM unchanged），`alpha = 2.0` 时混入视觉贡献。
3. 一个 interleaved-mask builder，为 “(image 1) (text 1) (image 2) (text 2)” 序列生成 2D attention mask。

## Ship It / 交付它

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定 open VLM 的 config（是否有 resampler、cross-attn frequency、gate scheme），它会识别 Flamingo lineage 元素并解释 freezing strategy。调试 fine-tune 后文本能力下降时很有用（常见原因：gate 打开得太快太大）。

## Exercises / 练习

1. 计算 Flamingo-9B 的 visual parameter count：9B LLM + 1.4B gated cross-attention layers + 64M resampler。被训练的参数占总参数多少？

2. 在 PyTorch 中实现 gated residual `y = tanh(alpha) * cross + x`。实验展示 `alpha=0` 时初始化上 `y==x` 完全成立。

3. 阅读 OpenFlamingo Section 3.2（arXiv:2308.01390），了解当 batch 中每个 prompt 的图像数量不同，他们如何处理 multiple images。描述 padding strategy。

4. 为什么 Flamingo 的 cross-attention mask 让 text token 只 attend 到最近的 preceding image，而不是所有 preceding images？阅读 Flamingo 论文 Section 2.4 并解释 tradeoff。

5. In-context few-shot：为一个新的 Flamingo variant 构造 4 个“image → main object color”示例的 prompt。描述 examples 数量从 0 到 8 变化时的预期 accuracy pattern。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Perceiver resampler | “Fixed-latent cross-attention” | 从可变数量 input patches 生成 K 个固定 tokens 的模块 |
| Gated cross-attention | “Tanh-gated bridge” | Residual layer `y = tanh(alpha)*cross + x`，alpha 可学习且初始化为 0 |
| Interleaved input | “Mixed sequence” | 图像和文本按阅读顺序自由混合的 prompt 格式 |
| Frozen LLM | “No LLM gradients” | 文本 LLM 权重不更新；只训练 resampler + cross-attn layers |
| Few-shot | “In-context examples” | 在 prompt 中给几个 `(image, answer)` pairs，模型无需 fine-tuning 即可泛化 |
| OBELICS | “Interleaved web corpus” | 141M 个按阅读顺序组织 image/text 的 open web pages 数据集 |
| Chinchilla | “70B frozen base” | Flamingo 的 frozen text LLM，来自 DeepMind Chinchilla 论文 |
| Gate schedule | “How alpha moves” | 训练中 cross-attention gate 打开的速率 |
| Cross-attn frequency | “Every M layers” | 插入 gated cross-attention block 的频率；Flamingo 使用 M=4 |
| OpenFlamingo | “Open reproduction” | MosaicML/LAION 的 3-9B open checkpoint，架构与 Flamingo 相同 |

## Further Reading / 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — open reproduction。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — interleaved web corpus。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver architecture。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — instruction-tuned Flamingo descendant。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 路线的现代简化。
