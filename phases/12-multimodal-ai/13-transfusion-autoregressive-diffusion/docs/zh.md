# Transfusion: Autoregressive Text + Diffusion Image in One Transformer / Transfusion：一个 Transformer 中的自回归文本与扩散图像

> Chameleon 和 Emu3 把赌注全压在 discrete tokens 上。它们有效，但 quantization bottleneck 很明显：图像质量 plateau 低于 continuous-space diffusion models。Transfusion（Meta, Zhou et al., 2024 年 8 月）做了相反选择：保留连续图像表示，完全去掉 VQ-VAE，用两个 loss 训练一个 transformer。Text tokens 使用 next-token-prediction。Image patches 使用 flow-matching / diffusion loss。两个目标优化同一组权重。Stable Diffusion 3 背后的 MMDiT 是近亲。本课读取 Transfusion thesis，构建 toy two-loss trainer，并追踪一个 transformer 同时做两件事所需的 attention mask。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites / 前置知识：** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 连接一个 transformer，使其在同一 backbone 上运行两个 losses：文本 token 上的 NTP，以及 image patches 上的 diffusion MSE。
- 解释为什么 image patches 内部使用 bidirectional attention、text tokens 上使用 causal attention 是正确 mask choice。
- 从 compute、quality 和 code complexity 对比 Transfusion-style（continuous images, diffusion loss）与 Chameleon-style（discrete images, NTP）。
- 说出 MMDiT 的贡献：每个 block 中 modality-specific weights，residual stream 上 joint attention。

## The Problem / 问题

Discrete vs continuous image tokens 的争论早于 LLM。Continuous representations（raw pixels、VAE latents）保留细节。Discrete tokens（VQ indices）符合 transformer native vocabulary，但在 quantization step 丢细节。

Chameleon / Emu3 走 discrete：一个 loss、一个 architecture，但 image fidelity 受 tokenizer quality 限制。

Diffusion models 走 continuous：图像质量极强，但与 LLM 是分开的模型，noise-schedule engineering 复杂，也没有与 text generation 的干净整合。

Transfusion 问：能否两者兼得？保留 continuous images，同时仍训练一个模型，把两个 losses 缝进一次 gradient step。

## The Concept / 概念

### The two-loss architecture / 双损失架构

一个 decoder-only transformer 处理包含以下内容的 sequence：

- Text tokens（离散，来自 BPE vocab）。
- Image patches（连续，16x16 pixel blocks 通过线性 embedding 投影到 hidden dim，和 ViT encoder 的输入一样）。
- 标记 continuous patches 位置的 `<image>` 和 `</image>` tags。

Forward pass 只跑一次。Loss 按 token 类型选择 head：

- Text tokens：vocab-logits head 上的标准 cross-entropy。
- Image patches：continuous patches 上的 diffusion loss，预测加到每个 patch 上的 noise。

梯度流过 shared transformer body。两个 losses 同时改进共享权重。

### Attention mask: causal text + bidirectional image / Attention mask：文本 causal、图像 bidirectional

Text tokens 必须 causal：不能让 text token attend 到 future text，否则 teacher forcing 会被破坏。Image patches 表示同一张 snapshot，应当在同一个 image block 内彼此 bidirectionally attend。

Mask：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

训练和推理中实现为 block-triangular mask。

### Diffusion loss inside the transformer / Transformer 内部的 diffusion loss

Diffusion loss 是标准形式：给 image patch 加 noise，让模型预测 noise（或等价地预测 clean patch）。Transfusion 使用 flow matching，预测 noisy 到 clean 的 velocity field。

训练时：

1. 对每个 image patch x0，采样随机 timestep t。
2. 采样 noise ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. Transformer 预测 v_theta(xt, t)；loss = MSE(v_theta(xt, t), ε - x0)。
4. 与同一 sequence 中的 text NTP losses 一起 backprop。

推理时，generation 是：

- Text tokens：标准 autoregressive sampling。
- Image patches：在 prior text tokens 条件下跑 diffusion sampling loop（典型 10-30 steps）。

### MMDiT: Stable Diffusion 3's variant / MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser et al., 2024 年 3 月）在 Transfusion 附近发布了 MMDiT（Multimodal Diffusion Transformer）。两者是 sibling architectures。

MMDiT 的关键差异：

- 每个 block 有 modality-specific weights。每个 transformer block 对 text tokens 与 image patches 使用分开的 Q、K、V 和 MLP weights。Attention 是 joint（cross-modality）；其他部分 modality-specific。
- Rectified flow training。特定 flow-matching 变体，比 DDPM 更简单，采样更清楚。
- Scale。MMDiT 是 SD3（2B 和 8B 参数变体）的 backbone。Transfusion 论文 scale 到 7B。

两者汇聚到同一个核心想法：一个 transformer 在 text 上跑 NTP，在 continuous image representations 上跑 diffusion。

### Why this beats Chameleon-style / 为什么它优于 Chameleon-style

Continuous-diffusion 与 discrete-NTP 在 image generation 上的质量差距可测。Transfusion 论文报告：

- 7B 参数下，在 FID 上比同规模 Chameleon-style model 好 3-5 点。
- 不需要训练 tokenizer，image encoder 更简单（线性投影到 hidden，和 ViT input layer 一样）。
- Image patch denoising 可并行化，不像 autoregressive image tokens。

缺点是 Transfusion 是 dual-loss model，训练动态更棘手。Loss weights 需要调。NTP 与 diffusion schedule mismatch 会导致某个 head 支配训练。

### What sits downstream / 后续分支

Janus-Pro（Lesson 12.15）通过解耦 understanding 与 generation 的 vision encoder 来细化 Transfusion 思路：理解用 SigLIP，生成用 VQ；共享 transformer body。Show-o（Lesson 12.14）把 diffusion 换成 discrete-diffusion（masked prediction）。Unified-generation family 在 Transfusion 后快速分叉。

2026 年会输出图像的 production VLM（Gemini 3 Pro、GPT-5、Claude Opus 4.7 的 image generation path）几乎肯定使用这个家族的某种后代。细节是 proprietary。

## Build It / 动手构建

本课构建一个 tiny Transfusion：同一个 shared body 接受 text captions 与 tiny image grids，对文本位置计算 cross-entropy，对图像 patch 位置计算 noisy-patch MSE，并显式构造 block-triangular attention mask。重点是 two-loss plumbing，而不是真实图像质量。

## Use It / 应用它

`code/main.py` 在 tiny MNIST-like problem 上构建 toy Transfusion：

- Text captions 是描述数字（0-9）的短整数序列。
- Images 是 4x4 byte grids。
- 一对 shared-weight linear projections 作为 transformer stand-in；文本上 NTP loss，noisy patches 上 MSE loss。
- Training loop 交替两个 losses，attention mask 显式展示。
- Generation 在一次 forward pass 中生成 text caption 和 4x4 image。

Transformer 是 toy。真正的 artifacts 是 two-loss plumbing、attention mask construction 和 inference loop。

## Ship It / 交付它

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定新的 multimodal training task（text + image、text + audio、text + video），它会设计 two-loss schedule（loss weights、mask shape、shared vs modality-specific blocks）并标记实现风险。

## Exercises / 练习

1. 一个 Transfusion-style model 训练时 70% text tokens、30% image patches。Image diffusion loss 的 magnitude 约为 text NTP loss 的 10x。什么 loss weights 能平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现 block-triangular mask。把每个 entry 标成 0 或 1。

3. MMDiT 有 modality-specific QKV weights。相比 Transfusion 完全共享 transformer，它增加多少参数量？在 7B 参数规模上值得吗？

4. Generation：给定 text prompt，模型先跑 NTP 生成 50 tokens，然后遇到 `<image>`，再对 256 patches 做 20 步 denoise。总共多少 forward passes？

5. 阅读 SD3 paper Section 3。描述 rectified flow，并解释它为什么比 DDPM 需要更少 inference steps。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Two-loss training | “NTP + diffusion” | 单个 transformer 在同一 gradient step 中同时优化 text token cross-entropy 与 continuous image patch MSE |
| Flow matching | “Rectified flow” | 预测从 noise 到 clean data 的 velocity field 的 diffusion 变体；数学比 DDPM 简单 |
| MMDiT | “Multimodal DiT” | Stable Diffusion 3 架构：joint attention，modality-specific MLPs and norms |
| Block-triangular mask | “Causal text + bidirectional image” | 跨 text causal、image region 内 bidirectional 的 attention mask |
| Continuous image representation | “No VQ” | 图像 patch 是 real-valued vectors，而不是 integer codebook indices |
| Velocity prediction | “v-parameterization” | 网络输出的是 noise 与 data 之间的 velocity field，而不是 noise 本身 |

## Further Reading / 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
