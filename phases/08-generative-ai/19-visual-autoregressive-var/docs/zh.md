# Visual Autoregressive Modeling (VAR): Next-Scale Prediction / 视觉自回归建模：下一尺度预测

> Diffusion models 按时间迭代采样（denoising steps）。VAR 按尺度迭代采样：先预测 1x1 token，再预测 2x2，再到 4x4，直到最终分辨率，每个尺度都 condition 在前一尺度上。2024 年论文证明，VAR 在图像生成上符合 GPT-style scaling laws，并在相同 compute budget 下超过 DiT。本课构建核心机制。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (with PyTorch)
**Prerequisites / 前置知识：** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 解释 next-scale prediction 如何修复 pixel/token raster-order 的生成顺序问题
- 理解 multi-scale VQ tokenizer、residual VQ 和 scale-structured transformer 的关系
- 区分 VAR 的 causal-across-scales 与 parallel-within-scale 生成机制
- 判断 VAR 与 diffusion/DiT 在 inference passes、scaling law 和 text conditioning 上的取舍

## The Problem / 问题

Autoregressive generation 主导语言建模，因为它可预测地 scaling：更多 compute、更多参数、更低 perplexity、更好输出。2024 年之前，图像生成有两类主要 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE codes 上逐 token）。

两者都受 generation-order problem 困扰。Pixels 和 tokens 排在 2D 网格里，但 AR 模型必须按 1D raster order 访问它们。早期角落像素根本不知道整张图最终会变成什么。生成质量的 scaling 比 GPT-on-text 差，在 matched compute 下也达不到 diffusion-model quality。

VAR 通过改变“正在生成的东西”来修复 generation-order problem。它不按空间逐个预测 image tokens，而是按越来越高的分辨率预测整张图。Step 1：预测 1x1 token（整体图像 “summary”）。Step 2：预测 2x2 token grid（更粗特征）。Step 3：预测 4x4 grid。Step K：预测最终 (H/8)x(W/8) grid。

每个尺度 attend 到所有前序尺度（按 “scale order” causal），并在自身尺度内并行。顺序问题消失：scale k 的整张图由一次 transformer pass 生成。

## The Concept / 概念

### VQ-VAE Multi-Scale Tokenizer / VQ-VAE 多尺度 tokenizer

VAR 需要一个 **multi-scale discrete tokenizer**。对图像 x，它产生一串逐渐更高分辨率的 token grids：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用同一个 codebook（典型大小 4096-16384）。各尺度 tokenization 不是独立的，它被训练成让各尺度 residual 求和能重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是 **residual VQ** 变体。Scale k 捕捉 scales 1..k-1 没捕捉到的内容。Decoder 接收所有 scale embeddings 的和并生成图像。

Multi-scale VQ tokenizer 像 VQGAN 一样先训练一次，再 freeze。所有生成工作都由上层 autoregressive model 完成。

### Next-Scale Prediction / 下一尺度预测

生成模型是一个 transformer，它看到所有前序尺度 tokens，并预测下一尺度 tokens。

Input sequence structure:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embeddings 同时编码 scale index 和该 scale 内的 spatial position。Attention 按 scale order causal：scale k、position (i, j) 的 token 可以 attend 到 scales 1..k 的所有 tokens，以及 scale k 内按某种 intra-scale order 更早的 tokens（VAR 使用 fixed positional attention，没有 intra-scale causality：同一尺度内所有位置并行预测）。

Training loss：在每个 scale k 上，给定所有 prior-scale tokens 预测 z_k tokens。对离散 VQ codes 做 cross-entropy loss。结构和 GPT 相同，只是这里的 “sequence” 具有尺度结构。

### Generation / 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

如果 K = 10 scales，生成就是 10 次 transformer forward passes。每次 pass 并行产生整层 scale，不在该 scale 内逐 token autoregress。对 256x256 图像，这大约是 10 passes，而 DiT 是 28-50 passes。

### Why Next-Scale Wins Over Next-Token / 为什么 next-scale 胜过 next-token

三个结构性优势：

1. **Coarse-to-fine aligns with natural image statistics / 粗到细符合自然图像统计。** 人类视觉和图像数据集都有 scale-dependent regularities：低频结构稳定且可预测，高频细节依赖低频内容。Next-scale prediction 利用了这一点。
2. **Parallel generation within scale / 尺度内并行生成。** 不像 GPT-style token AR，VAR 一步生成某个尺度的所有 tokens。有效生成长度是 log-scale，而不是 linear。
3. **No generation order bias / 没有生成顺序偏置。** Scale k 的 tokens 能看到完整 scale k-1；不存在 “left-of” 或 “above” 偏置，不会强迫早期 tokens 在缺少后续上下文时先承诺。

### Scaling Law / Scaling law

Tian et al. 证明 VAR 在 ImageNet 上的 FID 遵循 power-law scaling curve，就像 GPT 的 perplexity 一样。参数或 compute 翻倍，会可靠地降低 error。这是第一个像语言模型一样干净表现出 scaling behavior 的 image-generative model。结果是，VAR-scale predictions 可以从 compute 预测，而不是每个架构都靠经验猜。

### Relationship to Diffusion / 与 diffusion 的关系

VAR 和 diffusion 共享同一套 data-compression 思路：都把生成问题拆成一串更简单的子问题。

- Diffusion：逐步加噪，学习撤销一步。
- VAR：逐步增加分辨率，学习预测下一尺度。

它们是穿过问题空间的两条不同轴。两者都产生 tractable conditional distributions。经验上，VAR 推理更快（passes 更少，且每个尺度内部全并行），并在 class-conditional ImageNet 上匹配或超过 DiT。Text-conditional VAR（VARclip、HART）仍是活跃研究方向。

## Build It / 动手构建

在 `code/main.py` 中，你会：
1. 在 synthetic “image” data（2D Gaussian rings）上构建一个小型 **multi-scale VQ tokenizer**。
2. 训练一个 **VAR-style transformer** 做 next-scale-predict tokens。
3. 通过调用 transformer 4 次（4 scales）进行 sampling 和 decoding。
4. 验证按 scale ordering 训练后，同一 scale 内生成可以并行。

这是 toy implementation。重点是看到 scale-structured attention mask，以及 parallel-within-scale generation 真的在工作。

## Use It / 应用它

VAR 适合你关心 inference pass 数、图像离散 tokenization 和可预测 scaling 的场景：class-conditional image generation、研究 next-scale scaling law、设计多尺度 tokenizer，或探索 text-conditioned VAR（如 HART）。如果任务已经需要复杂 spatial conditioning、编辑、inpainting 或成熟生产生态，2026 年仍优先使用 diffusion / flow-matching stack；VAR 更像下一代图像 AR 的研究与高性能生成候选。

## Ship It / 交付它

本课产出 `outputs/skill-var-tokenizer-designer.md`：一个用于设计 multi-scale tokenizer 的 skill，包括 scales 数量、scale ratios、codebook size、residual sharing、decoder architecture。

## Exercises / 练习

1. **Scale count ablation / 尺度数量消融。** 用 4、6、8、10 scales 训练 VAR。测量 reconstruction quality 与 autoregressive passes 数量的关系。更多 scales = 更细 residuals = 更好质量，但 passes 更多。

2. **Codebook size / Codebook 大小。** 训练 codebook sizes 为 512、4096、16384 的 tokenizers。更大 codebook 有更好 reconstruction，但 prediction 更难。找到 knee。

3. **Parallel-within-scale check / 尺度内并行检查。** 对训练好的 VAR，显式测量 attention pattern。Scale k 内，模型是否 attend 到 cross-scale positions 而不是 intra-scale？验证 mask implementation。

4. **VAR vs DiT scaling / VAR 与 DiT scaling。** 对同一个 ImageNet class-conditional task，在 matched param budgets（例如 33M、130M、458M）下训练 VAR 和 DiT。画出 FID vs compute。VAR 应该在每个 size 上领先 DiT；在小规模复现论文结果。

5. **Text conditioning / 文本条件。** 扩展 VAR，让它通过 adaLN 接收一个 text embedding（CLIP pooled）作为额外 conditioning input。这是 HART recipe。它对 text-aligned sampling 的 FID 改善多少？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | 通过 VQ token grids 金字塔上的 next-scale prediction 做图像生成 |
| Next-scale prediction | "Predict coarser, then finer" | 模型按逐渐提升的 resolution scales 预测 tokens，并 condition 在所有前序尺度上 |
| Multi-scale VQ tokenizer | "Residual VQ" | 产生 K 个递增分辨率 token grids 的 VQ-VAE，decoder 求和所有尺度 |
| Scale k | "Pyramid level k" | K 个 resolution levels 之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| Parallel-within-scale | "One forward per scale" | Scale k 的所有 tokens 在一次 transformer pass 中预测，而不是自回归逐个预测 |
| Causal-across-scales | "Scale-ordered attention" | Scale k 的 token 可 attend 到 scales 1..k，但不能 attend 到 k+1..K |
| Residual VQ | "Additive tokenization" | 每个尺度 tokens 编码低尺度留下的 residual；decoder 求和所有 scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID 随 compute 遵循可预测 power law，类似语言模型 perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant，把 MaskGIT-style iterative decoding 与 VAR scale structure 结合 |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding 同时携带 scale index 和该 scale 内 spatial coordinates |

## Further Reading / 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，canonical reference
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，diffusion comparison baseline
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR multi-scale tokenizer 扩展的 tokenizer family
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，discrete image tokenization 的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditional VAR
