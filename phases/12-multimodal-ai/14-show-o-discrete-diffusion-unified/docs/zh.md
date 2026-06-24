# Show-o and Discrete-Diffusion Unified Models / Show-o 与离散扩散统一模型

> Transfusion 混合 continuous 与 discrete representations。Show-o（Xie et al., 2024 年 8 月）反向选择：text tokens 用 causal next-token prediction，image tokens 用 MaskGIT 风格的 masked discrete diffusion。两者位于同一个 transformer 中，并使用 hybrid attention mask。结果是在一个 backbone、每个 modality 一个 tokenizer、一个 loss formulation（next-token 扩展到 masked prediction）上统一 VQA、text-to-image、inpainting 和 mixed-modality generation。本课讲 Show-o 设计：为什么 masked discrete diffusion 是并行、少步的 image generator，并对比 Transfusion 与 Emu3。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites / 前置知识：** Phase 12 · 13 (Transfusion)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 解释 masked discrete diffusion：均匀 mask tokens，再让 transformer 恢复它们的 schedule。
- 比较 parallel image decoding（Show-o、MaskGIT）与 autoregressive image decoding（Chameleon、Emu3）的速度和质量。
- 说出 Show-o 单 checkpoint 处理的三个任务：T2I、VQA、image inpainting。
- 选择 masking schedule（cosine、linear、truncated），并分析它对 sample quality 的影响。

## The Problem / 问题

Transfusion 的 two-loss training 有效，但训练动态更复杂：continuous diffusion loss 与 discrete NTP loss 的 numerical scale 不同。平衡 loss weights 是一个 hyperparameter search。架构有效，但复杂。

Show-o 的答案是：像 Chameleon 一样保持两个 modality 都离散，但用 masked discrete diffusion 并行生成图像，而不是顺序生成。训练目标变成单一 masked-token-prediction，自然推广 next-token-prediction。

## The Concept / 概念

### Masked discrete diffusion (MaskGIT) / Masked discrete diffusion（MaskGIT）

Chang et al.（2022）提出的 MaskGIT 技巧很优雅。从 fully-masked image 开始（每个 token 都是特殊 `<MASK>` id）。每一步并行预测所有 masked tokens，然后保留 top-K 最有信心的预测，把其余重新 mask。大约 8-16 次迭代后，所有 token 填完。每一步 unmask 多少 token 的 schedule 需要调；cosine schedules 通常很好。

训练很简单：从 [0, 1] 中均匀采样 masking ratio，应用到图像的 VQ tokens，训练 transformer 恢复 masked tokens。这和 BERT 对文本做的事一样，只是扩展到 image generation。

### Show-o: one transformer, hybrid mask / Show-o：一个 transformer，hybrid mask

Show-o 把 MaskGIT 放进 causal-language-model transformer。Attention mask 是：

- Text tokens：causal（标准 LLM）。
- Image tokens：图像 block 内 full bidirectional（masked tokens 在预测时能看到其他 image tokens）。
- Text-to-image：text attends to prior images，image attends to prior text。

训练在以下任务间切换：

1. Text sequences 上的标准 NTP。
2. T2I samples：text → image with masked image tokens，masked-token-prediction loss。
3. VQA samples：image → text with masked text tokens（实际就是 NTP）。

Unified loss 是 `<MASK>` tokens 上的 cross-entropy，它覆盖 text NTP（只有最后一个 token “masked”）和 image masked-diffusion（随机子集 masked）。

### Parallel sampling / 并行采样

Show-o 约 16 步生成一张图，而不是 ~1000（按 token 自回归）或 ~20（diffusion）。每步并行预测全部 masked tokens；commit top-K confident；重复。

对比：

- Chameleon / Emu3（对 tokens 自回归）：N_tokens 次 forward passes，典型每图 1024-4096。
- Transfusion（continuous diffusion）：约 20 steps，每步一次 full transformer pass。
- Show-o（masked discrete diffusion）：约 16 steps，每步一次 full transformer pass。

Show-o 比 Chameleon 快，在类似规模下大致匹配 Transfusion 的 step count，但每步成本更低（discrete vocab logits vs continuous MSE loss）。

### Tasks in one checkpoint / 一个 checkpoint 中的任务

Show-o 在推理时通过 prompt format 选择四类任务：

- Text generation：标准 autoregressive text output。
- VQA：image in, text out。
- T2I：text in, image out，通过 masked discrete diffusion。
- Inpainting：image 中某些 tokens masked，填补它们。

Inpainting 能力来自 masked-prediction training。Mask 掉 VQ-token grid 中一个区域，输入剩余部分加 text prompt，预测 masked tokens。

### Masking schedule / Masking schedule

每一步 unmask 多少 token 的 schedule 会影响质量。Show-o 推荐 cosine：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Step 0 时所有 tokens masked（ratio 1.0）。Step T 时无 mask。Cosine 把质量集中在中间 ratio 区间，那里 prediction 最有信息量。Linear schedules 也能工作，但更快 plateau。

### Show-o2 / Show-o2

Show-o2（2025 follow-up, arXiv 2506.15564）scale 了 Show-o：更大的 LLM base、更好的 tokenizer、改进 mask schedule。架构模式相同。

### Where Show-o sits / Show-o 的位置

在 2026 年 taxonomy 中：

- Discrete tokens + NTP：Chameleon、Emu3。简单但推理慢。
- Discrete tokens + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行采样，但仍受 tokenizer 有损限制。
- Continuous + diffusion：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- Continuous + flow matching in a VLM：JanusFlow、InternVL-U。最新路线。

按任务选择：如果你要在 open model 中同时支持 T2I + inpainting + VQA 且需要合理速度，选 Show-o；如果质量最重要且能承受 two-loss plumbing，选 Transfusion。

## Build It / 动手构建

本课动手实现 masked-discrete-diffusion sampler：从全 mask 的 VQ grid 开始，每一步用 cosine schedule 决定保留多少高置信 token。你会看到 parallel decoding 为什么只需少量 full passes，而不必对每个 image token 自回归。

## Use It / 应用它

`code/main.py` 模拟 Show-o sampling：

- 一个 16 个 VQ tokens 的 toy grid。
- 一个 mock “transformer”，根据 prompt 和当前 unmasked tokens 预测 logits。
- 使用 cosine schedule 的 8-step parallel masked sampling。
- 打印中间状态（mask pattern evolution）和最终 tokens。

运行它，观察 mask 如何一步步消失。

## Ship It / 交付它

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个同时需要 understanding（VQA、captioning）和 generation（T2I、inpainting）的 open-weights 产品，它会在 Show-o family、Transfusion/MMDiT family、Emu3 / Chameleon family 之间选择，并给出具体 trade-offs。

## Exercises / 练习

1. Masked discrete diffusion 约 16 步采样。为什么不是 1 步？如果在 step 0 unmask 全部，会坏在哪里？

2. Inpainting 对 masked diffusion 来说几乎免费。提出一个真实或假设产品场景，让 Show-o 的 inpainting 优于 specialist model。

3. Cosine schedule vs linear schedule：对 T=8，追踪每步 unmasked tokens 数量。哪个更平衡？

4. 一张 512x512 Show-o 图像是 1024 tokens。vocab K=16384 时，模型输出 1024 * log2(16384) = 14,336 bits（约 1.75 KiB）数据。Stable Diffusion 输出 512*512*24 bits = 6,291,456 bits（约 768 KiB）raw pixels。compression ratio 是多少？它换来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的 class-conditional autoregressive image model 与 Show-o 的 masked approach 有什么不同？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Masked discrete diffusion | “MaskGIT-style” | 训练预测 masked tokens；推理时迭代 unmask 最有信心的预测 |
| Cosine schedule | “Unmask schedule” | 推理步中 mask ratio 的衰减；把 confidence growth 集中在中间区间 |
| Parallel decoding | “All tokens at once” | 每一步一次 forward 预测所有 masked tokens，再 commit top-K |
| Hybrid attention | “Causal + bidirectional” | Text tokens 上 causal、image blocks 内 bidirectional 的 mask |
| Inpainting | “Fill-in generation” | 以部分 tokens masked 的图像为条件，预测缺失部分；来自训练目标本身 |
| Commitment rate | “Top-K per step” | 每次迭代声明完成的 token 数；控制 inference vs quality trade-off |

## Further Reading / 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
