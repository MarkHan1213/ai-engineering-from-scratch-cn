# Janus-Pro: Decoupled Encoders for Unified Multimodal Models / Janus-Pro：统一多模态模型中的解耦 Encoder

> 统一多模态模型有一个不可避免的张力。Understanding 需要 semantic features：SigLIP 或 DINOv2 输出的向量富含概念级信息。Generation 需要 reconstruction-friendly codes：VQ tokens 能组合回清晰像素。这两个目标与单一 encoder 并不兼容。Janus（DeepSeek, 2024 年 10 月）和 Janus-Pro（DeepSeek, 2025 年 1 月）认为解决方法是不再强求：把两个 encoder 解耦。任务之间共享 transformer body，但 understanding 走 SigLIP，generation 走 VQ tokenizer。7B 的 Janus-Pro 在 GenEval 上超过 DALL-E 3，同时在 MMMU 上追平 LLaVA。本课解释为什么两个 encoder 比一个 encoder 更有效。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites / 前置知识：** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 解释为什么单一 shared encoder 会在 understanding 或 generation 质量上妥协。
- 描述 Janus-Pro routing：understanding 输入侧使用 SigLIP features，generation 输入与输出使用 VQ tokens。
- 追踪让 Janus-Pro 成功、而 Janus 未能成功的数据规模扩展。
- 比较 decoupled（Janus-Pro）、coupled-continuous（Transfusion）和 coupled-discrete（Show-o）架构。

## The Problem / 问题

Unified models 在 understanding 与 generation 之间共享 transformer body。此前尝试（Chameleon、Show-o、Transfusion）都让同一个 visual tokenizer 同时服务两个方向。Tokenizer 是妥协：

- 为 reconstruction 优化（generation）：VQ-VAE 捕捉细粒度像素细节，但 tokens 语义 coherence 较弱。
- 为 semantics 优化（understanding）：SigLIP embeddings 把 “cat” 图像聚到 “cat” tokens 附近，但无法良好重建。

Show-o 和 Transfusion 因此在某个方向上付出可见质量税。Janus-Pro 问：既然任务需求不同，为什么强迫一个 tokenizer 同时承担两者？

## The Concept / 概念

### Decoupled visual encoding / 解耦视觉编码

Janus-Pro 的架构分离两个 encoder：

- Understanding path。Input image → SigLIP-SO400m → 2-layer MLP → transformer body。
- Generation path。Input image（如果条件来自已有图像）→ VQ tokenizer → token IDs → transformer body。
- Output generation。Transformer 预测 image tokens → VQ decoder → pixels。

Transformer body 是共享的。Body 上游和下游是 task-specific。

Inputs 由 prompt format disambiguate：`<understand>` tag 路由到 SigLIP；`<generate>` 路由到 VQ。或者由 task 隐式决定。

### Why this works / 为什么有效

Understanding loss 得到 SigLIP features，而 CLIP-style pretraining 已经把这些 features 调成语义相似空间。相比 Show-o / Transfusion，模型的 perception benchmark 会提升，因为输入 features 更适合任务。

Generation loss 得到 VQ tokens，而 tokenizer 已经为 reconstruction 优化。相比 Show-o，图像质量提升，因为 VQ codes 能干净解码回 pixels。

Shared transformer body 看到两种 input distributions（SigLIP 与 VQ），并学习同时处理它们。Claim 是：足够的数据 + 足够参数，body 能吸收这种 switching。

### Data scaling — Janus vs Janus-Pro / 数据规模：Janus vs Janus-Pro

Janus（原始版，arXiv 2410.13848）引入了 decoupling，但规模较小（1.3B 参数、有限数据）。Janus-Pro（arXiv 2501.17811）扩展为：

- 7B 参数（vs 1.3B）。
- Stage 1（alignment）90M image-text pairs，高于 72M。
- Stage 2（unified）72M，高于 26M。
- Stage 3 加入 200k image-gen instruction samples。

结果：Janus-Pro-7B 在 MMMU 上追平 LLaVA（60.3 vs ~58），并在 GenEval 上超过 DALL-E 3（0.80 vs 0.67）。一个 open model 在统一谱系两端都具备竞争力。

### JanusFlow — the rectified flow variant / JanusFlow：rectified flow 变体

JanusFlow（arXiv 2411.07975）把 VQ generation path 换成 rectified-flow generation path（continuous）。分裂变成 SigLIP-for-understanding + rectified-flow-for-generation。质量上限进一步提高。架构仍然是 decoupled-encoders-shared-body。

### The shared body's job / Shared body 的职责

Transformer body 处理 unified sequence，但输入分布有两种。它的职责是：

- Understanding：消费 SigLIP features + text tokens → autoregressively 输出 text。
- Generation：消费 text tokens +（可选 image VQ tokens）→ autoregressively 输出 image VQ tokens。

Body 内没有 modality-specific weights per block。它就是你会在 Qwen 或 Llama 中看到的 text-style transformer，加上两个 input adapters。

有趣的是，这意味着 Janus-Pro 的 body 可以从 pretrained LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这很重要：LLM 提供了纯 from-scratch unified models 难以达到的 reasoning ability。

### Compared to InternVL-U / 与 InternVL-U 对比

InternVL-U（Lesson 12.10）是 2026 年 follow-up。它结合：

- Native multimodal pretraining（InternVL3 backbone）。
- Decoupled-encoder routing（SigLIP in，VQ + diffusion heads out）。
- Unified understanding + generation + editing。

InternVL-U 把 Janus-Pro 的架构选择吸收到更大框架里。Decoupled-encoder idea 已经成为大规模 unified models 的默认选择。

### Limitations / 限制

Decoupled encoders 增加架构复杂度。两个 tokenizers 要训练，两个 input paths 要维护，也有两组 failure modes。若产品不需要 generation，Janus-Pro 是过度工程；选 LLaVA-family understanding model。

若产品不需要 understanding，Janus-Pro 又过于复杂；选 Stable Diffusion 3 / Flux。

若产品同时需要二者，Janus-Pro 是当前参考 open architecture。

## Build It / 动手构建

本课动手构建一个 routing simulator：根据 task tag 在 SigLIP-like semantic encoder 与 VQ-like reconstruction encoder 之间切换，把两种输出都送进 shared body。你会看到解耦 encoder 如何让 understanding 与 generation 各用适合自己的表征。

## Use It / 应用它

`code/main.py` 模拟 Janus-Pro routing：

- 两个 mock encoders：SigLIP-like（产生 256-dim semantic vectors）和 VQ-like（产生 integer codes）。
- 一个 prompt router，根据 task tag 选择 encoder。
- 一个 shared body（stand-in），无论 token 来自哪个 encoder 都处理 token sequences。
- 一个从 stage 1（alignment）切换到 stage 3（instruction tune）的 weighted-sample schedule。

打印 3 个例子的 routed paths：image QA、T2I、image editing。

## Ship It / 交付它

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望同时具备 frontier-ish generation + understanding 质量的产品，它会在 Janus-Pro、JanusFlow、InternVL-U 之间选择，并给出具体 data-scale recommendation。

## Exercises / 练习

1. Janus-Pro-7B 在 GenEval 上超过 DALL-E 3。解释为什么 7B open model 可以在 generation 上追平 frontier proprietary model，却不能在 understanding 上追平。

2. 实现 router function：给定 prompt text，分类为 `understand` 或 `generate`。如何处理 “describe and then sketch” 这种 ambiguous prompt？

3. JanusFlow 用 rectified flow 替换 VQ path。Transformer body 现在输出什么？Loss 发生什么变化？

4. 提出 Janus-Pro 架构可用一个额外 decoupled encoder 支持的第四种任务。例如 image segmentation（DINO-style）、depth（MiDaS-style）。

5. 阅读 Janus-Pro Section 4.2 关于 data scaling。哪个 data stage 对 Janus 到 Janus-Pro 的 T2I quality gain 贡献最大？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Decoupled encoding | “Two visual encoders” | 每个方向使用独立 tokenizer 或 encoder：理解用 semantic，生成用 reconstruction |
| Shared body | “One transformer” | 单个 transformer 处理任一 encoder 输出；没有 modality-specific weights |
| SigLIP for understanding | “Semantic features” | CLIP-family vision tower，提供丰富概念特征，但不利于重建 |
| VQ for generation | “Reconstruction codes” | Vector-quantized tokens，能干净解码回 pixels |
| JanusFlow | “Rectified-flow variant” | Janus-Pro 的连续 flow-matching generation head 版本，替代 VQ |
| Routing tag | “Task tag” | 选择 input encoder 的 prompt marker（`<understand>` / `<generate>`） |

## Further Reading / 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
