# From CLIP to BLIP-2 — Q-Former as Modality Bridge / 从 CLIP 到 BLIP-2：作为模态桥的 Q-Former

> CLIP 能对齐图像和文本，但不能生成 caption、回答问题或进行对话。BLIP-2（Salesforce, 2023）用一个小型可训练 bridge 解决了这件事：32 个可学习 query vectors 通过 cross-attention 读取 frozen ViT 的 features，然后直接插入 frozen LLM 的输入流。188M 个 bridge 参数把一个 11B LLM 接到 ViT-g/14 上。到 2026 年，所有 adapter-based VLM（MiniGPT-4、InstructBLIP、LLaVA 的近亲）都可看作它的后代。本课读取 Q-Former 架构、解释两阶段训练，并构建一个 toy 版本，把 visual tokens 喂给 frozen text decoder。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites / 前置知识：** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 解释为什么在 frozen vision encoder 与 frozen LLM 之间放一个可训练 bottleneck，比 end-to-end finetuning 更省成本、更稳定。
- 实现一个 cross-attention block，让固定数量的 learnable queries attend 到外部 image features。
- 走通 BLIP-2 的两阶段预训练：representation（ITC + ITM + ITG），然后 generative（frozen decoder 上的 LM loss）。
- 将 Q-Former 与 LLaVA 使用的更简单 MLP projector 对比，并说明各自何时胜出。

## The Problem / 问题

你有一个 frozen ViT，每张图输出 256 个 dim 1408 的 patch token。你还有一个 frozen 7B LLM，它期望 dim 4096 的 token embedding。最直接的 bridge 是一个从 1408 到 4096 的线性层，这确实可用，但把全部 256 个 patch token 喂进 LLM context，会让每张图额外消耗 256 个 token。batch 32 张图时，光视觉模态就吃掉 8192 个 token。

BLIP-2 的问题是：能否把 256-token 图像表示压缩成更少 token（比如 32 个），同时保留足够信息，让 LLM 能 caption、answer questions、reason about image？并且能否不动 frozen backbones，只训练 bridge 参数？

答案是 Q-Former。32 个可学习 “query” vectors 对 ViT 的 patch tokens 做 cross-attention，产出 32-token visual summary 给 LLM 消费。总共 188M 参数。先用 contrastive、matching、generative objectives 训练，再接 LLM。

## The Concept / 概念

### Learnable queries / 可学习 query

Q-Former 的核心技巧：不是让 LLM 的 text tokens attend 到 image patches，而是引入一组新的 32 个可学习 query vectors `Q`，让它们去 attend 图像 patch。这些 query 是模型参数，训练时学习，并且每张图都使用同一组 32 个 query。

Cross-attention 之后，每个 query 持有一份压缩后的图像摘要，例如“主物体是什么”“背景是什么”“物体数量是多少”。这些 query 并不会字面上专门对应某个 semantic label；它们会学习任何能让下游 loss 下降的编码。

### Architecture / 架构

Q-Former 是一个小 transformer（12 层，约 100M 参数），有两条路径：

1. Query path：32 个 query vectors 经过 self-attention（彼此交互），再对 frozen ViT 的 patch tokens 做 cross-attention，然后 FFN。
2. Text path：一个 BERT-like text encoder，与 query path 共享 self-attention 和 FFN 权重。Text path 禁用 cross-attention。

训练时两条路径都会运行。Query 与 text 通过共享 self-attention 交互，这意味着在需要文本条件的任务（ITM、ITG）中，query 可以被 text condition。推理时用于 VLM handoff，只走 query path，得到 32 个 visual tokens。

### Two-stage training / 两阶段训练

BLIP-2 预训练分两阶段：

Stage 1：representation learning（没有 LLM）。三个 losses：

- ITC（image-text contrastive）：类似 CLIP，对 pooled query tokens 与 text CLS token 做 contrastive。
- ITM（image-text matching）：二分类器，判断 image-text pair 是否匹配；带 hard-negative mining。
- ITG（image-grounded text generation）：在 queries 条件下，用 causal LM head 生成 text。迫使 queries 编码可被文本解码的内容。

只有 Q-Former 训练。ViT 是 frozen。没有 LLM 参与。

Stage 2：generative learning。接上一个 frozen LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过小线性层把 32 个 query outputs 投影到 LLM embedding dim。把它们 prepend 到 text prompt。只在 concatenated prompt + image + caption sequence 的 LM loss 上训练 linear projection 和 Q-Former。

Stage 2 之后，Q-Former + projection 就是完整 visual adapter。推理路径是：image → ViT → Q-Former → linear proj → prepend 到 text → frozen LLM 输出。

### Parameter economics / 参数经济性

BLIP-2 使用 ViT-g/14（1.1B，frozen）+ OPT-6.7B（6.7B，frozen）+ Q-Former（188M，trained）= 总 8B，其中训练 188M。Q-Former 只占全栈参数的约 2.4%。训练成本也体现这一点：少量 A100 跑几天，而不是 end-to-end 跑数周。

质量上，BLIP-2 在 zero-shot VQA 上追平或超过 Flamingo-80B，同时小 50 倍。这个 bridge 有效。

### InstructBLIP and the instruction-aware Q-Former / InstructBLIP 与 instruction-aware Q-Former

InstructBLIP（2023）给 Q-Former 增加了一个输入：instruction text 本身。在 cross-attention 时，query 同时能访问 image patches 和 instruction。这样 query 可以针对 instruction specialize（“count the cars”、“describe the mood”），而不是学习单一固定摘要。在 held-out tasks 上带来 benchmark 提升。

### MiniGPT-4 and the projector-only approach / MiniGPT-4 与 projector-only 路线

MiniGPT-4 保留 Q-Former，但只训练 output linear projection，冻结其他所有部分。便宜，但质量代价明显：query 是 BLIP-2 的，不是你的。适合快速迭代，不是最佳架构。

### Why LLaVA went simpler / 为什么 LLaVA 更简单

LLaVA（2023，Lesson 12.05）把 Q-Former 换成普通 2-layer MLP，把每个 ViT patch token 投影到 LLM space：24x24 grid 就是每张图 576 token，全都喂给 LLM。压缩更差，但 LLM 能直接 attend 到 raw patches。当时这很有争议；到 2023 年末它成为主流，因为 visual instruction data（LLaVA-Instruct-150k）证明了 MLP 可以学到足够信号。代价是 LLaVA 的 context 更快填满，但它天然支持 multi-image 和 video。

到 2026 年，领域分成两派：token budget 紧张时 Q-Former 仍然存在（long video、多图）；追求 raw quality per token 时 MLP projector 更常见。

### Gated cross-attention: Flamingo, the ancestor / Gated cross-attention：祖先 Flamingo

Flamingo（Lesson 12.04）早于 BLIP-2，也使用 cross-attention 思路，但不是单个 bridge，而是在 frozen LLM 的每一层插入 cross-attention。BLIP-2 证明只压缩到输入层也能工作。Gemini 和 Idefics 把两者结合起来：interleaved input tokens 加可选 gated cross-attention，用于 in-context few-shot。

### The 2026 descendants / 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及许多因 token budget 受限的视频语言模型。
- Perceiver resampler：Flamingo 的变体（Lesson 12.04）；Idefics family、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四者都合理。决定因素是你被 token budget 限制，还是被 quality-per-token 限制。

## Build It / 动手构建

本课构建一个 toy Q-Former bridge：固定一组 learnable queries，让它们通过 cross-attention 从 image patch tokens 中抽取信息，再投影成 LLM-ready visual tokens。这个实现不会训练大模型，但会把 Q/K/V 来源、attention weight shape、query bottleneck 和最终投影全部摊开。

## Use It / 应用它

`code/main.py` 构建一个 stdlib Q-Former-style cross-attention：

1. 模拟 256 个 image patch tokens（dim 128）。
2. 实例化 32 个 learnable queries（dim 128）。
3. 运行 scaled-dot-product cross-attention（Q 来自 queries，K/V 来自 patches）。
4. 通过线性层投影到 LLM-dim（512）。
5. 输出 32 个 LLM-ready visual tokens。

所有数学都用 pure Python（向量上的 nested loops）。它是 toy，但 shape 正确。代码会打印 attention-weight matrix，让你看到每个 query 从哪些 patch 拉取信息。

## Ship It / 交付它

本课产出 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（vision encoder token count、LLM context budget、deployment constraints、quality target），它会在 Q-Former、MLP、Perceiver resampler 之间推荐，并为每种 bridge 给出简短理由和参数量估算。

## Exercises / 练习

1. 用 PyTorch 实现 cross-attention block。验证在 32 queries、256 keys/values 时，attention-weight matrix 是 32 x 256，且 softmax 后每行和为 1。

2. BLIP-2 stage 1 同时运行 ITC、ITM、ITG 三种 loss。为每个 loss 写出 pseudo-code 形式的 forward signature。哪个 loss 需要激活 text encoder path？

3. 对比参数量：Q-Former（12 layers, 768 hidden）vs 2-layer MLP projector（1408 → 4096，两层）。在什么 LLM scale 下，188M Q-Former 的成本能换回训练效率？

4. 阅读 BLIP-2 论文 Section 3.2（arXiv:2301.12597），了解 Q-Former 如何初始化。解释为什么从 BERT-base 初始化（而不是随机初始化）会加速收敛。

5. 对一个 10 分钟视频，以 1 FPS 采样成 60 帧，计算每帧 token 成本：Q-Former → 32 tokens/frame vs MLP projector → 576 tokens/frame。哪个能放进 128k-token LLM context window？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Q-Former | “Querying transformer” | 带 32 个可学习 query vectors 的小 transformer，对 frozen ViT features 做 cross-attention |
| Learnable queries | “Soft prompt for vision” | 一组固定参数，作为 cross-attention 的 query 端；每个模型学习一次，对所有输入共享 |
| Cross-attention | “Q from here, K/V from there” | query、key、value 来自不同来源的 attention；query 借此从 ViT patches 拉取信息 |
| ITC | “Image-text contrastive” | 应用于 Q-Former pooled queries 与 text CLS 的 CLIP-style loss |
| ITM | “Image-text matching” | 对 hard-negative-mined pairs 做二分类；迫使 query 区分细粒度 mismatch |
| ITG | “Image-grounded text generation” | 在 queries 条件下生成 text 的 causal LM loss；迫使 query 编码可被文本解码的内容 |
| Two-stage pretraining | “Representation then generative” | Stage 1 单独训练 Q-Former（ITC/ITM/ITG）；Stage 2 接 frozen LLM，只训练 projection + Q-Former |
| Frozen backbone | “Do not finetune” | vision encoder 和 LLM 权重固定；只训练 bridge |
| Projection head | “Linear to LLM dim” | 把 Q-Former output 映射到 LLM embedding dimension 的最终线性层 |
| Perceiver resampler | “Flamingo's version” | 类似的 learnable-query cross-attention；Flamingo 在每层使用，而不是单个 bridge |

## Further Reading / 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 使用 ITC/ITM/ITG 三件套的前作。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — “align before fuse”，stage 1 training 的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — instruction-aware Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — projector-only approach。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — learnable-query cross-attention 的通用架构。
