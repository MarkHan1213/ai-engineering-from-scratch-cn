# Vision Transformers and the Patch-Token Primitive / Vision Transformer 与 Patch Token 原语

> 在进入任何多模态模型之前，图像都必须先变成 transformer 能处理的 token 序列。2020 年的 ViT 论文用 16x16 像素 patch、线性投影和位置 embedding 给出了答案。五年后，2026 年的前沿模型（Claude Opus 4.7 的 2576px native、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这里开始：编码器从 ViT 演进到 DINOv2、SigLIP 2，加入了 register token，位置方案换成 2D-RoPE，但这个原语没有变。本课从头读完 patch-token pipeline，并用 stdlib Python 构建它，让 Phase 12 后续讨论“visual tokens”时有一个具体的心智模型。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites / 前置知识：** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 将一个 HxWx3 图像转换成 patch token 序列，并保留正确的位置编码。
- 给定 patch size、resolution、hidden dim 和 depth，计算 ViT 的 sequence length、参数量和 FLOPs。
- 说出让 ViT 从 2020 年研究原型走向 2026 年生产系统的三个升级：self-supervised pretraining（DINO / MAE）、register tokens 和 native-resolution packing。
- 为下游任务在 CLS pooling、mean pooling 和 register tokens 之间做取舍。

## The Problem / 问题

Transformer 处理的是向量序列。文本天然就是序列（bytes 或 tokens）。图像则是带三个颜色通道的二维像素网格，不是序列。如果把每个像素都摊平，一个 224x224 RGB 图像会变成 150,528 个 token；在这个长度上做 self-attention 基本不可行，因为代价随序列长度平方增长。

2020 年之前的主流做法是在前面接一个 CNN feature extractor：ResNet 产出 7x7 的 feature map，每个位置是 2048-dim 向量，再把这 49 个 token 送进 transformer。这个方案可用，但继承了 CNN 的 inductive bias（translation equivariance、local receptive fields），也削弱了 transformer 对规模的胃口。

Dosovitskiy 等人（2020）问了一个很直白的问题：如果跳过 CNN 呢？把图像切成固定大小的 patch（例如 16x16 像素），把每个 patch 线性投影成向量，加上 positional embedding，再送进普通 transformer。那时这很异端：没有卷积的视觉模型。只要数据足够多（JFT-300M，后来 LAION），它就在 ImageNet 上超过 ResNet，并且持续变强。

到 2026 年，ViT primitive 已经是没有争议的基础。每个 open-weights VLM 的 vision tower 都是它的某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“要不要用 patch”，而是“patch size 选多少、resolution schedule 怎么设、pretraining objective 用什么、position encoding 怎么做”。

## The Concept / 概念

### Patches as tokens / 作为 token 的 patch

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，先把图像切成 `(H/P) x (W/P)` 个不重叠 patch。每个 patch 是一个 `P x P x 3` 的像素块。把这个像素块摊平成 `3 P^2` 维向量，再应用共享线性投影 `W_E`，其形状为 `(3 P^2, D)`，把每个 patch 映射到模型 hidden dimension `D`。

对于标准 ViT-B/16 配置：

- Resolution 224，patch size 16 → grid 14x14 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 加一个可学习的 `[CLS]` token → sequence length 197。

Patch projection 在数学上等价于一个 2D convolution：kernel size `P`，stride `P`，输出通道数 `D`。生产代码也确实这么实现：`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“linear projection”是概念表述；kernel 表述更高效。

### Positional embeddings / 位置 embedding

Patch 本身没有内在顺序，transformer 看到的是一袋向量。早期 ViT 加了可学习的一维 positional embedding（每个位置一个 768-dim 向量，共 197 个）。这能工作，但把模型绑定在训练 resolution 上：推理时如果 grid 变了，就必须插值 position table。

现代视觉 backbone 使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或 factorized 2D positions。2D-RoPE 会根据 patch 的 `(row, column)` 索引旋转 query 和 key 向量，让模型从旋转角中推断相对二维位置。没有 position table，推理时就能处理任意 grid size。

### CLS token, pooled output, and register tokens / CLS token、pooling 输出与 register tokens

图像级 representation 从哪里来？现在三种选择并存：

1. `[CLS]` token。把一个可学习向量 prepend 到 patch 序列前。所有 transformer block 之后，CLS token 的 hidden state 就代表整张图。这继承自 BERT，原始 ViT 和 CLIP 都使用它。
2. Mean pool。对 patch token 的输出 hidden state 取平均。SigLIP、DINOv2 和大多数现代 VLM 都用这个。
3. Register tokens。Darcet 等人（2023）观察到，没有显式 sink token 的 ViT 会发展出高范数的“artifact” patch 来劫持 self-attention。加入 4-16 个可学习 register tokens 可以吸收这部分负载，提升 dense prediction 质量（segmentation、depth）。DINOv2 和 SigLIP 2 都带 registers。

这个选择会影响下游任务。CLS 用于 classification 很合适。对于把 patch token 喂给 LLM 的 VLM，通常完全跳过 pooling，每个 patch 都变成 LLM input token。Registers 在交给 LLM 前会被丢弃：它们是脚手架，不是内容。

### Pretraining: supervised, contrastive, masked, self-distilled / 预训练：监督、对比、遮蔽与自蒸馏

2020 年的 ViT 在 JFT-300M 上用 supervised classification 预训练。很快它被这些路线取代：

- CLIP (2021)：在 400M 图文对上做 contrastive image-text。见 Lesson 12.02。
- MAE (2021, He et al.)：遮蔽 75% patch，重建像素。自监督，只需要纯图像。
- DINO (2021) / DINOv2 (2023)：student-teacher self-distillation，无标签、无 caption。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉 backbone，也是 dense features 场景的默认选择。
- SigLIP / SigLIP 2 (2023, 2025)：用 sigmoid loss 的 CLIP，并引入 NaFlex 支持 native aspect ratio。它是 2026 年 open VLM 的主流 vision tower（Qwen、Idefics2、LLaVA-OneVision）。

选择哪种 pretraining 决定了 backbone 擅长什么：CLIP/SigLIP 适合与文本做 semantic matching，DINOv2 适合 dense visual features，MAE 适合作为下游微调起点。

### Scaling laws / 缩放规律

ViT scaling（Zhai et al. 2022）说明了 ViT 的质量会随 model size、data size 和 compute 呈现可预测规律。在固定 compute 下：

- 更大的模型 + 更多数据 → 更好质量。
- Patch size 是 sequence length 与 fidelity 的杠杆。Patch 14（DINOv2/SigLIP SO400m 常用）比 patch 16 每张图产出更多 token；对 OCR 和 dense tasks 更好，但更慢。
- Resolution 是另一个大杠杆。从 224 到 384 到 512 几乎总是有帮助，但 FLOPs 以平方成本增加。

ViT-g/14（1B 参数，patch 14，resolution 224 → 256 tokens）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年 open VLM 的两个主力 encoder。

### Parameter count for a ViT / ViT 参数量

完整计算在 `code/main.py`。以 224 分辨率的 ViT-B/16 为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

加载 checkpoint 前，先用这种方式粗算每个 ViT。Backbone 大小会决定任何下游 VLM 的 VRAM 下限。

### 2026 production config / 2026 生产配置

2026 年大多数 open VLM 搭载的 encoder 是 native resolution（NaFlex）下的 SigLIP 2 SO400m/14。它有：

- 400M 参数。
- Patch size 14，默认 resolution 384 → 每张图 729 个 patch token。
- 图像级任务用 mean pool；VQA 时全部 729 个 patch 都流入 LLM。
- 4 个 register tokens，在交给 LLM 前丢弃。
- 2D-RoPE，并带 image-level scaling 以支持 native aspect ratio。

这套配置里的每个决定，都能追溯到一篇可读的论文。

```figure
image-patch-tokens
```

## Build It / 动手构建

本课的动手构建不是训练完整 ViT，而是把 patch-token pipeline 拆成可检查的几何与参数计算：切 patch、摊平、投影、加位置、估算 sequence length、参数量和 FLOPs。先把这些量算准，后面选择 vision tower、resolution 和 token budget 才不会靠猜。

## Use It / 应用它

`code/main.py` 是一个 patch tokenizer 和 geometry calculator。它接收 image H、W、patch P、hidden D、depth L，并报告：

- Patching 后的 grid shape 和 sequence length。
- 一个合成 8x8 像素 toy image 的 token sequence，用来走通 flatten + project 路径。
- 按 patch embed、position embed、transformer blocks 和 head 拆解的参数量。
- 目标 resolution 下一次 forward pass 的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。把参数量和公开数值对齐。调整 patch size 和 resolution，直观感受 token-count 的成本。

## Ship It / 交付它

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT config（patch size、resolution、hidden dim、depth），它会产出 token-count、parameter-count 和 VRAM estimate，并附上理由。每次为 VLM 选择 vision backbone 时都用这个 skill，它能避免“视觉 token 爆炸，把 LLM context 塞满”的意外。

## Exercises / 练习

1. 计算 Qwen2.5-VL 在 native 1280x720 输入、patch size 14 下的 patch-token sequence length。它和 CLS-only representation 相比如何？

2. 一个 1080p frame（1920x1080）在 patch 14 下会产生多少 token？30 FPS 的 5 分钟视频总共多少 visual tokens？pooling、frame sampling、token merging 中哪个最省成本？

3. 用 pure Python 实现 patch token 的 mean pooling。验证对 DINOv2 输出的 196 个 token 做 mean-pool，能匹配模型 `forward` 返回的 pooled embedding。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）的 Section 3。用两句话描述 register 吸收了什么 artifact，以及它为什么影响下游 dense prediction。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同 resolution 的图像，生成单个 packed sequence 和 block-diagonal attention mask。等学到 Lesson 12.06 时再对照验证。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Patch | “16x16 pixel square” | 输入图像中固定大小、互不重叠的区域；会变成一个 token |
| Patch embedding | “Linear projection” | 共享的可学习矩阵（或 stride=P 的 Conv2d），把摊平 patch 像素映射到 D-dim 向量 |
| CLS token | “Class token” | prepend 的可学习向量，其最终 hidden state 表示整张图；2026 年已是可选项 |
| Register token | “Sink token” | 额外可学习 token，用来吸收 ViT 在预训练中形成的高范数 attention artifact |
| Position embedding | “Positional info” | 让序列具备顺序感的 per-position 向量或旋转；2D-RoPE 是现代默认 |
| Grid | “Patch grid” | 给定 resolution 和 patch size 后得到的 `(H/P) x (W/P)` 二维 patch 数组 |
| NaFlex | “Native flexible resolution” | SigLIP 2 特性：单个模型无需重训即可服务多种 aspect ratio 和 resolution |
| Backbone | “Vision tower” | 预训练图像 encoder，其 patch-token 输出会流入 VLM 中的 LLM |
| Pooling | “Image-level summary” | 把 patch token 变成一个向量的策略：CLS、mean、attention pool 或 register-based |
| Patch 14 vs 16 | “Finer vs coarser grid” | Patch 14 每张图产生更多 token，OCR fidelity 更好但更慢；patch 16 是经典默认 |

## Further Reading / 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模 self-distillation，无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens 与 artifact 分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认 vision tower。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验 scaling laws。
