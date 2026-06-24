# Any-Resolution Vision: Patch-n'-Pack and NaFlex / 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前 VLM 的答案是把所有东西 resize 到固定正方形，这会丢掉 OCR、document understanding 和高分辨率 scene parsing 最需要的信号。NaViT（Google, 2023）证明可以用 block-diagonal masking 把 variable-resolution patches 打包进一个 transformer batch。Qwen2-VL 的 M-RoPE（2024）直接抛弃 absolute positional tables。LLaVA-NeXT 的 AnyRes 把高分辨率图像切成 base + sub-images。SigLIP 2 的 NaFlex 变体（2025）已经成为 open VLM 支持各种 aspect ratio 的默认 encoder。本课从头实现 patch-n'-pack。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites / 前置知识：** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 把一个 batch 中 variable-resolution images 的 patches 打包成单个序列，并构建 block-diagonal attention mask。
- 面向给定任务，在 AnyRes tiling（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间选择。
- 在不 resize 的情况下，为 OCR、charts 和 photography 计算 token budgets。
- 说出 square-resize 的三个 failure modes：文字被拉伸、内容被裁剪、padding 上浪费 token。

## The Problem / 问题

Transformer 需要序列。Batch 是一组长度相同的序列。如果图像都是 224x224，每次都是 196 个 patch tokens，不需要 padding，训练和推理都很简单。

现实不配合。文档是竖版（8.5x11 英寸，接近 2:3）。图表截图是横版（16:9）。收据又高又窄（1:3）。医学图像常见 2048x2048 或更大。移动设备截图是 1170x2532（0.46:1）。

2024 年前有三种选择，每种都失败：

1. Resize 到固定正方形（224x224 或 336x336）。拉伸会扭曲文字和人脸。下采样会毁掉 chart labels 和 OCR 内容。LLaVA-1.5 之前这是标准做法。
2. Crop 到固定 aspect ratio。会扔掉大部分图像，而且怎么选 crop 位置本身就是一个视觉问题。
3. Pad 到最长边。解决扭曲，但 portrait 图像上 50%+ token 都浪费在 padding，attention 仍要为 pad token 付平方成本。

2024-2025 年的答案是：让 transformer 按图像 native resolution 吃 patch，然后找到一种方式，把 heterogeneous batch 打成一个序列且不浪费计算。

## The Concept / 概念

### NaViT and patch-n'-pack / NaViT 与 patch-n'-pack

NaViT（Dehghani et al., 2023）是证明这个方案可以 scale 的论文。思路很机械：

1. 对 batch 中每张图，以选定 patch size（比如 14）计算 native patch grid。
2. 把每张图的 patches flatten 成自己的 variable-length sequence。
3. 把所有图像的 patches concatenated 成 batch 的一个长序列。
4. 构建 block-diagonal attention mask，让 image A 的 patches 只 attend 到 image A 内部。
5. 携带 per-patch position 信息（2D RoPE 或 fractional position embeddings）。

三张图分别是 336x336（576 tokens）、224x224（256 tokens）、448x336（768 tokens），会变成一个 1600-token sequence 和一个 1600x1600 的 block-diagonal mask。没有 padding，没有浪费计算。Transformer 处理任意 aspect ratio。

NaViT 还引入了 fractional patch dropping：训练时在 batch 内随机 drop 50% patches，既正则化又加速训练。SigLIP 2 继承了这个思想。

### AnyRes (LLaVA-NeXT) / AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实替代方案。给定高分辨率图像和一个固定 encoder（CLIP 或 SigLIP @ 336），先 tile 图像：

1. 从预定义集合中选一个 grid layout，例如 (1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2)，尽量匹配 aspect ratio。
2. 把整图切进这个 grid；每个 tile 成为一个 336x336 crop。
3. 再生成一个 thumbnail：整图 resize 到 336x336，作为 global-context token。
4. 每个 tile 通过 frozen 336-encoder。把 tile tokens + thumbnail tokens concatenate。

672x672 图像用 2x2 grid 加 thumbnail：4 * 576 + 576 = 2880 visual tokens。昂贵但有效；LLM 同时看到局部细节和全局上下文。

当 encoder frozen 且只支持一个 resolution 时，AnyRes 是首选。它会让大图 token count 爆炸（1344x1344 图像在 4x4 grid 下是 9216 + 576 ≈ 9800 tokens，几乎填满 8k LLM context）。

### M-RoPE (Qwen2-VL) / M-RoPE（Qwen2-VL）

Qwen2-VL 引入 Multimodal Rotary Position Embedding。不是 NaViT 的 fractional positions，也不是 AnyRes 的 tile-and-thumbnail；每个 patch 携带 3D position（temporal, height, width）。Query/key rotations 处理任意 H、W 和 temporal length。

M-RoPE 带来无需重训的 native dynamic resolution。推理时输入任意 HxW 图像，patch embedder 产出 H/14 x W/14 tokens，每个 token 有 `(t=0, r=row, c=col)` position，RoPE 按正确频率旋转 attention，完成。Qwen2.5-VL 和 Qwen3-VL 延续这条路线。InternVL3 的 V2PE 是同一思想在可变模态编码上的版本。

不同于 AnyRes，M-RoPE 在 native resolution 下 token 数量是 O(H x W / P^2)，没有乘法式 tile overhead。不同于 NaViT，它仍然期望每次 forward 一张图；跨 resolution batching 仍然需要在上层做 patch-n'-pack。

### NaFlex (SigLIP 2) / NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的 native-flex 模式。单个模型在 inference 支持多种 sequence length（256、729、1024 tokens）。内部训练使用 NaViT-style patch-n'-pack，并为每个 patch 使用 absolute fractional positions。卖点是：一个 checkpoint，根据任务在 inference 选择 token budget。

Semantic task（classification、retrieval）用 256 tokens。OCR 或 chart understanding 用 1024 tokens。不需要重训。

### The packing mask / 打包 mask

Block-diagonal mask 是很多实现翻车的地方。对一个总长度为 `N_total` 的 packed sequence，覆盖 `i=0..B-1` 张图，长度分别为 `n_i`，形状 `(N_total, N_total)` 的 mask `M` 在两个 index 位于同一图像 block 时为 1，否则为 0。可以从 cumulative length list 构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

PyTorch 中可以用 `torch.block_diag` 或显式 gather 一行完成。FlashAttention 的 variable-length path（`cu_seqlens`）完全跳过 dense mask，直接用 cumulative-length tensor 在各 sequence 内 attend，典型 batch 下比 dense mask 快约 10x。

### Token budgets / Token 预算

按任务选择策略：

- OCR / documents：1024-4096 tokens。SigLIP 2 NaFlex @ 1024，或 AnyRes 3x3 + thumbnail。
- Charts and UI：384-448 native 下 729-1024 tokens。Qwen2.5-VL dynamic resolution 加 max pixels cap。
- Natural photos：256-576 tokens 足够。下游 LLM 已经能看到足够信息。内容密度高时才为 token 付费。
- Video：spatial pooling 后每帧 64-128 tokens，2-8 FPS。Lesson 12.17 会覆盖。

2026 年生产规则：按任务设 per-task max-pixels cap，按 native aspect ratio 编码到该 cap，pack batch，并跳过 padding。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是这个旋钮。

## Build It / 动手构建

本课动手实现 patch-n'-pack：从一组 `(H, W)` 计算每张图的 patch sequence length，把它们串成一个 packed sequence，并生成 block-diagonal attention mask。你会同时比较 fixed-square resize、AnyRes tiling 和 native packing 的 token 成本。

## Use It / 应用它

`code/main.py` 为一批 heterogeneous image sizes 实现 patch-n'-pack，使用整数像素坐标。它会：

- 接收一组 `(H, W)` image sizes。
- 在 patch size 14 下计算每张图的 patch sequence length。
- 打包成总长度为 `sum(n_i)` 的单序列。
- 构建 block-diagonal attention mask（dense，便于理解）。
- 对比 packed cost、square-resize 和 AnyRes tiling。
- 打印混合 batch（receipt、chart、screenshot、photo）的 token budget table。

运行它。输出数字会直接说明为什么 2026 年每个 open VLM 都在用 patch-n'-pack。

## Ship It / 交付它

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个 mixed-aspect-ratio workload（OCR、charts、photos、video frames）和 total-token budget，它会选择合适策略（NaFlex、AnyRes、M-RoPE 或 fixed-square），并输出 per-request configuration。做 VLM 产品 sizing 时使用这个 skill，它能避免悄无声息的 10x token blowup 杀死 latency budget。

## Exercises / 练习

1. 一张收据是 600x1500（1:2.5）。patch size 14 下有多少 native-resolution tokens？resize 到 336 的 square 后有多少？实践中哪个更损失 OCR accuracy？

2. 为长度为 256、576、729、1024 的四张图构建 block-diagonal mask。验证 attention matrix 是 2585x2585，并且非零项正好是 `256^2 + 576^2 + 729^2 + 1024^2`。

3. 对一张 1792x896 图像、patch 14，比较：(a) square-resize 到 336 后编码，(b) AnyRes 2x1 + thumbnail，(c) M-RoPE native。哪个 token 最少？哪个保留最多细节？

4. 实现 fractional patch dropping：给定 packed sequence，均匀随机 drop 50% tokens，并相应更新 block-diagonal mask。测量 mask sparsity 的变化。

5. 阅读 Qwen2-VL 论文 Section 3.2（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么上下界都重要。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Patch-n'-pack | “NaViT-style packing” | 把不同图像的 variable-length patch sequences concatenate 到一个 batch dimension |
| Block-diagonal mask | “Packing mask” | 限制每张图的 patches 只 attend 自己、不 attend packed 邻居的 attention mask |
| AnyRes | “LLaVA-NeXT tiling” | 把高分辨率图像切成固定大小 tile grid，再加 global thumbnail；每个 tile 用固定 encoder |
| NaFlex | “SigLIP 2 native-flex” | 单个 SigLIP 2 checkpoint 在 inference 支持 256/729/1024-token budgets，无需重训 |
| M-RoPE | “Multimodal RoPE” | 3D rotary position encoding（time, row, column），无需 position table 即可处理任意 H、W、T |
| cu_seqlens | “FlashAttention packing” | FlashAttention varlen path 使用的 cumulative-length tensor，替代 dense block-diagonal mask |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL per-request knobs，用于限制很小或很大输入上的 token count |
| Visual token budget | “How many tokens per image” | 每张图大致产出的 patch token 数；决定 LLM prompt budget 与 attention cost |

## Further Reading / 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
