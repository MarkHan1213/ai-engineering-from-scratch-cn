# CLIP and Contrastive Vision-Language Pretraining / CLIP 与对比式视觉语言预训练

> OpenAI 的 CLIP（2021）证明了一个足够大的想法：只用嘈杂的网页图文对和 contrastive loss，把 image encoder 与 text encoder 对齐到同一个向量空间，就能驱动之后五年的 VLM。零监督标签。400M 对数据。得到的 embedding space 可以做 zero-shot classification、image-text retrieval，并作为 2026 年几乎每个 VLM 的 vision tower。SigLIP 2（2025）把 softmax 换成 sigmoid，以更低成本扩展到 CLIP 之后的规模。本课从 InfoNCE 推到 sigmoid pairwise loss，并用 stdlib Python 构建训练步。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites / 前置知识：** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 从 mutual information 推导 InfoNCE loss，并实现数值稳定的 vectorized 版本。
- 解释为什么 sigmoid pairwise loss（SigLIP）可以扩展到 batch 32768+，且不需要 softmax 的 all-gather 开销。
- 通过构造 text templates（`a photo of a {class}`）并对 cosine similarity 取 argmax，运行 zero-shot ImageNet classification。
- 说出 CLIP / SigLIP pretraining 给你的四个杠杆：batch size、temperature、prompt template、data quality。

## The Problem / 问题

CLIP 之前的视觉模型主要靠监督学习。收集带标签数据集（ImageNet：1.2M 图像，1000 类），训练 CNN，然后上线。标签昂贵，标签会偏向标注者能达成共识的概念，而且没有微调就很难迁移到新任务。

网页上的 image-caption 对有十亿级，几乎免费。一张金毛犬的照片，alt text 写着 “my dog Max in the park”，其中已经有监督信号：文本描述了图像。问题是：如何把这种弱信号变成有用训练？

CLIP 的答案是把图文对当成 matching task。给定一个 batch 的 N 张图和 N 条 caption，学习把每张图匹配到自己的 caption，同时区分 N-1 个干扰项。监督信号是“这两者属于一对；另外 N-1 个不属于”。没有 class label，没有人工标注，只有 contrastive loss。

这个 embedding space 做到了训练目标之外的事。ImageNet zero-shot 可行，是因为 “a photo of a cat” 会 embed 到没有显式标注为 cat 的猫图附近。这一赌注催生了 2026 年的每个 VLM。

## The Concept / 概念

### The dual encoder / 双塔 encoder

CLIP 有两个 tower：

- Image encoder `f`：ViT 或 ResNet，每张图输出一个 D-dim 向量。
- Text encoder `g`：小 transformer，每条 caption 输出一个 D-dim 向量。

两个 tower 都把输出归一化到 unit length。因为都是 unit-norm，similarity 就是 `cos(f(x), g(y)) = f(x)^T g(y)`。

对一个 N 个 `(image, caption)` pair 的 batch，构造形状为 `(N, N)` 的 similarity matrix `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习 temperature（CLIP 初始化为 0.07，并在 log-space 学习）。

### InfoNCE loss / InfoNCE 损失

CLIP 对行和列做对称 cross-entropy：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 里的 softmax 会迫使每张图与自己的 caption 的匹配度高于 batch 中所有其他 caption。其他 batch items 都是 “negatives”。更大的 batch = 更多 negatives = 更强信号。CLIP 用 batch 32k 训练；规模很重要。

### Temperature / 温度

`tau` 控制 softmax 的尖锐程度。低 tau → 分布更尖，类似 hard negative mining。高 tau → 分布更软，所有样本都参与。CLIP 学习 log(1/tau)，并裁剪以防 collapse。SigLIP 2 固定初始 tau，改用可学习 bias。

### Why sigmoid scales better (SigLIP) / 为什么 sigmoid 更好扩展（SigLIP）

Softmax 需要全量 similarity matrix 同步。在分布式训练中，每个 replica 都必须 all-gather 每个 embedding，然后做 softmax。这对通信很不友好，随 world size 增长很快。

SigLIP 用 element-wise sigmoid 替代 softmax：对每个 `(i, j)` pair，做一个二分类问题：“这是不是匹配对？” diagonal 是正例，其余都是负例。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当且仅当 `i == j`。每个 pair 的 loss 相互独立。不需要 all-gather。每张 GPU 计算本地 block 后求和即可。SigLIP 2 可以便宜地扩展到 batch 32k-512k，而 CLIP 会需要成比例更多通信。

### Zero-shot classification / 零样本分类

给定 N 个 class names，对每个 class 构造一个 text template：

```
"a photo of a {class}"
```

用 text encoder embed 每个 template。用 image encoder embed 输入图像。cosine similarity 的 argmax 就是预测类别。目标类别上不需要训练。

Prompt templates 很重要。CLIP 原论文为每类用了 80 个 template（plain、artistic、photo、painting 等）并平均 embedding，在 ImageNet 上提升 3 个点。现代用法通常只选一两个 template。

### Linear probes and finetuning / Linear probe 与微调

Zero-shot 是 baseline。Linear probe（在 frozen CLIP features 上为目标类别训练一层 linear layer）在 in-domain 任务上通常超过 zero-shot。Full finetuning 在 in-domain 上进一步提升，但可能损害 zero-shot transfer。三种 regime，对应三种 trade-off。

### SigLIP 2: NaFlex and dense features / SigLIP 2：NaFlex 与 dense features

SigLIP 2（2025）加入：

- NaFlex：单模型处理可变 aspect ratio 和 resolution。
- 更好的 dense features，面向 segmentation、depth estimation，以及作为 VLM frozen backbone。
- Multilingual：训练覆盖 100+ 语言，而 CLIP 基本是 English-only。
- 1B 参数规模，而 CLIP 止步于 400M 左右。

在 2026 年 open VLM 中，SigLIP 2 SO400m/14 是默认 vision tower。CLIP 仍然是纯 image-text retrieval 的默认选择，尤其当 LAION-2B 训练分布刚好匹配你的 query pattern 时。

### ALIGN, BASIC, OpenCLIP, EVA-CLIP / 同族模型

ALIGN（Google, 2021）：和 CLIP 同思路，1.8B 对数据，90% 嘈杂。证明 noisy data 可以 scale。OpenCLIP（LAION）：在 LAION-400M / 2B 上的 open reproduction，多种规模，是常用 open checkpoint。EVA-CLIP：从 masked image modeling 初始化，是强 VLM backbone。BASIC：Google 的 CLIP+ALIGN 混合路线。它们属于同一家族，差别在数据和调参。

### The zero-shot ceiling / Zero-shot 上限

CLIP 类模型在 ImageNet zero-shot 上大约封顶在 76%（CLIP-G、OpenCLIP-G）。再往上需要更大数据（SigLIP 2 到 80%+）或架构变化（监督 head、更大参数）。这个 benchmark 正在饱和；真正价值是下游 VLM 消费的 embedding space。

```figure
multimodal-fusion
```

## Build It / 动手构建

本课动手实现的核心是一个最小 dual-encoder trainer：先构造 image/text embeddings，再计算 similarity matrix，分别实现 InfoNCE 与 sigmoid pairwise loss，最后把同一套 embedding 用到 zero-shot classification。目标不是追求真实模型精度，而是让每个 loss 的形状、数值稳定技巧和 batch negative 机制都可见。

## Use It / 应用它

`code/main.py` 实现：

1. 一个 toy dual encoder（hash-based image features、text char features），让你在不用 numpy 的情况下看清 InfoNCE 的形状。
2. Pure Python 的 InfoNCE loss（通过 log-sum-exp 保证 numerical stability）。
3. Sigmoid pairwise loss，用于对比。
4. 一个 zero-shot classification routine：计算图像与一组 text prompts 的 cosine similarity，取 argmax 作为预测。

运行它并观察 loss curve。绝对数值是 toy，但形状和真实 CLIP trainer 的输出一致。

## Ship It / 交付它

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组 images（path）和目标 classes 列表，它用 CLIP template 构造 text prompts，用指定 checkpoint（例如 `openai/clip-vit-large-patch14`）embed 两侧，并返回 top-1 / top-5 predictions 与 similarity scores。该 skill 会拒绝对 prompt list 外的类别做声明。

## Exercises / 练习

1. 手算一个 batch=4 的 InfoNCE。构造 4x4 similarity matrix，跑 softmax，取 diagonal，计算 cross-entropy。用你的 Python 实现验证手算结果。

2. SigLIP 除 temperature 外还使用 bias 参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 中 class imbalance 很大（每行负例远多于正例）时，`b` 起什么作用？阅读 SigLIP Section 3（arXiv:2303.15343）。

3. 构建一个 cats vs dogs 的 zero-shot classifier。尝试两个 prompt template：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图上测量 accuracy。template ensemble 是否优于单一 template？

4. 计算 512-GPU、batch 32k 训练中 softmax InfoNCE 与 sigmoid pairwise 的通信成本。哪个按 O(N) scale，哪个按 O(N^2) scale？引用 SigLIP Section 4。

5. 阅读 OpenCLIP scaling-laws 论文（arXiv:2212.07143, Cherti et al.）。从图中复现它对 data scaling 的结论：在固定 model size 下，ImageNet zero-shot accuracy 与 training data size 的 log-linear 关系是什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| InfoNCE | “Contrastive loss” | 对 batch similarity matrix 做 cross-entropy；每个 item 的正例是配对 item，其他都是负例 |
| Sigmoid loss | “SigLIP loss” | Per-pair binary cross-entropy；没有 softmax、没有 all-gather，分布式训练更便宜 |
| Temperature | “tau” | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布尖锐程度 |
| Zero-shot | “no-finetune classification” | 用 text prompts 构造 class embeddings，并按 cosine similarity 分类；目标类别上不训练 |
| Prompt template | “a photo of a ...” | 包在 class name 外的文本脚手架；可影响 zero-shot accuracy 1-5 个点 |
| Dual encoder | “Two-tower” | 一个 image encoder + 一个 text encoder，输出到共享 D-dim 空间 |
| Hard negative | “Tough distractor” | 与正例足够相似的负例，迫使模型学习更细粒度区分 |
| Linear probe | “Frozen + one layer” | 只在 frozen features 上训练 linear classifier；用于衡量 feature quality |
| NaFlex | “Native flexible resolution” | SigLIP 2 能力：无需 resize 就能 ingest 任意 aspect ratio 和 resolution |
| Temperature scaling | “log-parametrized tau” | CLIP 参数化 `log(1/tau)` 以改善梯度，并裁剪以防 tau collapse 到接近 0 |

## Further Reading / 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — multilingual + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用 noisy web data 做规模化。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws。
