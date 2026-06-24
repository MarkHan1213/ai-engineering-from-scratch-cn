# Vision Transformers (ViT) / 视觉 Transformer（ViT）

> 图像是 patches 的网格。句子是 tokens 的网格。同一个 transformer 都能吃下去。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 ViT 如何把 image patches 转成 token sequence 并送入标准 transformer encoder
- 实现 patchify、linear embedding、`[CLS]` token 和 positional embeddings 的核心流程
- 比较 ViT、DeiT、Swin、DINOv2、SAM 等视觉 transformer 变体的关键变化
- 根据 dataset size、resolution 和 compute budget 选择 patch size 与 ViT variant

## The Problem / 问题

2020 年以前，computer vision 基本等于 convolutions。ImageNet、COCO 和 detection benchmarks 上的每个 SOTA 都使用 CNN backbone。Transformers 属于语言。

Dosovitskiy et al.（2020）的 "An Image is Worth 16x16 Words" 证明可以完全丢掉 convolutions。把 image 切成固定大小 patches，把每个 patch 线性投影成 embedding，再把这个 sequence 送入 vanilla transformer encoder。在足够 scale 下（ImageNet-21k pretraining 或更大），ViT 能匹配或超过 ResNet-based models。

ViT 开启了 2026 年一个更广泛的模式：one architecture, many modalities。Whisper tokenize audio。ViT tokenize images。Robotics 用 action tokens。Video 用 pixel tokens。Transformer 不在乎输入是什么，只要喂给它 sequence，它就会学。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了大多数 vision 任务。CNNs 在 edge devices 和 latency-sensitive tasks 上仍然赢。其他地方的 stack 里几乎都有一个 ViT。

## The Concept / 概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### Step 1 — patchify / 第 1 步：patchify

把 `H × W × C` image 切成 `N × (P·P·C)` 的 flat patches sequence。典型配置：`224 × 224` image，`16 × 16` patches → 196 个 patches，每个有 768 个 values。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch size 是主要 lever。更小 patches = 更多 tokens、更好 resolution、quadratic attention cost。更大 patches = 更粗、更便宜。

### Step 2 — linear embedding / 第 2 步：linear embedding

单个 learned matrix 把每个 flat patch 投影到 `d_model`。这等价于 kernel size `P`、stride `P` 的 convolution。在 PyTorch 中它字面上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`，两行即可实现。

### Step 3 — prepend `[CLS]` token, add positional embeddings / 第 3 步：prepend `[CLS]` token 并添加 positional embeddings

- Prepend 一个 learnable `[CLS]` token。它最终的 hidden state 是 classification 使用的 image representation。
- 添加 learnable positional embeddings（ViT-original）或 sinusoidal 2D（后续变体）。
- 2024+ 的模型有时把 RoPE 扩展到 2D position，甚至不再使用 explicit embeddings。

### Step 4 — standard transformer encoder / 第 4 步：标准 transformer encoder

堆 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` blocks。与 BERT 完全一致。没有 vision-specific layers。这正是那篇论文的教学 punchline。

### Step 5 — head / 第 5 步：head

Classification：取 `[CLS]` hidden state → linear → softmax。DINOv2 或 SAM 则丢弃 `[CLS]`，直接使用 patch embeddings。

### Variants that mattered / 关键变体

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | The original. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trainable on ImageNet-1k only. |
| Swin | 2021 | Hierarchical with shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (no labels). Best general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws apply. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Why it took a while / 为什么它花了一段时间才起飞

ViT 需要*大量*数据才能匹配 CNNs，因为它没有 CNN 的 inductive biases（translation invariance、locality）。如果没有 >100M labeled images 或强 self-supervised pretraining，在 matched compute 下 CNNs 仍然赢。DeiT 在 2021 年用 distillation tricks 修复了这个问题；DINOv2 在 2023 年用 self-supervision 永久修复。

## Build It / 动手构建

见 `code/main.py`。Pure-stdlib patchify + linear embedding + sanity checks。不做 training，因为任何现实规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### Step 1: fake image / 第 1 步：fake image

把 24 × 24 RGB image 表示成 `(R, G, B)` tuples 的 rows list。我们使用 6×6 patches → 16 patches，每个 patch 是 108-d embedding vector。

### Step 2: patchify / 第 2 步：patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order：按 row-major 顺序遍历 grid。每个 ViT 都使用这种 ordering。

### Step 3: linear embed / 第 3 步：linear embed

把每个 flat patch 乘以 random `(patch_flat_size, d_model)` matrix。Prepend `[CLS]` 后验证 output shape 是 `(N_patches + 1, d_model)`。

### Step 4: count parameters for a realistic ViT / 第 4 步：计算真实 ViT 的参数量

打印 ViT-Base 的 param count：12 layers、12 heads、d=768、patch=16。与 ResNet-50（约 25M）比较。ViT-Base 约 86M。ViT-Large 约 307M。ViT-Huge 约 632M。

## Use It / 应用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings are the 2026 default for image features / DINOv2 embeddings 是 2026 年 image features 默认选择。** 冻结 backbone，训练 tiny head。Classification、retrieval、detection、captioning 都能用。Meta 的 DINOv2 checkpoints 在每个 non-text vision task 上都超过 CLIP。

**Patch-size picking / Patch size 选择。** Small models 使用 16×16（ViT-B/16）。Dense prediction（segmentation）使用 8×8 或 14×14（SAM、DINOv2）。Very large models 使用 14×14。

## Ship It / 交付它

见 `outputs/skill-vit-configurator.md`。这个 skill 会根据 dataset size、resolution 和 compute budget，为新 vision task 选择 ViT variant 与 patch size。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。验证 patches 数量等于 `(H/P) * (W/P)`，flat patch dimension 等于 `P*P*C`。
2. **Medium / 中等。** 实现 2D sinusoidal positional embeddings：分别为每个 patch 的 `row` 和 `col` 编码两个 independent sinusoidal codes，然后 concatenate。把它们喂给 tiny PyTorch ViT，并在 CIFAR-10 上与 learnable positional embeddings 比较 accuracy。
3. **Hard / 困难。** 构建一个 3-layer ViT（PyTorch），用 4×4 patches 在 1,000 张 MNIST images 上训练。测量 test accuracy。然后在相同 1,000 images 上加入 DINOv2 pretraining（简化版：训练 encoder 从 masked patches 预测 patch embeddings）。Accuracy 是否提升？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | “The vision-transformer token” | Image 中一个 `P × P × C` 区域的 pixel values flat vector。 |
| Patchify | “Chop + flatten” | 把 image 切成不重叠 patches，并把每个 patch flatten 成 vector。 |
| `[CLS]` token | “The image summary” | Prepend 的 learnable token；其最终 embedding 是 image representation。 |
| Inductive bias | “What the model assumes” | ViT 的 priors 比 CNNs 少；需要更多数据弥补差距。 |
| DINOv2 | “Self-supervised ViT” | 不用 labels，通过 image augmentation + momentum teacher 训练。2026 年最好的 general image features。 |
| SigLIP | “CLIP's successor” | ViT + text encoder，使用 sigmoid contrastive loss 训练；matched compute 下优于 CLIP。 |
| Swin | “Windowed ViT” | 带 local attention + shifted windows 的 hierarchical ViT；sub-quadratic。 |
| Register tokens | “2023 trick” | 少量额外 learnable tokens，用来吸收 attention sinks；提升 DINOv2 features。 |

## Further Reading / 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token fix。
