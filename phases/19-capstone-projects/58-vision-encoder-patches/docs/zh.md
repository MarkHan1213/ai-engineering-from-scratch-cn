# Vision Encoder Patches / 视觉编码器 Patch

> 读取像素的视觉模型也需要一个“像素 tokenizer”。Patch embedding 就是这个 tokenizer：把图像切成方格，展平每个方格，用一层 linear layer 投影，再加上 2D position signal，让 transformer 知道每个方格原本在图像中的位置。

**类型：** 构建
**语言：** Python
**前置知识：** 第 19 阶段第 30-37 课（Track B 基础）
**时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 把图像 token 化为固定长度的 patch embedding 序列。
- 实现一个基于 `Conv2d` 的 patch projection，并让它与 unfold-then-linear 的数学形式一致。
- 构建 deterministic 2D sinusoidal position embedding，让 token order 编码空间位置。
- 在 synthetic fixture 上验证 patch count、embedding shape，以及 `Conv2d`/unfold equivalence。

## The Problem / 问题

transformer 吃的是 vector sequence。图像是 3-channel grid。把每个 pixel 当成一个 token 会让 sequence length 爆炸：一张 224x224 RGB image 有 150,528 个 tokens，12-layer transformer 在 attention 上负担不起。把整张图展平成一个巨大 vector 又会丢掉 locality，attention layer 之后也很难恢复。encoder front end 的任务，是把 pixel grid 压缩成几百个 tokens，每个 token 概括一个方形区域。

Patch embedding 用一层 linear projection 解决这个问题。224x224 image 切成 16x16 patches 后得到 14x14 grid，也就是 196 个 patches。每个 patch 从 `(3, 16, 16) = 768` 个 pixel values 展平成一个 vector，再由 linear layer 映射到模型 hidden dimension。transformer 看到的是 196 个维度为 `hidden`（常见是 768）的 tokens，再加一个 CLS token。这是后续网络可以处理的序列长度。

## The Concept / 概念

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### Why patches, not pixels / 为什么是 patches 而不是 pixels

attention 的成本与 sequence length 二次相关。196-token sequence 每个 head、每层需要 `196 * 196 = 38,416` 个 attention scores；150,528-token sequence 需要 `150,528 * 150,528 = 22.6 billion`。patches 带来约 590,000x 的 attention compute 降幅，而一个 16x16 区域已经足够承载 high-level vision tasks 的信号。代价是一个 patch 内的细粒度 spatial detail 会损失；这也是为什么在需要精细定位时，下游 multimodal stacks 往往会再跑一条 high-resolution branch。

### Why a linear projection is enough / 为什么一层 linear projection 足够

每个 patch 被视为独立 vector。projection 学到一组 basis：edge detectors、color filters、simple textures。一层 linear layer 很小（ViT-Base 下 `768 * 768 = 589,824` parameters），训练也快。更深的 convolutional stems 当然存在，也就是 “hybrid” ViT，但 flat linear projection 是标准做法，大多数现代 open-weight encoders 都采用这个形状。

### The `Conv2d` trick / `Conv2d` 技巧

`Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` 且 no padding，与 unfold-then-linear 的数值结果相同，因为每个 output position 都是在 patch pixels 和一个 filter 之间做 dot product。convolution 就是 patch projection。生产代码通常这样写，因为它在 GPU 上更快，而且少一次 reshape。

### Position embeddings / 位置嵌入

projection 输出的 tokens 本身不携带顺序。2D sinusoidal embedding 给每个 token 一个 fixed signal，编码它的 `(row, col)` 位置。embedding dimension 的一半用多个频率的 sin/cos 编码 row position，另一半编码 column position。该 encoding 是 deterministic 的，因此可以在不重新训练的情况下切换 resolution，并能平滑插值到训练时没见过的 grids。

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

对 224 resolution 的 ViT-Base/16 来说，projection 有 590,592 个 parameters，CLS token 有 768 个，sinusoidal position 为零参数。下一课（59）会把一个 12-layer transformer 堆在这个 front end 上。

### Equivalence as a sanity check / 等价性作为健全性检查

patch step 有两种写法：`Conv2d` projection 和显式 unfold-then-linear。给定相同 weights 时，它们必须产生相同 output。否则 unfold math 就错了，后面的 encoder 也建立在不可靠基础上。本课 tests 会覆盖这个 equivalence。

## Build It / 动手构建

`code/main.py` implements:

- `PatchEmbed`, an `nn.Module` wrapping `Conv2d` for patch projection.
- `sinusoidal_2d(grid_h, grid_w, dim)`, a stateless function that builds the 2D position table.
- `VisionFrontEnd`, which composes patch embedding, CLS prepend, and position addition into one forward pass.
- A `synthesize_image(seed)` helper that builds a deterministic 224x224x3 fixture from `numpy.random`.
- A demo that runs one fixture image through the front end and prints the output shape, the CLS token norm, and one row of the position embedding.

Run it:

```bash
python3 code/main.py
```

输出：224x224 fixture 会被 tokenized 成 shape `(1, 197, 768)` 的 sequence。第一个 token 是 CLS，后面 196 个是 patch tokens。position embedding norms 在同一 row 内保持一致，这是 sinusoidal signature。

## Use It / 应用它

同样的 patch front end 出现在几乎所有现代 vision-language model 中：CLIP ViT-L/14、SigLIP、DINOv2、Qwen-VL family、InternVL stack 都从 `Conv2d` patch projection 加 position signal 开始。不同模型的差异主要在下游：CLS vs no-CLS pooling、register tokens、patch size 14 vs 16、通过 interpolated positions 支持 dynamic resolution。本课的 frontend 是这些模型共同站立的基底。

## Tests / 测试

`code/test_main.py` covers:

- patch count matches `(image_size / patch_size) ** 2`
- output shape matches `(batch, num_patches + 1, hidden)`
- the `Conv2d` projection equals manual unfold-then-linear on a small fixture
- sinusoidal position table is deterministic across calls
- CLS token broadcasts across batch dim without leakage

Run them:

```bash
python3 -m unittest code/test_main.py
```

## Ship It / 交付它

交付物是一个可复用的 `VisionFrontEnd`：给定 image tensor，它输出带 CLS 和 position signal 的 token sequence。它应该能作为后续 ViT encoder、projection head、cross-attention decoder 的稳定输入面，并通过 `Conv2d`/unfold equivalence 测试。

## Exercises / 练习

1. 把 sinusoidal position 替换为 learned `nn.Parameter`，并在 tiny synthetic classification task 上比较 first-epoch loss。fixed resolution 下 learned positions 更强；训练后切换 resolution 时 sinusoidal 更稳。

2. 把 `Conv2d` 替换成显式 `nn.Unfold` 加 `nn.Linear`，并断言 outputs 在 float tolerance 内一致。同一数学，两种写法。

3. 支持 non-square patch sizes（例如面向 wide-aspect inputs 的 32x16），并验证 position table 能处理 non-square grids。

4. 在 batch sizes 1、8、64 下 profile patch step。patch projection 很少是瓶颈；下游 attention layers 才主导成本。

5. 在 4-class synthetic shape dataset（circles、squares、triangles、stars）上把 front end 当作 frozen feature extractor 训练。CLS token output 应该能被线性分开。

## Key Terms / 关键术语

| Term | What it means |
|------|---------------|
| Patch | 图像的方形子区域，通常是 14x14 或 16x16 |
| Patch embedding | 把一个 flattened patch 线性投影到 hidden dim |
| Sequence length | patch tokenization 后的 token 数量，通常还要加 CLS |
| Sinusoidal position | 编码 2D grid coordinates 的 fixed sin/cos signal |
| CLS token | 作为 pooling head 被 prepend 到 sequence 前面的 learned vector |

## Further Reading / 延伸阅读

- An Image is Worth 16x16 Words (ViT, 2021) for the original patch-embed framing.
- Attention Is All You Need (2017) for the sinusoidal position formula adapted here to 2D.
- DINOv2 paper for register tokens, an extension you can add as exercise 6.
