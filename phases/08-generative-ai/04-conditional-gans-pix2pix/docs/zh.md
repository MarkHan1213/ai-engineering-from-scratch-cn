# Conditional GANs & Pix2Pix / 条件 GAN 与 Pix2Pix

> 2014-2017 年第一个重要突破，是控制 GAN 生成什么。接上 label、image 或 sentence。Pix2Pix 做的是图像版本；在窄 image-to-image 任务上，它到现在仍能击败很多通用 text-to-image 模型。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 conditional GAN 如何把条件 `c` 注入 `G` 和 `D`
- 理解 Pix2Pix 中 U-Net generator、PatchGAN discriminator 和 L1 loss 的分工
- 区分 paired data 的 Pix2Pix 与 unpaired data 的 CycleGAN
- 根据数据可用性和延迟预算选择 Pix2Pix、CycleGAN、ControlNet 或 diffusion fallback

## The Problem / 问题

Unconditional GAN 会随机采样人脸。演示有用，生产没用。你真正想要的是：*把草图变成照片*、*把地图变成航拍图*、*把白天场景变成夜晚*、*给灰度图上色*。在这些任务中，你都有输入图像 `x`，必须输出与之有语义对应的 `y`。每个 `x` 都可能对应许多合理的 `y`。Mean-squared error 会把它们压成糊状。Adversarial loss 不会，因为“看起来真实”这个信号很锐利。

Conditional GAN（Mirza & Osindero, 2014）把 condition `c` 作为输入加到 `G` 和 `D`。Pix2Pix（Isola et al., 2017）把它专门化到图像：condition 是完整输入图像，generator 是 U-Net，discriminator 是 *patch-based* classifier（PatchGAN），loss 是 adversarial + L1。到 2026 年，这个配方在窄 image-to-image 领域仍能超过从零训练的 text-to-image 模型，因为它使用的是 *paired data*：你正好拥有所需信号。

## The Concept / 概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Conditional G / 条件生成器。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入 noise；Isola 发现显式 noise 会被忽略）。

**Conditional D / 条件判别器。** `D(x, y) → [0, 1]`。输入是 *pair*（condition, output）。这是关键差异：D 必须判断 `y` 是否与 `x` 一致，而不仅是 `y` 是否看起来真实。

**U-Net generator / U-Net 生成器。** 带 skip connections 的 encoder-decoder。对于输入和输出共享低层结构（边缘、轮廓）的任务非常关键。没有 skips，高频细节会消失。

**PatchGAN discriminator / PatchGAN 判别器。** D 不输出单个 real/fake 分数，而是输出一个 `N×N` 网格，每个 cell 判断约 70×70 像素的 receptive field。最后取平均。这是一个 Markov random field 假设：真实感是局部的。训练更快、参数更少、输出更锐利。

**Loss / 损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 term 稳定训练，并把 G 推向已知 target。L1 比 L2 给出更锐利边缘（medians，不是 means）。`λ = 100` 是 Pix2Pix 默认值。

## CycleGAN — When You Don't Have Pairs / CycleGAN：没有配对数据时

Pix2Pix 需要 paired `(x, y)` data。CycleGAN（Zhu et al., 2017）放弃这个要求，但额外引入 *cycle consistency* loss。两个 generators：`G: X → Y` 和 `F: Y → X`。训练目标是 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这样就能在没有配对样本的情况下，把 horse 转成 zebra、summer 转成 winter。

到 2026 年，unpaired image-to-image 大多用 diffusion（ControlNet、IP-Adapter）而不是 CycleGAN，但 cycle-consistency 思想仍存在于几乎所有 unpaired domain adaptation 论文中。

## Build It / 动手构建

`code/main.py` 在 1-D 数据上实现一个小型 conditional GAN。Condition `c` 是 class label（0 或 1）。任务：对给定 class 生成来自条件分布的样本。

### Step 1: append condition to both G and D inputs / 第 1 步：把 condition 拼到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding 是最简单的方式。更大的模型会用 learned embeddings、FiLM modulation 或 cross-attention。

### Step 2: train conditional / 第 2 步：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator 必须匹配 *给定 condition 下* 的真实分布，而不是 marginal distribution。

### Step 3: verify per-class output / 第 3 步：按 class 验证输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pitfalls / 常见坑

- **Condition ignored。** G 学会 marginalize，D 不惩罚，因为 condition signal 太弱。修复：更强地 condition D（早层，而不是只在后层），使用 projection discriminator（Miyato & Koyama 2018）。
- **L1 weight too low。** G 漂到任意真实感输出，而不是 faithful outputs。Pix2Pix-style 任务从 λ≈100 开始。
- **L1 weight too high。** G 输出模糊，因为 L1 仍是 L_p norm。训练稳定后可以 anneal down。
- **Ground-truth leakage in D。** D 输入应 concat `(x, y)`，而不是只给 `y`。否则 D 无法检查一致性。
- **Mode collapse per class。** 每个 class 都可能独立 collapse。运行 class-conditional diversity checks。

## Use It / 应用它

2026 年 image-to-image 任务状态：

| Task / 任务 | Best approach / 最佳方法 |
|------|---------------|
| Sketch → photo, same domain, paired data | Pix2Pix / Pix2PixHD（仍然快、仍然锐利） |
| Sketch → photo, unpaired | 带 Scribble conditioning model 的 ControlNet |
| Semantic seg → photo | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| Style transfer | Diffusion + IP-Adapter 或 LoRA；GAN 方法已偏 legacy |
| Depth → photo | Stable Diffusion 上的 ControlNet-Depth |
| Super-resolution | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（diffusion） |
| Colorization | ColTran、diffusion-based colorizers 或 Pix2Pix-color |
| Daytime → nighttime, seasons, weather | CycleGAN 或 ControlNet-based |

当 (a) 你有数千个 paired examples，(b) 任务狭窄且重复，(c) 需要快速 inference 时，Pix2Pix 仍是正确工具。通用开放域任务则 diffusion 赢。

## Ship It / 交付它

保存 `outputs/skill-img2img-chooser.md`。Skill 接收任务描述、数据可用性（paired vs unpaired、N samples）和 latency/quality budget，然后输出：approach（Pix2Pix、CycleGAN、ControlNet variant、SDXL + IP-Adapter）、training data requirements、inference cost 和 eval protocol（LPIPS、FID、task-specific）。

## Exercises / 练习

1. **Easy / 简单。** 修改 `code/main.py`，加入第三个 class。确认 G 仍能把每个 class 的 noise 映射到正确 mode。
2. **Medium / 中等。** 在 1-D 设置里用 perceptual-style loss 替换 L1（例如把一个小型 frozen D 当 feature extractor）。它会改变 conditional distribution 的 sharpness 吗？
3. **Hard / 困难。** 在 1-D 设置中草拟一个 CycleGAN：两个 distributions、两个 generators、cycle loss。展示它如何在无 paired data 下学习映射。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN with labels" | G(z, c)、D(x, c)。两个网络都看到 condition。 |
| Pix2Pix | "Image-to-image GAN" | Paired cGAN，使用 U-Net G 和 PatchGAN D + L1 loss。 |
| U-Net | "Encoder-decoder with skips" | 对称 conv network；skips 保留 high-freq。 |
| PatchGAN | "Local-realism classifier" | D 输出 per-patch score，而不是 global score。 |
| CycleGAN | "Unpaired image translation" | 两个 G + cycle-consistency loss；不需要 paired data。 |
| SPADE | "GauGAN" | 用 semantic map 调制中间 activations 的 normalization；segmentation-to-image。 |
| FiLM | "Feature-wise linear modulation" | condition 生成 per-feature affine transform；便宜的 conditioning。 |

## Production Note: Pix2Pix as a Latency-Bound Baseline / 生产备注：Pix2Pix 是低延迟基线

当你有 paired data 和窄任务（sketch → render、semantic map → photo、day → night）时，Pix2Pix 的 one-shot inference 在延迟上通常比 diffusion 快一个数量级。生产比较一般是：

| Path / 路径 | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在 static batches 中赢在 throughput（每个 request FLOPs 相同）。Diffusion 赢在质量和泛化。现代打法通常是在窄任务上发布一个 Pix2Pix-style distilled model，再给 tail inputs 留一个 diffusion fallback。

## Further Reading / 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D。
