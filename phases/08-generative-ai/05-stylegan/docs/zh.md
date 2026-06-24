# StyleGAN / StyleGAN

> 大多数 generator 会把 `z` 一次性搅进所有层。StyleGAN 把它拆开：先把 `z` 映射成中间变量 `w`，再通过 AdaIN 在每个 resolution level 注入 `w`。这个改动让 latent space 解缠，也让 photorealistic faces 连续七年几乎成为已解决问题。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 StyleGAN 为什么把 `Z` 映射到中间 `W` space
- 理解 mapping network、synthesis network、AdaIN 和 per-layer noise 的作用
- 区分 StyleGAN 1/2/3 的关键改动及其修复的 artifact
- 判断 StyleGAN 在 2026 年 narrow-domain photorealism 和 inversion editing 中的适用边界

## The Problem / 问题

DCGAN 通过一串 transposed convolutions 把 `z` 映射到图像。问题是：`z` 控制一切，包括 pose、lighting、identity、background，而且都纠缠在一起。沿着 `z` 的某个轴移动，四者一起变。你无法要求模型“同一个人，不同 pose”，因为表示并没有这样分解。

Karras et al.（2019, NVIDIA）提出：不要把 `z` 直接喂给 conv layers。把一个常量 `4×4×512` tensor 作为网络输入。学习一个 8-layer MLP，把 `z ∈ Z → w ∈ W`。在每个 resolution 通过 *adaptive instance normalization*（AdaIN）注入 `w`：先 normalize 每个 conv feature map，再用 `w` 的 affine projections 做 scale 和 shift。再加 per-layer noise 来处理 stochastic detail（皮肤毛孔、发丝）。

结果是：`W` 里“high-level style”（pose、identity）和“fine style”（lighting、color）大致形成正交轴。你可以用图像 A 的 `w` 控制低分辨率层，用图像 B 的 `w` 控制高分辨率层，实现 style swap。这打开了 editing、cross-domain stylization 和整个 “StyleGAN-inversion” 研究线。

## The Concept / 概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network / 映射网络。** `f: Z → W`，一个 8-layer MLP。`Z = N(0, I)^512`。`W` 不被强制成 Gaussian，它会学习适配数据的形状。

**Synthesis network / 合成网络。** 从 learned constant `4×4×512` 开始。每个 resolution block：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。Resolution 依次翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的 affine projections。先按 feature map normalize，再重新设定 style。这里的 “Style” 指 feature map 的一阶和二阶统计。

**Per-layer noise / 逐层噪声。** 给每个 feature map 加 single-channel Gaussian noise，并用可学习的 per-channel factor 缩放。它控制 stochastic detail，但不影响全局结构。

**Truncation trick / 截断技巧。** 推理时 sample `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多样本的平均 `w`。`ψ < 1` 用 diversity 换 quality。几乎所有 StyleGAN demo 都用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3 / StyleGAN 版本演进

| Version / 版本 | Year / 年份 | Innovation / 创新 |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing。 |
| StyleGAN2 | 2020 | Weight demodulation 替代 AdaIN（修复 droplet artifacts）；skip/residual architecture；path-length regularization。 |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernels；消除 texture sticking to pixel grid。 |
| StyleGAN-XL | 2022 | Class-conditional、1024²、ImageNet。 |
| R3GAN | 2024 | 用更强 reg 重新包装；以 20x 更少参数缩小与 diffusion 在 FFHQ-1024 上的差距。 |

到 2026 年，StyleGAN3 仍是默认选择，适用于：(a) 高 FPS 的窄领域 photorealism，(b) few-shot domain adaptation（用 100 张新数据集图像训练，freeze mapping），(c) inversion-based editing（找到能重建真实照片的 `w`，再编辑 `w`）。对于开放域 text-to-image，它不是正确工具；diffusion 才是。

## Build It / 动手构建

`code/main.py` 在 1-D 中实现 toy “style-GAN lite”：一个 mapping MLP，一个以 learned constant vector 为输入并用 `w` 派生的 scale/bias 调制的 synthesis function，以及 per-layer noise。它展示通过 affine-modulation 注入 `w`，可以匹配或超过把 `z` concat 到 generator 输入的做法。

### Step 1: mapping network / 第 1 步：mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Step 2: adaptive instance normalization / 第 2 步：adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Per-feature-map scale 和 bias 来自 `w` 的 linear projection。

### Step 3: per-layer noise / 第 3 步：per-layer noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per-channel 是 learnable 的。

## Pitfalls / 常见坑

- **Droplet artifacts。** StyleGAN 1 会在 feature maps 中产生 blob-like droplet，因为 AdaIN 把 mean 归零了。StyleGAN 2 的 weight demodulation 通过缩放 convolution weights 修复它。
- **Texture sticking。** StyleGAN 1 和 2 的 textures 跟着 pixel coordinates 走，而不是 object coordinates（插值时很明显）。StyleGAN 3 用 alias-free convolutions 和 windowed sinc filters 修复。
- **Mode coverage。** Truncation `ψ < 0.7` 看起来干净，但只在一个窄 cone 中采样；如果需要 diversity，用 `ψ = 1.0`。
- **Inversion is lossy。** 把真实照片 inversion 到 `W` 通常通过 optimization 或 encoder（e4e、ReStyle、HyperStyle）。迭代多了结果会 drift。

## Use It / 应用它

| Use case / 使用场景 | Approach / 方法 |
|----------|----------|
| Photoreal human faces (anime, product, narrow) | StyleGAN3 FFHQ / custom fine-tune |
| Face editing from a photo | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment | StyleGAN + encoder + blending |
| Avatar pipelines | StyleGAN3 w/ ADA for low-data fine-tune |
| Domain adaptation from a few images | Freeze mapping network, fine-tune synthesis |
| Multi-modal or text-conditioned generation | 不要用；用 diffusion |

对于答案就是“某个人脸照片”的 product-grade demo，StyleGAN 在 inference cost（single forward pass，4090 上 <10ms）和同质量门槛下的 sharpness 上都胜过 diffusion。

## Ship It / 交付它

保存 `outputs/skill-stylegan-inversion.md`。Skill 接收一张真实照片并输出：inversion method（e4e / ReStyle / HyperStyle）、expected latent loss、editing budget（在 `W` 里能移动多远而不出 artifact），以及已知可用 editing directions（age、expression、pose）。

## Exercises / 练习

1. **Easy / 简单。** 分别用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定 latent 与 perturbed latent 下输出的 spread。
2. **Medium / 中等。** 实现 mixing regularization：对一个 training batch 计算 `w_a`、`w_b`，在 synthesis 前半段用 `w_a`，后半段用 `w_b`。Decoder 是否学到了 disentangled styles？
3. **Hard / 困难。** 取一个 pretrained StyleGAN3 FFHQ model（ffhq-1024.pkl）。通过在 labelled samples 上训练 SVM，找到控制 “smile” 的 `w` direction；报告 identity drift 之前能推多远。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mapping network | "The MLP" | `f: Z → W`，8 层，把 latent geometry 和 data statistics 解耦。 |
| W space | "The style space" | Mapping network 的输出；大致 disentangled。 |
| AdaIN | "Adaptive instance norm" | Normalize feature map，然后用 `w`-projection 做 scale + shift。 |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 用 diversity 换 quality。 |
| Path-length regularization | "PL reg" | 惩罚 `w` 单位变化导致的 image 大幅变化，使 `W` 更平滑。 |
| Weight demodulation | "The StyleGAN2 fix" | Normalize conv weights，而不是 activations；消除 droplet artifacts。 |
| Alias-free | "StyleGAN3's trick" | Windowed sinc filters；消除 texture sticking to pixel grid。 |
| Inversion | "Find w for a real image" | Optimize 或 encode `x → w`，使 `G(w) ≈ x`。 |

## Production Note: Why StyleGAN Still Ships in 2026 / 生产备注：为什么 2026 年仍有人上线 StyleGAN

StyleGAN3 在 4090 上生成一张 1024² FFHQ 人脸不到 10 ms：`num_steps = 1`，没有 VAE decode，没有 cross-attention pass。用生产术语说，这是任何 image generator 的延迟下限。同分辨率的 50-step SDXL + VAE-decode pipeline 约 3 秒。这是 **300× gap**，对 avatar services、ID document pipelines、stock face generation 这类窄领域产品，TCO 上直接胜出。

两个运营后果：

- **No scheduler, no batcher / 不需要复杂 scheduler 或 batcher。** 目标 occupancy 下的 static batch 最优。Continuous batching（LLM 和 diffusion 必需）没有收益，因为每个 request FLOPs 相同。
- **Truncation `ψ` is the safety knob / Truncation `ψ` 是安全旋钮。** `ψ < 0.7` 会从 mapping network 范围里的窄 cone 采样。这是 serving layer 唯一能控制 sample variance 的杠杆。高峰期降低 `ψ`，premium users 提高 `ψ`。

## Further Reading / 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代 minimal GAN recipe。
