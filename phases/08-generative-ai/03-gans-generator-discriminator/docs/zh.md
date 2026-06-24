# GANs — Generator vs Discriminator / GAN：生成器与判别器

> Goodfellow 在 2014 年的技巧，是完全跳过 density。两个网络：一个造假，一个抓假。它们互相对抗，直到假样本和真样本无法区分。它本不该稳定工作，也经常确实不工作。但一旦奏效，在窄领域里它生成的样本至今仍是文献中最锐利的一类。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 GAN 如何在不写出 `log p(x)` 的情况下学习 data distribution
- 推导 generator、discriminator 和 non-saturating loss 的训练信号
- 识别 mode collapse、vanishing gradient 和 discriminator 过强等常见失效模式
- 判断 2026 年 GAN 在窄领域生成、perceptual loss 和 diffusion distillation 中的实际用途

## The Problem / 问题

VAE 会生成模糊样本，因为它的 MSE decoder loss 对 *mean* image 是 Bayes-optimal；而许多合理 digit 的均值就是模糊 digit。你想要一种奖励 *plausibility* 的 loss，而不是奖励它和某个 target 的逐像素接近。plausibility 没有 closed-form，你必须学习它。

Goodfellow 的想法：训练一个 classifier `D(x)` 区分真图和假图；训练一个 generator `G(z)` 欺骗 `D`。`G` 的 loss signal 就是 `D` 当前认为“什么东西看起来真实”。这个信号会随着 `G` 进步而变化，目标始终在移动。如果两个网络都收敛，`G` 就在从未写下 `log p(x)` 的情况下学到了 data distribution。

这就是 adversarial training。数学上是一个 minimax game：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到 2026 年，GAN 已不再是 SOTA generator（diffusion 和 flow matching 拿走了王冠）。但 StyleGAN 2/3 仍是发布过的最锐利人脸模型，GAN discriminator 被用作 diffusion training 里的 *perceptual losses*，adversarial training 也支撑了快速 1-step distillations（SDXL-Turbo、SD3-Turbo、LCM），让 real-time diffusion 能够上线。

## The Concept / 概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Generator `G(z)` / 生成器。** 把 noise vector `z ~ N(0, I)` 映射成样本 `x̂`。形状通常像 decoder（dense 或 transposed conv）。

**Discriminator `D(x)` / 判别器。** 把样本映射成 scalar probability（或 score）。Real → 1，fake → 0。

**Loss / 损失。** 两个交替更新：

- **Train `D` / 训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。Binary cross-entropy，real=1，fake=0。
- **Train `G` / 训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的 *non-saturating* 形式（原始 `log(1 - D(G(z)))` 在 `D` 很自信时会 saturate，杀死梯度）。

**Training loop / 训练循环。** 一步 `D`，一步 `G`。重复。

**Why it works / 为什么有效。** 如果 `G` 完全匹配 `p_data`，`D` 最多只能随机猜测，到处输出 0.5；`G` 不再得到梯度。达到均衡。

**Why it breaks / 为什么会坏。** Mode collapse（`G` 找到一个 `D` 抓不住的 mode 后无限复制）、vanishing gradient（`D` 学太快，`log D` saturate）、训练不稳定（learning rate、batch size、任何细节都可能触发）。

## Variants That Made GANs Work / 让 GAN 变得可用的变体

| Year / 年份 | Innovation / 创新 | Fix / 修复点 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、batch norm、LeakyReLU：第一个稳定架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein distance + gradient penalty 替换 BCE。修复 vanishing gradient。 |
| 2017 | Spectral normalization | 给 discriminator 加 Lipschitz bound。2026 年 discriminator 仍常用。 |
| 2018 | Progressive GAN | 先训低分辨率，再逐层加入。第一次出现 megapixel results。 |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm。固定领域 photorealism 的 SOTA。 |
| 2021 | StyleGAN3 | Alias-free、translation-equivariant：2026 年仍是人脸 gold standard。 |
| 2022 | StyleGAN-XL | Conditional、class-aware、更大规模。 |
| 2024 | R3GAN | 用更强 regularization 重新包装；无需花哨技巧即可在 1024² 工作。 |

```figure
gan-minimax
```

## Build It / 动手构建

`code/main.py` 在 1-D 数据上训练一个小 GAN：双峰 Gaussian mixture。Generator 和 discriminator 都是 single-hidden-layer MLP。我们手写 forward、backward 和 minimax loop。目标是让你亲眼看到两个关键失效模式（mode collapse + vanishing gradient）如何发生。

### Step 1: non-saturating loss / 第 1 步：non-saturating loss

Vanilla Goodfellow loss `log(1 - D(G(z)))` 在 D 高置信地把 G 的 fake 判为 fake 时趋近 0。此时 G 的梯度几乎为零，无法改进。Non-saturating 形式 `-log D(G(z))` 有相反的渐近行为：当 D 很自信时它会变大，给 G 强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Step 2: one discriminator step per generator step / 第 2 步：每步 generator 对应一步 discriminator

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G 训练时要用 fresh fakes，否则梯度是 stale 的。

### Step 3: watch for mode collapse / 第 3 步：观察 mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实 mode 中有一个不再被生成。discriminator 也不再纠正它，因为它从没作为 fake 出现过。

## Pitfalls / 常见坑

- **Discriminator too strong。** 把 D 的 learning rate 降低 2-5 倍，或加 instance/layer noise。如果 D accuracy 到了 >95%，G 基本死了。
- **Generator memorizes a mode。** 给 D input 加噪声，使用 minibatch-discriminator layer，或切到 WGAN-GP。
- **Batch norm leaking statistics。** Real batch + fake batch 同时穿过同一 BN layer，会混合统计量。用 instance norm 或 spectral norm。
- **Inception-score gaming。** FID 和 IS 在低 sample count 时噪声很大。评估使用 ≥10k samples。
- **One-shot sampling is a lie for conditional tasks。** 你仍然需要 CFG scales、truncation tricks 和 re-sampling 才能得到可用输出。

## Use It / 应用它

2026 年的 GAN 技术栈：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| Photoreal human faces, fixed pose | StyleGAN3（最锐利、最小） |
| Anime / stylized faces | StyleGAN-XL 或 Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| Fast 1-step text-to-image | Adversarial distillation of diffusion（SDXL-Turbo、SD3-Turbo） |
| Perceptual loss inside a diffusion trainer | 小型 GAN discriminator，作用在 image crops 上 |
| Anything multi-modal, open-ended | 不要用 GAN；用 diffusion 或 flow matching |

GAN 锐利但窄。一旦 domain 打开，比如照片、任意文本 prompt、视频，就切到 diffusion。Adversarial trick 会作为组件继续存在（perceptual losses、distillation），而不是单独当 generator。

## Ship It / 交付它

保存 `outputs/skill-gan-debugger.md`。Skill 接收一个失败的 GAN run（loss curves、sample grid、dataset size），输出可能原因的排序列表、单行修复建议和 rerun protocol。

## Exercises / 练习

1. **Easy / 简单。** 使用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重跑。G 的 loss 多快塌成常数？
2. **Medium / 中等。** 用 WGAN loss 替换 Goodfellow BCE loss：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并把 D 的 weights clip 到 `[-0.01, 0.01]`。训练是否更稳定？比较 wall-clock convergence。
3. **Hard / 困难。** 把 1-D 例子扩展到 2-D 数据（8 个 Gaussian 组成的 ring）。跟踪 generator 在 steps 1k、5k、10k 捕获了 8 个 mode 中的几个。实现 minibatch discrimination 并重新测量。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Generator | "G" | Noise-to-sample network，`G: z → x̂`。 |
| Discriminator | "D" | Classifier `D: x → [0, 1]`，判断 real vs fake。 |
| Minimax | "The game" | 联合目标的 `min_G max_D`。 |
| Non-saturating loss | "The fix" | G 使用 `-log D(G(z))`，而不是 `log(1 - D(G(z)))`。 |
| Mode collapse | "G memorized one thing" | Generator 只产生少数不同输出，尽管数据很多样。 |
| WGAN | "Wasserstein" | 用 Earth-Mover distance + gradient penalty 替换 BCE，梯度更平滑。 |
| Spectral norm | "Lipschitz trick" | 约束 D 的 weight norms 来限制斜率，稳定训练。 |
| StyleGAN | "The one that works" | Mapping network + AdaIN；人脸领域 best-in-class，2026 年仍然如此。 |

## Production Note: One-Shot Inference Is GAN's Lasting Advantage / 生产备注：一次前向推理是 GAN 留下的优势

在开放域生成质量上，GAN 已经不再赢，但它仍然赢在 inference cost。用 production-inference 文献里的词汇描述，GAN 具备：

- **No prefill, no decode stages / 没有 prefill，也没有 decode 阶段。** 一次 `G(z)` forward pass。TTFT ≈ total latency。
- **No KV-cache pressure / 没有 KV-cache 压力。** 唯一状态是 weights。Batch size 受 activation memory 限制，而不是 cache。
- **Trivial continuous batching / continuous batching 很简单。** 每个 request FLOPs 固定相同，因此服务器目标 occupancy 下的 static batch 通常就是最优。不需要 in-flight scheduler。

这就是为什么 GAN distillation（SDXL-Turbo、SD3-Turbo、ADD、LCM）会成为 2026 年快速 text-to-image 的主流技术：它把 20-50-step diffusion pipeline 压成 1-4 次 GAN-style forward passes，同时保留 diffusion base 的分布。Adversarial loss 作为训练期旋钮继续存在，用来把慢 generator 变成快 generator。

## Further Reading / 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
