# Autoencoders & Variational Autoencoders (VAE) / 自编码器与变分自编码器

> 普通 autoencoder 先压缩再重建。它会记忆，但不会生成。加一个技巧：强制 code 看起来像 Gaussian，你就得到了 sampler。正是 `z = μ + σ·ε` 这个 reparameterization，让你在 2026 年使用的每个 latent-diffusion 和 flow-matching 图像模型，都在输入端放了一个 VAE。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分普通 autoencoder 和真正可采样的 VAE
- 推导 `q(z|x) = N(μ(x), σ(x)²)`、KL penalty 与 ELBO 的角色
- 解释 reparameterization trick 为什么让采样节点可以反向传播
- 判断 2026 年 VAE 在 latent diffusion、audio codec 和 discrete latents 中的实际位置

## The Problem / 问题

把一个 784-pixel MNIST digit 压缩成 16 个数字，再重建出来。普通 autoencoder 会把 reconstruction MSE 做得很好，但 code space 会变成一团凹凸不平的乱麻。你在 code space 里随机取一个点再 decode，得到的只会是噪声。它没有 sampler；它只是穿着生成模型外衣的压缩模型。

你真正想要的是三件事同时成立：(a) code space 是一个干净、平滑、可采样的分布，比如 isotropic Gaussian `N(0, I)`；(b) decode 任意样本都能生成合理 digit；(c) encoder 和 decoder 仍然压缩得好。三个目标，一个架构，一个 loss。

Kingma 2013 年的 VAE 通过让 encoder 输出一个 *distribution* `q(z|x) = N(μ(x), σ(x)²)` 来解决这个问题，用 KL penalty 把这个分布拉向 prior `N(0, I)`，再从 `q(z|x)` 中 sample `z` 后 decode。推理时丢掉 encoder，直接 sample `z ~ N(0, I)`，再 decode。KL penalty 负责把 code space 变成有结构的空间。

到 2026 年，VAE 很少单独作为产品发布，因为原始图像质量已经被 diffusion 超过；但它仍是每个 latent-diffusion 模型的首选 encoder（SD 1/2/XL/3、Flux、AudioCraft）。学会 VAE，你就理解了每天使用的图像 pipeline 里那层看不见的第一层。

## The Concept / 概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder / 自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，loss = `||x - x̂||²`。Code space 没有结构。

**VAE encoder / VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**Reparameterization trick / 重参数化技巧。** 从 `q(z|x)` 采样不可微。把样本改写成 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加上一个非参数噪声的确定性函数，梯度可以穿过 `μ` 和 `σ`。

**Loss / 损失。** Evidence Lower BOund（ELBO）有两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Reconstruction 把 `x̂` 推向 `x`。KL 把 `q(z|x)` 推向 prior。两者会互相拉扯。小 β（<1）= 样本更锐利，code space 没那么 Gaussian。大 β（>1）= code space 更干净，样本更模糊。β-VAE（Higgins 2017）让这个旋钮出名，并开启了 disentanglement 研究。

**Sampling / 采样。** 推理时：抽 `z ~ N(0, I)`，送进 decoder。一次 forward pass，不像 diffusion 那样迭代采样。

```figure
vae-latent-grid
```

## Build It / 动手构建

`code/main.py` 实现了一个不用 numpy 或 torch 的小型 VAE。输入是从 8-D 双组分 Gaussian mixture 中采样的 8 维 synthetic data。Encoder 和 decoder 都是单 hidden-layer MLP。我们手写 tanh activation、forward pass、loss 和 backward pass。它不是生产代码，而是教学代码。

### Step 1: encoder forward / 第 1 步：encoder 前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用 `log σ²` 而不是 `σ`，这样网络输出不受约束（对 σ 做 softplus 是个陷阱：在 σ ≈ 0 时梯度会死）。

### Step 2: reparameterize and decode / 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Step 3: the ELBO / 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是 Gaussian，KL 有精确 closed-form。不要数值积分。到 2026 年仍有人上线 monte-carlo KL estimate，慢 3 倍且没有理由。

### Step 4: generate / 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行。

## Pitfalls / 常见坑

- **Posterior collapse。** KL term 太强地把 `q(z|x) → N(0, I)`，导致 `z` 不再携带 `x` 的信息。修复：β-annealing（从 β=0 开始逐步升到 1）、free bits，或跳过 inactive dimensions 上的 KL。
- **Blurry samples。** Gaussian decoder likelihood 意味着 MSE reconstruction，而 MSE 对 L2 的 Bayes-optimal 输出是均值；一组合理 digit 的均值就是模糊 digit。修复：discrete decoder（VQ-VAE、NVAE），或只把 VAE 当 encoder，再在 latents 上叠 diffusion（Stable Diffusion 就这么做）。
- **β too large, too early。** 见 posterior collapse。先从 β≈0.01 开始，再逐步拉升。
- **Latent dim too small。** 16-D 适合 MNIST，256-D 适合 ImageNet 256²，2048-D 适合 ImageNet 1024²。Stable Diffusion 的 VAE 把 512×512×3 压缩成 64×64×4（空间面积 32x 下采样，channels 也压 32x）。

## Use It / 应用它

2026 年的 VAE 技术栈：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| Audio-latent encoder | Encodec (Meta)、SoundStream 或 DAC (Descript) |
| Video latents | Sora 的 spatiotemporal patches、Latte VAE、WAN VAE |
| Disentangled representation learning | β-VAE、FactorVAE、TCVAE |
| Discrete latents (for transformer modelling) | VQ-VAE、RVQ (ResidualVQ) |
| Continuous latents for generation | Plain VAE，然后在该 latent space 中 condition 一个 flow/diffusion model |

Latent-diffusion model 本质上是一个 VAE，中间夹着一个 diffusion model。VAE 做粗压缩，diffusion model 做重活。Video（VAE + video-diffusion DiT）和 audio（Encodec + MusicGen transformer）也是同一模式。

## Ship It / 交付它

保存 `outputs/skill-vae-trainer.md`。

Skill 接收：dataset profile + latent-dim target + downstream use（reconstruction、sampling 或 latent-diffusion input），并输出：architecture choice（plain/β/VQ/RVQ）、β schedule、latent dim、decoder likelihood（Gaussian vs categorical）和 evaluation plan（recon MSE、KL per dim、`q(z|x)` 与 `N(0, I)` 的 Fréchet distance）。

## Exercises / 练习

1. **Easy / 简单。** 把 `code/main.py` 里的 `β` 改成 `0.01`、`0.1`、`1.0`、`5.0`。记录最终 reconstruction MSE 和 KL。哪个 β 对你的 synthetic data 是 Pareto-best？
2. **Medium / 中等。** 把 Gaussian decoder likelihood 换成 Bernoulli likelihood（cross-entropy loss）。在同一 synthetic data 的二值化版本上比较 sample quality。
3. **Hard / 困难。** 把 `code/main.py` 扩展成 mini VQ-VAE：用 K=32 entries 的 codebook 中 nearest-neighbour lookup 替换连续 `z`。比较 reconstruction MSE，并报告多少 codebook entries 被使用（codebook collapse 是真实问题）。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`，学习 MSE。不是生成模型。 |
| VAE | AE with a sampler | Encoder 输出一个分布，KL penalty 塑造 code space。 |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时 tight。 |
| Reparameterization | `z = μ + σ·ε` | 把随机节点改写成 deterministic + pure noise，使采样可反向传播。 |
| Prior | `p(z)` | latent 的目标分布，通常是 `N(0, I)`。 |
| Posterior collapse | "KL term wins" | Encoder 忽略 `x`，输出 prior；decoder 只能幻觉生成。 |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`。β 越高越 disentangled，但也越模糊。 |
| VQ-VAE | Discrete latent | 用 nearest codebook vector 替换连续 `z`；支持 transformer modelling。 |

## Production Note: The VAE Is the Hottest Path in a Diffusion Server / 生产备注：VAE 是 diffusion server 里的高压路径

在 Stable Diffusion / Flux / SD3 pipeline 中，VAE 每个 request 通常会被调用两次：img2img / inpainting 时 encode 一次，最后 decode 一次。到 1024² 时，decoder pass 往往是整个 pipeline 最大的 activation-memory peak，因为它要把 `128×128×16` latents 上采样回 `1024×1024×3`。这带来两个实际后果：

- **Slice or tile the decode / 切片或分块 decode。** `diffusers` 暴露 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。Tiling 用轻微 seam artifact 换取 `O(tile²)` memory，而不是 `O(H·W)`。消费级 GPU 上做 1024²+ 时基本必需。
- **bf16 decoder, fp32 numerics for the final resize / decoder 用 bf16，最终 resize 用 fp32 数值。** SD 1.x VAE 发布时是 fp32，cast 到 fp16 后在 1024²+ 会 *悄悄产生 NaNs*。SDXL 有 `madebyollin/sdxl-vae-fp16-fix`，始终优先用 fp16-fix variant 或 bf16。

## Further Reading / 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为 encoder。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，audio VAE 标准。
