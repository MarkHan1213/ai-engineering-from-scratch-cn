# Diffusion Models — DDPM from Scratch / Diffusion Models：从零构建 DDPM

> Ho、Jain、Abbeel（2020）给了这个领域一个再也放不下的配方：用上千个小步骤把数据逐步毁成噪声；训练一个神经网络预测噪声；推理时反向走回去。今天主流图像、视频、3D 和音乐模型都跑在这个循环上，只是上面可能叠了 flow matching 或 consistency trick。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 DDPM 如何把生成问题拆成 forward noise 和 reverse denoise 两条 Markov chains
- 推导 `q(x_t | x_0)` 的 closed form 与 noise-prediction 训练 loss
- 从零实现 1-D DDPM 的 schedule、forward sample、training step 和 reverse sample
- 识别 schedule、timestep embedding、prediction target 和 sampler step count 的生产影响

## The Problem / 问题

你想为 `p_data(x)` 做一个 sampler。GAN 玩 minimax game，经常发散。VAE 从 Gaussian decoder 里生成模糊样本。你真正想要的是一个训练目标，满足：(a) 单一稳定 loss（没有 saddle point，没有 minimax），(b) 是 `log p(x)` 的 lower bound（所以有 likelihood 解释），(c) 生成样本达到 SOTA 质量。

Sohl-Dickstein et al.（2015）给出过理论答案：定义一个 Markov chain `q(x_t | x_{t-1})`，逐步加入 Gaussian noise；再训练一个 reverse chain `p_θ(x_{t-1} | x_t)` 做 denoise。Ho、Jain、Abbeel（2020）证明 loss 可以简化成一行：预测噪声，并把数学清理干净。2020 年它还是 curiosity；2021 年它生成了 state-of-the-art samples；2022 年它变成 Stable Diffusion；到 2026 年，它已经是底座。

## The Concept / 概念

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Forward process `q` / 前向过程。** 用 `T` 个小步骤加入 Gaussian noise。数学可 tractable 的原因，是累计步骤仍是 Gaussian：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 是噪声 schedule。把 `β_t` 在 T=1000 steps 中从 1e-4 线性拉到 0.02，`x_T` 就近似 `N(0, I)`。

**Reverse process `p_θ` / 反向过程。** 学一个 neural net `ε_θ(x_t, t)` 预测加入的噪声。给定 `x_t`，按下式 denoise：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 可以是 `sqrt(β_t)`，也可以是 learned variance。这个表达式看起来丑，但只是代数：根据 posterior `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，再用 noise-predicted estimate 替代 `x_0`。

**Training loss / 训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从 data sample `x_0`，随机选 `t`，sample `ε ~ N(0, I)`，通过 closed form 一步算出 noisy `x_t`，然后回归噪声。一个 loss，没有 minimax，没有 KL，也没有 reparameterization tricks。

**Sampling / 采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。结束。

## Why It Works / 为什么有效

三个直觉：

1. **Denoising is easy; generating is hard / 去噪比生成容易。** 在 `t=T` 时，数据是纯噪声，网络要解决的是平凡问题。在 `t=0` 时，网络只需要清理少量像素。中间 `t` 比较难，但每个噪声级别都把梯度流进同一组 weights。

2. **Score matching in disguise / 伪装成噪声预测的 score matching。** Vincent（2011）证明，预测噪声等价于估计 `∇_x log q(x_t | x_0)`，也就是 *score*。Reverse SDE 使用这个 score 沿密度梯度往高概率区域走。

3. **The ELBO reduces to simple MSE / ELBO 化简成简单 MSE。** 完整 variational lower bound 每个 timestep 都有 KL term。DDPM 的参数化让这些 KL terms 化简成带特定系数的 noise-prediction MSE；Ho 去掉系数（称作 “simple” loss）后，质量反而 *提升*。

```figure
diffusion-denoise
```

## Build It / 动手构建

`code/main.py` 实现了一个 1-D DDPM。数据是双峰 mixture。“net” 是一个小 MLP，接收 `(x_t, t)` 并输出 predicted noise。训练就是一行 loss。采样则迭代 reverse chain。

### Step 1: the forward schedule (closed form) / 第 1 步：forward schedule（closed form）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: sample `x_t` in one shot / 第 2 步：一步采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: one training step / 第 3 步：一个训练步

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: reverse sampling / 第 4 步：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

在一个 1-D 问题上，40 timesteps 和 24-unit MLP 大约 200 epochs 就能学到双峰 mixture。

## Time Conditioning / 时间条件

网络必须知道自己正在 denoise 哪个 timestep。两个标准选择：

- **Sinusoidal embedding。** 像 Transformer positional encoding。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。送进 MLP，再 broadcast 到网络中。
- **Film / group-norm conditioning。** 把 embedding 投影成每个 block 的 per-channel scale/bias（FiLM）。

我们的 toy code 用 sinusoidal → concat。生产 U-Net 用 FiLM。

## Pitfalls / 常见坑

- **Schedule matters a lot。** Linear `β` 是 DDPM 默认值，但 cosine schedule（Nichol & Dhariwal, 2021）在相同 compute 下 FID 更好。如果质量停滞，先换 schedule。
- **Timestep embedding is fragile。** 把 raw `t` 当 float 传进去在 toy 1-D 中可行，图像上会失败；始终使用合适 embedding。
- **V-prediction vs ε-prediction。** 在很小或很大的 t 区域，`ε` 的 signal-to-noise 很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都用它。
- **Classifier-free guidance。** 推理时同时计算 conditional 和 unconditional `ε`，再用 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。Lesson 08 会覆盖。
- **1000 steps is a lot。** 生产使用 DDIM（20-50 steps）、DPM-Solver（10-20 steps）或 distillation（1-4 steps）。见 Lesson 12。

## Use It / 应用它

| Role / 角色 | Typical stack in 2026 / 2026 年典型栈 |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net 或 DiT（Lesson 07） |
| Video latent diffusion | Spatiotemporal DiT（Sora、Veo、WAN） |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | Equivariant diffusion（EDM、RFdiffusion、AlphaFold3） |

Diffusion 是通用生成 backbone。Flow matching（Lesson 13）是 2024-2026 年的竞争者，通常在同等质量下推理更快。

## Ship It / 交付它

保存 `outputs/skill-diffusion-trainer.md`。Skill 接收 dataset + compute budget，并输出：schedule（linear/cosine/sigmoid）、prediction target（ε/v/x）、steps 数、guidance scale、sampler family 和 eval protocol。

## Exercises / 练习

1. **Easy / 简单。** 把 `code/main.py` 中的 T 从 40 改成 10。样本质量（输出 histogram）如何退化？T 到多少时双峰结构 collapse？
2. **Medium / 中等。** 从 ε-prediction 切到 v-prediction。重新推导 reverse step。比较最终 sample quality。
3. **Hard / 困难。** 加入 classifier-free guidance。用 class label `c ∈ {0, 1}` 做 condition，训练时 10% 概率 drop 掉；采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量 `w = 0, 1, 3, 7` 下的 conditional-mode-hit rate。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Forward process | "Adding noise" | 固定 Markov chain `q(x_t \| x_{t-1})`，逐步毁掉数据。 |
| Reverse process | "Denoising" | 学到的 chain `p_θ(x_{t-1} \| x_t)`，逐步重建数据。 |
| β schedule | "The noise ladder" | 每步 variance；linear、cosine 或 sigmoid。 |
| α̅ | "Alpha bar" | 累计乘积 `∏(1 - β)`；给出从 `x_0` 直接到 `x_t` 的 closed form。 |
| Simple loss | "MSE on noise" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有 variational derivation 都会塌到这里。 |
| ε-prediction | "Predict noise" | 输出加入的噪声；标准 DDPM。 |
| V-prediction | "Predict velocity" | 输出 `α·ε - σ·x`；跨 t 的 conditioning 更好。 |
| DDPM | "The paper" | Ho et al. 2020；linear β、1000 steps、U-Net。 |
| DDIM | "Deterministic sampler" | Non-Markov sampler，20-50 steps，训练目标相同。 |
| Classifier-free guidance | "CFG" | 混合 conditional 和 unconditional noise predictions 来放大 conditioning。 |

## Production Note: Diffusion Inference Is a Step-Count Problem / 生产备注：diffusion inference 是 step-count 问题

DDPM 论文跑 T=1000 reverse steps。生产里没人这么做。真实 inference stack 选择三种策略之一，而且每种都能清晰映射到“延迟来自哪里”：

1. **Faster sampler, same model / 更快 sampler，同一个模型。** DDIM（20-50 steps）、DPM-Solver++（10-20）、UniPC（8-16）。直接替换 reverse loop；训练好的 `ε_θ` weights 不变。延迟砍 20-50×。
2. **Distillation / 蒸馏。** 训练 student 用更少 steps 匹配 teacher：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。再砍 5-10× 延迟，但需要重新训练。
3. **Caching and compilation / 缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM diffusion backends、`xformers`/SDPA attention、bf16 weights。每步延迟约砍 2×，可与 (1)(2) 叠加。

对生产 diffusion server，预算讨论和 LLM production 文献类似：latency 是 `num_steps × step_cost + VAE_decode`，throughput 是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT-equivalent 是完整 response time，因为对用户来说图像生成是 “all-at-once”。

## Further Reading / 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 超前时代的 diffusion 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少 steps。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — cosine schedule、learned variance。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一 notation，最干净 recipe。
