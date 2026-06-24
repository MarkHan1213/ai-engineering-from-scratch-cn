# Flow Matching & Rectified Flows / Flow Matching 与 Rectified Flows

> Diffusion models 需要 20-50 个 sampling steps，因为它们从噪声到数据走的是弯曲路径。Flow matching（Lipman et al., 2023）和 rectified flow（Liu et al., 2022）训练的是更直的路径。路径越直，steps 越少，inference 越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切到了 flow matching。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 flow matching 如何把生成建模改写成学习 velocity field
- 推导 straight-line interpolant、训练 target 和 Euler sampler
- 区分 rectified flow 的 reflow 过程与普通 DDPM sampling
- 判断 2026 年 fast image/audio/video generation 为什么偏向 flow-matched base + distillation

## The Problem / 问题

DDPM 的 reverse process 是从 `N(0, I)` 回到 data distribution 的 1000-step stochastic walk。DDIM 把它压到 20-50 deterministic steps。你还想更少，最好一步完成。阻碍是求解 reverse process 的 ODE 很 stiff；路径是弯的。

如果能训练模型，让从 noise 到 data 的路径是 *straight line*，那么从 `t=1` 到 `t=0` 的单个 Euler step 就能工作。Flow matching 直接构建这个目标：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练 vector field `v_θ(x, t)` 匹配它的时间导数，推理时积分。

Rectified flow（Liu 2022）更进一步：通过 reflow procedure 迭代拉直路径，得到逐渐接近线性的 ODE。两次 reflow 后，2-step sampler 就能匹配 50-step DDPM 质量。

## The Concept / 概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Straight-Line Flow / 直线 flow

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`，`x_1 ~ N(0, I)`。这条直线上的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义神经 vector field `v_θ(x_t, t)`，训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是 **conditional flow matching** loss（Lipman 2023）。训练不需要模拟：你从不 unroll ODE。只需 sample `(x_0, x_1, t)` 并做 regression。

### Sampling / 采样

推理时，沿时间 *反向* 积分 learned vector field：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，用 Euler step 走到 `t=0`。

### Rectified Flow (Liu 2022) / Rectified flow

Straight-line flow 能工作，但学到的路径 *并不真的直*，因为许多 `x_0` 可能映射到同一个 `x_1`。Rectified flow 的 reflow step：

1. 用 random pairings 训练 flow model v_1。
2. 通过从 `x_1` 积分 v_1 到落点 `x_0`，sample N 对 `(x_1, x_0)`。
3. 在这些 paired examples 上训练 v_2。由于 pairs 现在是 “ODE-matched”，它们之间的 straight-line interpolant 真的更平。
4. 重复。

实践中 2 次 reflow 就能接近 linear，支持 2-4 step inference。SDXL-Turbo、SD3-Turbo、LCM 都是从 flow-matching models 蒸馏来的。

### Why This Won for Images in 2024 / 为什么它在 2024 年赢了图像

三个原因：

1. **Simulation-free training / 训练不需要模拟**：训练时无需 ODE unrolling，实现极简单。
2. **Better loss geometry / 更好的 loss geometry**：直线路径有一致的 signal-to-noise，而 DDPM ε-loss 在 schedule 边缘 SNR 很差。
3. **Faster inference / 更快推理**：SDXL-Turbo 质量下 4-8 steps；配合 consistency distillation 可以 1 step。

## Flow Matching vs DDPM — The Exact Connection / Flow Matching 与 DDPM 的精确关系

带 Gaussian-conditional path 的 flow matching，本质上是 *特定 noise schedule 下的 diffusion*。选择 `x_t = α(t) x_0 + σ(t) x_1` schedule，flow matching 就会恢复 Stratonovich-reformulated diffusion，其中 `v = α'·x_0 - σ'·x_1`。对 Gaussian paths，两者代数上等价。

Flow matching 新增的是：target 更清楚（plain velocity）、loss 更干净，并允许你实验 non-Gaussian interpolants。

## Build It / 动手构建

`code/main.py` 在双峰 Gaussian mixture 上实现 1-D flow matching。Vector field `v_θ(x, t)` 是一个小 MLP，用 straight-line target 训练。推理时分别积分 1、2、4 和 20 个 Euler steps，并比较 sample quality。

### Step 1: training loss / 第 1 步：训练 loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: multi-step inference / 第 2 步：multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: compare step counts / 第 3 步：比较 step counts

预期 4-step sampler 已经接近 20-step quality；这对延迟非常关键。

## Pitfalls / 常见坑

- **Time parameterization。** Flow matching 使用 `t ∈ [0, 1]`，`t=0` 是 data，`t=1` 是 noise。DDPM 使用 `t ∈ [0, T]`，`t=0` 是 data，`t=T` 是 noise。方向相同，尺度不同。论文经常把这里写乱。
- **Schedule choice。** Rectified flow 的 straight line 是 “the” flow-matching schedule，但你可以用 cosine 或 logit-normal t-sampling（SD3 就这么做）覆盖不同 scale。
- **Reflow cost。** 生成 reflow 所需 paired dataset 等于每个 sample 跑一次完整 inference。只有确实需要 1-2 step inference 时才做 reflow。
- **Classifier-free guidance still applies。** 只是把 ε 换成 v 做线性组合：`v_cfg = (1+w) v_cond - w v_uncond`。

## Use It / 应用它

| Use case / 使用场景 | 2026 stack |
|----------|-----------|
| Text-to-image, best quality | Flow matching：SD3、Flux.1-dev |
| Text-to-image, 1-4 steps | Distilled flow matching：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| Real-time inference | 从 flow-matched base 做 consistency distillation（LCM、PCM） |
| Audio generation | Flow matching：Stable Audio 2.5、AudioCraft 2 |
| Video generation | Flow matching mixed with diffusion（Sora、Veo、Stable Video） |
| Science / physics (particle trajectories, molecules) | Flow matching + equivariant vector field |

2025-2026 年，论文里说 “faster than diffusion” 时，几乎总是 flow matching + distillation。

## Ship It / 交付它

保存 `outputs/skill-fm-tuner.md`。Skill 接收 diffusion-style model spec，并转换成 flow-matching training config：schedule choice、time sampling distribution（uniform / logit-normal）、optimizer、reflow plan、target step count、eval protocol。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`，比较 1-step vs 20-step 相对 true data distribution 的 MSE。
2. **Medium / 中等。** 从 uniform `t` sampling 切到 logit-normal（把采样集中到 mid-t）。模型质量是否提升？
3. **Hard / 困难。** 实现一次 reflow iteration：通过积分第一个模型生成 paired (x_0, x_1)，再在这些 pairs 上训练第二个模型，比较 1-step sample quality。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Flow matching | "Straight-line diffusion" | 训练 `v_θ(x, t)` 匹配 interpolant 上的 `x_1 - x_0`。 |
| Rectified flow | "Reflow" | 逐步拉直 learned flows 的迭代过程。 |
| Velocity field | "v_θ" | 模型输出：`x_t` 应移动的方向。 |
| Straight-line interpolant | "The path" | `x_t = (1-t)·x_0 + t·x_1`；target derivative 平凡。 |
| Euler sampler | "1st order ODE solver" | 最简单的 integrator；路径直时效果好。 |
| Logit-normal t | "SD3 sampling" | 把 `t` sampling 集中到 gradients 最强的中间值。 |
| Consistency distillation | "1-step sampler" | 训练 student 把任意 `x_t` 直接映射到 `x_0`。 |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；同一个技巧，变量变了。 |

## Production Note: Flux.1-schnell Is Flow Matching at Its Fastest / 生产备注：Flux.1-schnell 是最快形态的 flow matching

Flow matching 的生产胜利是 Flux.1-schnell：一个 flow-matched DiT 被蒸馏到 1-4 inference steps，同时保留 Flux-dev 级质量。Niels 的 “Run Flux on an 8GB machine” notebook 是 reference deployment recipe：T5 + CLIP encode、quantized MMDiT denoise（schnell 用 4 steps，dev 用 50）、VAE decode。成本账如下：

| Variant / 变体 | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则：**flow-matched base + distillation = 2026 年快速 text-to-image 默认方案。** 每个主要 vendor 都在交付这个组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。纯 diffusion base 只剩 legacy checkpoints。

## Further Reading / 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模 rectified flow。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 覆盖 FM + diffusion 的 general framework。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — diffusion / flow 的 1-step distillation。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo variant。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的 flow matching。
