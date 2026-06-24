# Generative Models — Taxonomy & History / 生成模型：分类与历史

> 每个图像模型、文本模型、视频模型和 3D 模型，都能放进五个桶之一。选错桶，你会和数学硬扛好几周；选对桶，过去十二年的进展会在脑子里自然排成一条线。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释生成模型共同要解决的 `p_data(x)` 采样问题
- 区分显式密度、隐式密度、score-based、flow matching 和 token-based AR 路线
- 用五个问题快速判断一篇新生成模型论文属于哪类方法
- 根据任务、采样速度和评估指标选择 2026 年合适的生成模型家族

## The Problem / 问题

生成模型只做一件事：给定从某个未知分布 `p_data(x)` 抽出的训练样本，输出看起来像是来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白质结构，抽象来看都是同一个问题。

麻烦在于，`p_data` 位于一个有数百万维的空间中（512x512 RGB 图像约 786k 维），样本只落在这个空间里一张很薄的流形上，而你手里可能只有 10M 个样本。暴力估计密度完全不可行。每一种生成模型都是一种折中：把一个难题换成另一个稍微没那么难的难题。

过去十二年里有五个家族活了下来。理解每个家族做了什么折中，你就能明白它为什么在某些任务上赢，又为什么在另一些任务上崩。

## The Concept / 概念

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Explicit density, tractable / 显式密度，且可精确计算。** 把 `log p(x)` 写成一个真的能求值的和式。Autoregressive models（PixelCNN、WaveNet、GPT）把 `p(x) = ∏ p(x_i | x_<i)` 分解成条件概率。Normalizing flows（RealNVP、Glow）把 `p(x)` 建成简单基础分布的可逆变换。优点：精确 likelihood，训练 loss 干净。缺点：autoregressive inference 是串行的（长序列很慢），flows 需要可逆架构（架构限制很强）。

**2. Explicit density, approximate / 显式密度，但近似优化。** 从下界约束 `log p(x)`（ELBO），然后优化这个下界。VAEs（Kingma 2013）使用 encoder-decoder 和 variational posterior。Diffusion models（DDPM, Ho 2020）训练 denoiser，隐式优化一个加权 ELBO。到 2026 年，diffusion 是图像、视频和 3D 的主干路线。

**3. Implicit density / 隐式密度。** 完全跳过密度；学习一个 generator `G(z)` 生成样本，再学习一个 discriminator `D(x)` 判断真假。GANs（Goodfellow 2014）。推理快（一次 forward pass），但训练出了名不稳定。即使到 2026 年，StyleGAN 1/2/3 在固定领域 photorealism（人脸、卧室）上仍是一线方案。

**4. Score-based / continuous-time / 基于 score 的连续时间模型。** 直接学习 log-density 的梯度 `∇_x log p(x)`，也就是 score。Song & Ermon（2019）证明 score matching 可以把 diffusion 推广成 SDE。Flow matching（Lipman 2023）是 2024-2026 年的热点：训练不需要模拟，路径更直，采样比 DDPM 快 4-10 倍。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. Token-based autoregressive over discrete codes / 离散 code 上的 token-based autoregressive。** 用 VQ-VAE 或 residual quantizer 把高维数据压成短的离散 token 序列，再用 Transformer 建模这个 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都属于这一类。这其实是第 1 类加上一个 learned tokenizer。

## A Brief History / 简史

| Year / 年份 | Model / 模型 | Why it mattered / 重要性 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个拥有可用训练 loss 的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度、没有 likelihood，却能生成惊人锐利的样本。 |
| 2015 | DRAW, PixelCNN | 顺序式图像生成。 |
| 2017 | Glow, RealNVP | 可逆 flows；用深度换来精确 likelihood。 |
| 2017 | Progressive GAN | 第一次生成百万像素级人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 在人脸这个单一领域中，photorealistic 质量至今很难击败。 |
| 2020 | DDPM (Ho) | Diffusion 变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image 进入主流视野。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning 变成通用商品。 |
| 2022 | ControlNet, LoRA | 对预训练 diffusion 做精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 规模提升，训练动力学更好。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion；flow matching 胜出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级视频生成。 |
| 2026 | Consistency + Rectified Flow | 从 diffusion backbone 上做一步采样。 |

## The Five-Question Triage / 五问分诊法

当一篇新的生成模型论文出现时，读 method section 之前先回答这五个问题。

1. **What is being modeled? / 建模对象是什么？** Pixels、latents、discrete tokens、3D Gaussians、meshes、waveforms？
2. **Is the density explicit or implicit? / 密度是显式还是隐式？** 它有没有写出 `log p(x)`？
3. **Sampling: one-shot or iterative? / 采样是一次完成还是迭代？** 迭代意味着推理更慢；一次完成通常意味着 adversarial 或 distilled。
4. **Conditioning: unconditional, class, text, image, pose? / 条件是什么？** 这决定 loss 和架构支架。
5. **Evaluation: FID, CLIP score, IS, human preference, task accuracy? / 怎么评估？** 每个指标都有已知失效模式（见 Lesson 14）。

本 Phase 的每一课都会重新回答这五个问题。到最后，它会变成你的本能反应。

## Build It / 动手构建

本课代码是一个轻量可视化：从样本拟合一个 1-D mixture-of-Gaussians，分别使用三种玩具方法（kernel density、离散 histogram、nearest-sample “GAN-ish” generator），让你在一个屏幕里看清 explicit density 和 implicit density 的区别。

运行 `code/main.py`。它从双峰 Gaussian mixture 中抽 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两种允许你问“这个点有多可能？”第三种不能。这就是 *explicit vs implicit* 的区别，后续每一课都会用到。

## Use It / 应用它

2026 年，什么任务该选哪个家族？

| Task / 任务 | Best family / 最佳家族 | Why / 原因 |
|------|-------------|-----|
| Photoreal faces, narrow domain | StyleGAN 2/3 | 仍然最锐利，推理最快。 |
| General text-to-image | Latent diffusion + flow matching | SD3、Flux.1、DALL-E 3。 |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo、SD3-Turbo、LCM。 |
| Text-to-video | Diffusion Transformer + flow matching | Sora、Veo 2、Kling。 |
| Speech + music | Token-based AR (AudioLM, VALL-E, MusicGen) or flow matching (AudioCraft 2) | Discrete tokens 扩展便宜。 |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS 做重建，diffusion 做 novel-view。 |
| Density estimation (no sampling) | Flows | 唯一能精确给出 `log p(x)` 的家族。 |
| Simulation / physics | Flow matching, score SDE | 直线路径，平滑 vector fields。 |

## Ship It / 交付它

保存为 `outputs/skill-model-chooser.md`。

这个 skill 接收一个任务描述，并输出：(1) 应使用哪个家族，(2) 三个 open options 和三个 hosted options 的排序列表，(3) 你应该关注的可能失效模式，(4) compute/time budget。

## Exercises / 练习

1. **Easy / 简单。** 对下面五个产品，识别它们的家族和 backbone：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **Medium / 中等。** 你明天要读的一篇论文声称采样比 diffusion 快 100x。写下三个问题，检查这个加速在 conditioning 和高分辨率下是否仍然成立。
3. **Hard / 困难。** 选择一个你关心的领域（例如 protein structure、CAD、molecules、trajectories）。对该领域当前 SOTA 模型回答五问分诊，并草拟一个更好模型会改变什么。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Generative model | “It makes new stuff” | 学习 `p_data(x)` 的 sampler，并可选择暴露 `log p(x)`。 |
| Explicit density | “You can evaluate it” | 模型提供 closed-form 或可 tractable 求值的 `log p(x)`。 |
| Implicit density | “GAN-style” | 只有 sampler，无法评估给定点的 `p(x)`。 |
| ELBO | “Evidence lower bound” | `log p(x)` 的可 tractable 下界；VAEs 和 diffusion 都优化它。 |
| Score | “Gradient of log-density” | `∇_x log p(x)`；diffusion 和 SDE 模型学习这个场。 |
| Manifold hypothesis | “Data lives on a surface” | 高维数据集中在低维流形上；这解释了为什么降维有效。 |
| Autoregressive | “Predict the next piece” | 把 joint 分布分解成条件分布的乘积。 |
| Latent | “Compressed code” | 低维表示，decoder 可以从中重建输入。 |

## Production Note: Five Families, Five Inference Shapes / 生产备注：五个家族，五种推理形态

每个家族对应不同的 inference-server 成本曲线。production-inference 文献把 LLM inference 拆成 prefill + decode；同样的拆法也适用于这里：

- **Autoregressive（第 1 和第 5 类）。** 顺序 decode 主导延迟；KV-cache、continuous batching 和 speculative decoding 都可以直接使用。
- **VAE / diffusion / flow-matching（第 2 和第 4 类）。** 这里没有 LLM 意义上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整 latent resolution 上做一次 transformer 或 U-Net forward。生产旋钮是 step count（DDIM / DPM-Solver / distillation）、batch size 和 precision（bf16 / fp8 / int4）。
- **GAN（第 3 类）。** 一次 forward pass。没有 schedule，没有 KV-cache。TTFT ≈ total latency。这就是为什么 StyleGAN 在窄领域 UX 上仍然赢。

当你在论文摘要里看到 “faster than diffusion”，把它翻译成 “fewer steps × same step cost” 或 “same steps × cheaper step cost”。其余大多是营销话术。

## Further Reading / 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — 把 diffusion 写成 SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching 论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3。
