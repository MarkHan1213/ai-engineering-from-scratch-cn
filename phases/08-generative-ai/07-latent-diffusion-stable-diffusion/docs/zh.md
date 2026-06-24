# Latent Diffusion & Stable Diffusion / Latent Diffusion 与 Stable Diffusion

> 在 512×512 图像上跑 pixel-space diffusion，是一种计算浪费。Rombach et al.（2022）意识到，生成图像不需要全部 786k 维；你只需要足够表达语义结构的 latent，再用独立 decoder 补回细节。把 diffusion 放进 VAE 的 latent space。这个想法就是 Stable Diffusion。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 latent diffusion 为什么把 diffusion 从 pixel space 移到 VAE latent space
- 描述 Stable Diffusion 的 two-stage training、text encoder 和 cross-attention 注入
- 实现 toy latent diffusion，并加入 classifier-free guidance
- 判断 SDXL、SD3、Flux 等 2026 模型在 backbone、latent shape 和 text encoder 上的差异

## The Problem / 问题

512² 的 pixel-space diffusion 意味着 U-Net 在 `[B, 3, 512, 512]` 形状的 tensors 上运行。一个 500M-param U-Net 每个 sampling step 大约 100 GFLOPS。50 steps 就是每张图 5 TFLOPS。在十亿图像上训练，compute bill 会非常夸张。

大部分 FLOPs 都花在把感知上不重要的细节推进网络里，比如一个有损 VAE 本可以压掉的高频纹理。Rombach 的想法是：先训练一个 VAE（*first stage*），freeze 它，再完全在 4-channel 64×64 latent space 中跑 diffusion（*second stage*）。同样的 U-Net，1/16 的 pixels，约 64x 更少 FLOPs，质量却接近。

这就是 Stable Diffusion 配方。SD 1.x / 2.x 用 860M U-Net 处理 `64×64×4` latents，SDXL 用 2.6B U-Net 处理 `128×128×4`，SD3 把 U-Net 换成带 flow matching 的 Diffusion Transformer（DiT）。Flux.1-dev（Black Forest Labs, 2024）发布了 12B-param DiT-MMDiT。它们都跑在同一个 two-stage substrate 上。

## The Concept / 概念

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Two stages, separately trained / 两个阶段，分别训练。**

1. **Stage 1 — VAE。** Encoder `E(x) → z`，decoder `D(z) → x`。目标压缩：每个 spatial axis 下采样 8×，再调整 channels，使 latent 总大小约为 pixel count 的 1/16。Loss = reconstruction（L1 + LPIPS perceptual）+ KL（小权重；因为我们不需要从 `z` 精确采样，不必强迫 `z` 太 Gaussian）。通常还会用 adversarial loss，让 decoded images 更锐利。

2. **Stage 2 — diffusion on `z`。** 把 `z = E(x_real)` 当作 data。训练 U-Net（或 DiT）denoise `z_t`。推理时：通过 diffusion sample `z_0`，再 `x = D(z_0)`。

**Text conditioning / 文本条件。** 另外两个组件：frozen text encoder（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL），以及 cross-attention injection：每个 U-Net block 接收 `[Q = image features, K = V = text tokens]` 并混合。Text tokens 是文本影响图像的唯一通道。

**The loss function is identical to Lesson 06 / Loss 和 Lesson 06 完全相同。** 仍是 DDPM / flow matching 的 noise MSE。你只是换了 data domain。

## Architecture Variants / 架构变体

| Model / 模型 | Year / 年份 | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT（latent patches 上的 transformer）替代 U-Net；扩大 text encoder（T5 比 CLIP 更符合 prompt）；增加 latent channels（4 → 16 给细节更多空间）。

```figure
noise-schedule
```

## Build It / 动手构建

`code/main.py` 把一个 toy 1-D “VAE”（为了演示，identity encoder + decoder；真实 VAE 会是 conv net）叠在 Lesson 06 的 DDPM 上，并加入 classifier-free guidance 的 class conditioning。它展示同一个 diffusion loss 无论跑在原始 1-D values 上，还是 encoded values 上，都能工作。这就是关键洞察。

### Step 1: encoder/decoder / 第 1 步：encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实 VAE 有训练好的 weights。教学上，这个 linear map 足以说明 diffusion 在 `z` 上运行，并不关心原始 data space。

### Step 2: diffusion in `z`-space / 第 2 步：在 `z`-space 中做 diffusion

和 Lesson 06 的 DDPM 相同。网络看到的数据是 `z = E(x)`。采样得到 `z_0` 后，再用 `D(z_0)` decode。

### Step 3: classifier-free guidance / 第 3 步：classifier-free guidance

训练时 10% 概率 drop class label（替换成 null token）。推理时同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无 guidance（保留完整 diversity），`w = 3` = 默认值，`w = 7+` = saturated / over-sharp。

### Step 4: text conditioning (concept, not code) / 第 4 步：文本条件（概念，不写代码）

把 class label 换成 frozen text encoder output。通过 cross-attention 把 text embedding 喂给 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这就是 class-conditional diffusion model 和 Stable Diffusion 的唯一实质差异。

## Pitfalls / 常见坑

- **VAE-scale mismatch。** SD 1.x VAE 在 encode 后会乘一个 scaling constant（`scaling_factor ≈ 0.18215`）。忘记它会让 U-Net 在 variance 完全错误的 latents 上训练。每个 checkpoint 都带一个。
- **Text encoder silently wrong。** SD3 需要 T5-XXL 且 >=128 tokens，退化到 CLIP-only 会损失很大。始终检查 `use_t5=True`，否则 prompt fidelity 会塌。
- **Mixing latent spaces。** SDXL、SD3、Flux 使用不同 VAE。SDXL latents 上训练的 LoRA 不能用于 SD3。Hugging Face diffusers 0.30+ 会拒绝加载 mismatched checkpoints。
- **CFG too high。** `w > 10` 会生成饱和、油腻图像，并以牺牲 diversity 为代价过拟合 prompt。较稳的取值区间是 `w = 3-7`。
- **Negative prompts leaking。** Empty negative prompt 会变成 null token；填了内容的 negative prompt 会变成 `ε_uncond`。这两者不是一回事；有些 pipeline 会悄悄默认 null。

## Use It / 应用它

2026 年生产栈：

| Target / 目标 | Recommended backbone / 推荐 backbone |
|--------|----------------------|
| Narrow domain, paired data, training a model from scratch | SDXL fine-tune（LoRA / full）——最快上线 |
| Open-domain text-to-image, open weights | Flux.1-dev（12B, Apache / non-commercial）或 SD3.5-Large |
| Fastest inference, open weights | Flux.1-schnell（1-4 step, Apache）或 SDXL-Lightning |
| Best prompt adherence, hosted | GPT-Image / DALL-E 3（仍然强）、Midjourney v7、Imagen 4 |
| Edit workflows | Flux.1-Kontext（Dec 2024）——原生接收 image + text |
| Research, baseline | SD 1.5——古老但研究充分 |

## Ship It / 交付它

保存 `outputs/skill-sd-prompter.md`。Skill 接收 text prompt + target style，并输出：model + checkpoint、CFG scale、sampler、negative prompt、resolution、可选 ControlNet/IP-Adapter combo，以及 per-step QA checklist。

## Exercises / 练习

1. **Easy / 简单。** 用 guidance `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每个 class 的 mean sample。到哪个 `w` 时 class means 超过真实数据 means？
2. **Medium / 中等。** 把 toy linear encoder 换成带 reconstruction loss 的 tanh-MLP encoder/decoder pair。重新在新 latents 上训练 diffusion。Sample quality 有变化吗？
3. **Hard / 困难。** 用 diffusers 设置一个真实 Stable Diffusion inference：加载 `sdxl-base`，CFG=7，跑 30 Euler steps，并计时。再切到 `sdxl-turbo`，4 steps，CFG=0。同一个 subject，不同质量；描述变化和原因。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| First stage | "The VAE" | 训练好的 encoder/decoder pair；把 512² 压成 64²。 |
| Second stage | "The U-Net" | Latent space 上的 diffusion model。 |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`；调节 conditioning 强度。 |
| Null token | "Empty prompt embed" | 用于 `ε_uncond` 的 unconditional embed。 |
| Cross-attention | "How text gets in" | 每个 U-Net block 把 text tokens 作为 K 和 V attend。 |
| DiT | "Diffusion Transformer" | 用 latent patches 上的 transformer 替换 U-Net；更易 scaling。 |
| MMDiT | "Multi-modal DiT" | SD3 的架构：text 和 image streams 做 joint attention。 |
| VAE scaling factor | "Magic number" | 把 latents 除以约 5.4，使 diffusion 在 unit-variance space 中运行。 |

## Production Note: Running Flux-12B on an 8GB Consumer GPU / 生产备注：在 8GB 消费级 GPU 上跑 Flux-12B

reference Flux integration 是经典问题“我只有消费级 GPU，能上线吗？”的配方。这个技巧其实就是 production inference 文献里的三旋钮配方，只是应用到 diffusion DiT：

1. **Staggered loading / 交错加载。** Flux 有三个不需要同时存在于 VRAM 的网络：T5-XXL text encoder（fp32 约 10 GB）、CLIP-L（小）、12B MMDiT 和 VAE。先 encode prompt，*delete* encoders，load DiT，denoise，*delete* DiT，load VAE，decode。消费级 8GB GPU 一次只放得下一个 stage。
2. **4-bit quantization via bitsandbytes / 通过 bitsandbytes 做 4-bit quantization。** 对 T5 encoder 和 DiT 使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存砍 8×；按 Aritra benchmark（notebook 中有链接），text-to-image 质量下降不可见。
3. **CPU offload / CPU 卸载。** `pipe.enable_model_cpu_offload()` 会随着每个 forward pass 自动在 CPU 和 GPU 之间交换 modules。增加 10-20% 延迟，但让 pipeline 能跑起来。

内存账是：`10 GB T5 / 8 = 1.25 GB` quantized，`12 B params × 0.5 bytes = ~6 GB` quantized DiT，再加 activations。用 stas00 的术语，这是 TP=1 inference 的极端版本：没有 model parallelism，最大化 quantization。生产里你会在 H100 上跑 TP=2 或 TP=4；单台开发笔记本则用这个 recipe。

## Further Reading / 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述每种 checkpoint 的 reference implementation。
