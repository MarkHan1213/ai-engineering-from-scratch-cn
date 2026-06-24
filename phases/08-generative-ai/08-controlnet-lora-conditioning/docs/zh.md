# ControlNet, LoRA & Conditioning / ControlNet、LoRA 与条件控制

> 纯文本是很笨拙的控制信号。ControlNet 让你克隆一个预训练 diffusion model，并用 depth map、pose skeleton、scribble 或 edge image 去引导它。LoRA 让你只训练 1000 万参数，就能 fine-tune 一个 2B-parameter model。两者一起，把 Stable Diffusion 从玩具变成了 2026 年每家 agency 都在交付的图像 pipeline。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么 text prompt 不能提供足够的 spatial control
- 理解 ControlNet 的 cloned encoder、zero-convolution skip 和可组合推理
- 推导 LoRA 的低秩增量 `W + B @ A` 及其参数量收益
- 为 2026 年图像 pipeline 组合 ControlNet、LoRA、IP-Adapter 和 text conditioning

## The Problem / 问题

像 “a woman in a red dress walking a dog on a busy street” 这样的 prompt，并没有告诉模型狗在 *哪里*，女人是什么 *pose*，街道是什么 *perspective*。文本只能钉住一张图大约 10% 的信息。剩下的是视觉信息，没法高效用语言描述。

为每一种信号（pose、depth、canny、segmentation）从零训练一个新的 conditional model，成本太高。你想保留 frozen 的 2.6B-param SDXL backbone，只接一个读 conditioning 的小 side-network，让它轻推 backbone 的 intermediate features。这就是 ControlNet。

你还想教模型新的 concepts（你的脸、你的产品、你的风格），但不重新训练完整模型。你需要一个小 100x 的 delta。这就是 LoRA：low-rank adapters，插入已有 attention weights。

ControlNet + LoRA + text = 2026 年 practitioner 的工具箱。大多数生产图像 pipeline 会在 SDXL / SD3 / Flux base 上叠 2-5 个 LoRAs、1-3 个 ControlNets 和一个 IP-Adapter。

## The Concept / 概念

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023) / ControlNet

取一个 pretrained SD。*Clone* U-Net 的 encoder half。Freeze 原始模型。训练这个 clone 接受额外 conditioning input（edges、depth、pose）。再用 *zero-convolution* skip connections（初始化为 zero 的 1×1 convs：一开始是 no-op，逐渐学习 delta）把 clone 接回原始模型的 decoder half。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init 意味着 ControlNet 起点是 identity：训练前也不会伤害原模型。用 1M 条（prompt, condition, image）triples 和标准 diffusion loss 训练。

每个 modality 的 ControlNet 都是小 side model（SDXL 约 360M，SD 1.5 约 70M）。推理时可以组合：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021) / LoRA

对模型中的任意 linear layer `W ∈ R^{d×d}`，freeze `W` 并加一个 low-rank delta：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。Attention 常用 rank 4-16，heavy fine-tunes 常用 rank 64-128。新增参数量是 `2 · d · r`，而不是 `d²`。对 `d=640` 的 SDXL attention，`r=16` 时每个 adapter 是 20k params，而不是 410k，减少 20x。整模型上，一个 LoRA 通常 20-200MB，而 base 是 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是常见范围。多个 LoRA 加性堆叠（注意它们仍会非线性相互作用）。

### IP-Adapter (Ye et al., 2023) / IP-Adapter

一个很小的 adapter，接收 *image* 作为 conditioning（与 text 并列）。使用 CLIP image encoder 产生 image tokens，并把它们和 text tokens 一起注入 cross-attention。每个 base model 约 20MB。它允许你做“按这张 reference 的风格生成”，不需要训练 LoRA。

## Composability Matrix / 可组合性矩阵

| Tool / 工具 | What it controls / 控制内容 | Size / 大小 | When to use / 何时使用 |
|------|------------------|------|-------------|
| ControlNet | Spatial structure（pose、depth、edges） | 70-360MB | 精确 layout、composition |
| LoRA | Style、subject、concept | 20-200MB | Personalization、style |
| IP-Adapter | 来自 reference image 的 style 或 subject | 20MB | 文本无法描述外观时 |
| Textual Inversion | 把单个 concept 学成新 token | 10KB | Legacy，基本被 LoRA 替代 |
| DreamBooth | Subject 的 full fine-tune | 2-5GB | 强 identity，高 compute |
| T2I-Adapter | 更轻的 ControlNet alternative | 70MB | Edge devices、inference budget |

ControlNet ≈ spatial。LoRA ≈ semantic。两者一起用。

## Build It / 动手构建

`code/main.py` 在 1-D 中模拟两个机制：

1. **LoRA。** 一个 pretrained linear layer `W`。Freeze 它。训练一个 low-rank `B @ A`，让 `W + BA` 匹配目标 linear layer。展示 `r = 1` 足以完美学习 rank-1 correction。

2. **ControlNet-lite。** 一个 “frozen base” predictor，以及一个读取额外信号的 “side network”。Side network 的输出由一个 learnable scalar gate 控制，gate 初始化为 0（我们的 zero-conv 版本）。训练时观察 gate 如何 ramp up。

### Step 1: LoRA math / 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: zero-init side network / 第 2 步：zero-init side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Step 0 时输出与 base 完全相同。早期训练会缓慢更新 `gate`，避免 catastrophic drift。

## Pitfalls / 常见坑

- **Over-scaling LoRAs。** `α = 2` 或 `α = 3` 是常见的“让它更强”黑招，会产生过度 stylized / broken outputs。保持 `α ≤ 1.5`。
- **ControlNet weight conflict。** Pose ControlNet 用 weight 1.0，同时 Depth ControlNet 也用 weight 1.0，通常会 overshoot。weights 总和 ≈ 1.0 是安全默认值。
- **LoRA on the wrong base。** SDXL LoRA 在 SD 1.5 上会 silent no-op，因为 attention dimensions 不匹配。Diffusers 0.30+ 会警告。
- **Textual Inversion drift。** 在一个 checkpoint 上训练的 token 到另一个 checkpoint 上会 drift 很严重。LoRA 更可移植。
- **LoRA weight-merging and storage。** 你可以把 LoRA bake 进 base model weights 以加速 inference（没有 runtime addition），但会失去 runtime 缩放 `α` 的能力。两个版本都保留。

## Use It / 应用它

| Goal / 目标 | 2026 pipeline |
|------|---------------|
| Reproduce a brand's art style | 用约 30 张 curated images 训练 rank 32 LoRA |
| Put my face in a generated image | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| Specific pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| Exact layout | ControlNet-Scribble 或 ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting（Lesson 09） |
| Fast 1-step style | SDXL-Turbo 上的 LCM-LoRA |

## Ship It / 交付它

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接收任务（input assets：prompt、optional reference image、optional pose、optional depth、optional scribble），并输出 tool stack、weights 和 reproducible seed protocol。

## Exercises / 练习

1. **Easy / 简单。** 在 `code/main.py` 中，把 LoRA rank `r` 从 1 改到 4。到哪个 rank 时 LoRA 能精确匹配 rank-2 target delta？
2. **Medium / 中等。** 分别在两个 target transforms 上训练两个 LoRA。一起加载它们，展示 additive interaction。交互什么时候会破坏线性？
3. **Hard / 困难。** 用 diffusers 叠：SDXL-base + Canny-ControlNet（weight 0.8）+ style LoRA（α 0.8）+ IP-Adapter（weight 0.6）。随着 stack weights 变化，测量 FID-vs-prompt-adherence trade-off。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| ControlNet | "Spatial control" | Cloned encoder + zero-conv skips；读取 conditioning image。 |
| Zero convolution | "Starts as identity" | 初始化为 zero 的 1×1 conv；ControlNet 起点是 no-op。 |
| LoRA | "Low-rank adapter" | `W + B @ A`，`r << d`；参数比 full fine-tune 少 100x。 |
| rank r | "The knob" | LoRA compression；4-16 常见，64+ 用于 heavy personalization。 |
| α | "LoRA strength" | LoRA delta 的 runtime scaling。 |
| IP-Adapter | "Reference image" | 通过 CLIP-image tokens 做 image-conditioning 的小 adapter。 |
| DreamBooth | "Full subject fine-tune" | 在约 30 张 subject 图上训练完整模型。 |
| Textual Inversion | "New token" | 只学习一个新 word embedding；legacy，基本被替代。 |

## Production Note: LoRA Swaps, ControlNet Lanes, Multi-Tenant Serving / 生产备注：LoRA 热切换、ControlNet lanes 与多租户 serving

真实 text-to-image SaaS 会在同一个 base checkpoint 上服务数百个 LoRAs 和十几个 ControlNets。Serving 问题很像 LLM multi-tenancy（production 文献在 continuous batching 和 LoRAX / S-LoRA 下讨论 LLM 情况）：

- **Hot-swap LoRAs, do not merge / 热切换 LoRA，不要 merge。** 把 `W' = W + α·B·A` merge 进 base，每步 inference 快约 3-5%，但会冻结 `α` 和 base。把 LoRAs 作为 rank-r deltas 常驻 VRAM；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`，支持 per-request 激活。Swap cost 是 `2 · d · r · num_layers` 的 weights，MB 级、亚秒级。
- **ControlNet as a second attention lane / ControlNet 是第二条 attention lane。** Cloned encoder 与 base 并行运行。两个 weight 1.0 的 ControlNets = 每个 step 多两次 forward passes，而不是一次 merged pass。Batch-size headroom 会大幅下降。每个 active ControlNet 预算约 ~1.5× step cost。
- **Quantized LoRAs too / LoRA 也量化。** 如果 base 已量化（见 Lesson 07，8GB 上跑 Flux），LoRA delta 也能干净地量化到 8-bit 或 4-bit。QLoRA-style loading 允许你在 4-bit Flux base 上堆 5-10 个 LoRAs，而不爆内存。

Flux-specific：Niels 的 Flux-on-8GB notebook 把 base 量化到 4-bit；在该 quantized base 上叠 style LoRA（`pipe.load_lora_weights("user/style-lora")`，`weight_name="pytorch_lora_weights.safetensors"`）仍能工作。这就是 2026 年多数 SaaS agencies 上线的 recipe。

## Further Reading / 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM，后来移植到 diffusion）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量替代。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — reference pipelines。
