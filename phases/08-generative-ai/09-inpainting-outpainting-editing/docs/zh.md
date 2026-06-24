# Inpainting, Outpainting & Image Editing / Inpainting、Outpainting 与图像编辑

> Text-to-image 负责创造新东西。Inpainting 负责修旧东西。生产里，70% 可计费的图像工作都是编辑：换背景、去 logo、扩画布、重生一只手。Inpainting 是 diffusion 真正赚钱的地方。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 inpainting、outpainting、SDEdit 和 instruction-based image editing
- 理解 proper inpainting model 为什么使用 `noisy_latent | encoded_image | mask` 的 9-channel 输入
- 实现 toy 1-D mask-aware DDPM，并比较 naive inpainting 与真实图像 inpainting 的差距
- 为生产编辑工作流选择 mask generation、base model、CFG 和 QA 策略

## The Problem / 问题

客户发来一张完美产品照，背景里有个碍眼的标牌。你想擦掉标牌，并让其他区域逐像素保持一致。不能从零跑 text-to-image，因为那会改变颜色、光照和产品角度。你只想重新生成 masked region，而且希望它尊重周围上下文。

这就是 inpainting。变体包括：

- **Inpainting。** 在 mask 内重新生成，保持外部像素。
- **Outpainting。** 在 mask 外（或画布外）重新生成，保持内部。
- **Image editing。** 重生成整张图，但保持和原图的 semantic 或 structural fidelity（SDEdit、InstructPix2Pix）。

2026 年每个 diffusion pipeline 都有 inpainting mode。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于同一个原则。

## The Concept / 概念

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### The Naive Approach (and Why It's Wrong) / 朴素做法，以及它为什么不够

用 mask 跑标准 text-to-image。在每个 sampling step，把 unmasked region 的 noisy latent 替换成 clean image 的 forward-diffused 版本。它能工作，但效果差。Boundary artifacts 会渗出来，因为模型不知道 masked region 里应该有什么。

### The Proper Inpainting Model / 正确的 inpainting 模型

训练一个修改后的 U-Net，输入从 4 channels 变成 9 channels：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外 channels 是 VAE-encoded source image 的副本，加上 single-channel mask。训练时随机 mask 图像区域，让模型只 denoise masked region，同时把 unmasked region 作为 clean conditioning signal。推理时，模型能“看到” mask 周围是什么，从而生成 coherent completions。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9-channel（或类似）输入。Diffusers 有 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) — Free Editing / SDEdit：无需重训的编辑

把 source image 加噪到中间某个 `t`，再用新 prompt 从 `t` 反向走到 0。不需要重新训练。起始 `t` 决定 fidelity 和 creative freedom 的权衡：

- `t/T = 0.3` → 几乎和 source 相同，只做小的风格变化
- `t/T = 0.6` → 中等编辑，保留粗结构
- `t/T = 0.9` → 接近从噪声生成，source preservation 很低

### InstructPix2Pix (Brooks et al., 2023) / InstructPix2Pix

在 `(input_image, instruction, output_image)` triples 上 fine-tune diffusion model。推理时同时 condition 输入图像和文本指令（“make it sunset”、“add a dragon”）。有两个 CFG scales：image scale 和 text scale。

### RePaint (Lugmayr et al., 2022) / RePaint

保留标准 unconditional diffusion model。在每个 reverse step 里 resample：偶尔跳回更 noisy 的状态并重新生成。它能减轻 boundary artifacts。适合没有 trained inpainting model 时使用。

## Build It / 动手构建

`code/main.py` 在 5-dimensional data 上实现一个 toy 1-D inpainting scheme。我们在 5-D mixture data 上训练 DDPM，每个 sample 是来自两个 clusters 之一的 5 个 floats。推理时，“mask” 5 维中的 2 维，在每个 step 注入 unmasked 三维的 noisy-forward 版本，只重新生成 masked dimensions。

### Step 1: 5-D DDPM data / 第 1 步：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims / 第 2 步：在全部 5 维上训练 denoiser

标准 DDPM。Net 对 5-D noisy input 输出 5-D noise prediction。

### Step 3: at inference, mask-aware reverse / 第 3 步：推理时做 mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是 naive approach，并且在 toy 1-D data 上有效。真实图像 inpainting 使用 9-channel input，因为 texture coherence 更重要。

### Step 4: outpainting / 第 4 步：outpainting

Outpainting 就是 mask 反过来的 inpainting：mask 新的（原本不存在的）canvas，其余部分用原图填入。训练目标完全相同。

## Pitfalls / 常见坑

- **Seams。** Naive approach 会留下明显边界，因为 gradient info 没有穿过 mask。修复：把 mask 扩张 8-16 pixels，或使用 proper inpainting model。
- **Mask leakage。** Conditioning image 的 unmasked region 如果质量低或有噪声，会污染 mask 内生成。先轻微 denoise 或 blur。
- **CFG interacts with mask size。** 小 mask + 高 CFG = saturated patch。小编辑时降低 CFG。
- **SDEdit fidelity cliff。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能突然丢失 subject identity。Sweep 并 checkpoint。
- **Prompt mismatch。** Prompt 应描述 *整张图*，不是只描述新内容。写 “A cat sitting on a chair”，不要只写 “a cat”。

## Use It / 应用它

| Task / 任务 | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint 或 Flux-Fill，标准 prompt |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode（8px feather）或带 outpaint mask 的 Flux-Fill |
| Regenerate hand / face | SD-Inpaint，prompt 重新描述 subject + ControlNet-Openpose |
| Change style of one region | masked region 上 `t/T=0.5` 的 SDEdit |
| "Make it sunset" | InstructPix2Pix 或 Flux-Kontext |
| Background replacement | SAM mask → SD-Inpaint |
| Ultra-high-fidelity | 最难场景用 Flux-Fill 或 GPT-Image（hosted） |

SAM（Meta's Segment Anything, 2023）+ diffusion inpaint 是 2026 年背景移除 pipeline。SAM 2（2024）可用于视频。

## Ship It / 交付它

保存 `outputs/skill-editing-pipeline.md`。Skill 接收 original image + edit description + optional mask（或 SAM prompt），并输出：mask-generation approach、base model、CFG scales（image + text）、SDEdit-t 或 inpainting mode、QA checklist。

## Exercises / 练习

1. **Easy / 简单。** 在 `code/main.py` 中，把 masked dimensions 比例从 0.2 改到 0.8。到哪个比例时 inpaint quality（masked dims residual）等同于 unconditional generation？
2. **Medium / 中等。** 实现 RePaint：每 10 个 reverse steps，跳回 5 steps（add noise）并重新 denoise。测量它是否降低 mask edge 的 boundary residual。
3. **Hard / 困难。** 用 Hugging Face diffusers 比较：20 个 face-regeneration tasks 上的 SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill。分别给 pose adherence 和 identity preservation 打分。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Inpainting | "Fill the hole" | 在 mask 内重新生成；保留外部像素。 |
| Outpainting | "Extend the canvas" | 画布外重新生成；保留内部。 |
| 9-channel U-Net | "Proper inpainting model" | U-Net 以 `noisy \| encoded-source \| mask` 作为输入。 |
| SDEdit | "Img2img with noise level" | 加噪到 time `t`，再用新 prompt denoise。 |
| InstructPix2Pix | "Text-only edits" | 在 (image, instruction, output) triples 上 fine-tune 的 diffusion。 |
| RePaint | "No retraining" | Reverse 过程中周期性 re-noise，减少 seams。 |
| SAM | "Segment Anything" | 通过 clicks 或 boxes 生成 mask；常与 inpaint 配套。 |
| Flux-Kontext | "Edit with context" | 接收 reference image + instruction 做编辑的 Flux variant。 |

## Production Note: Edit Pipelines Are Latency-Sensitive / 生产备注：编辑 pipeline 对延迟敏感

用户编辑图像时期待 5 秒内往返。30-step SDXL-Inpaint 在 L4 上跑 1024² 约 3-4 s，再加 SAM mask generation（约 200 ms）和 VAE encode/decode（合计约 500 ms）。用生产术语说，这更像 TTFT-bound，而不是 throughput-bound：batch 1、低并发、每个 stage 都要压缩。

- **SAM-H is the slow one / SAM-H 是慢环节。** SAM-H 在 1024² 上约 200 ms；SAM-ViT-B 约 40 ms，质量损失很小。SAM 2（video）有额外 temporal overhead；单图编辑不要用。
- **Skip the encode when possible / 能跳过 encode 就跳过。** `pipe.image_processor.preprocess(img)` 会 encode 成 latents。如果你已经有上一轮生成的 latents（iterative-edit UI 常见），直接通过 `latents=...` 传入，省掉一次 VAE encode。
- **Mask dilation matters for throughput too / mask dilation 也影响 throughput。** 小 mask 意味着大部分 U-Net forward 都浪费了（unmasked pixels 最终会 clamped）。`diffusers` 的 `StableDiffusionInpaintPipeline` 仍然运行完整 U-Net；只有 9-channel proper-inpaint variants 才能利用 masked compute。
- **Flux-Kontext is the 2025 answer / Flux-Kontext 是 2025 年答案。** 在 `(source_image, instruction)` 上单次 forward，不需要单独 mask，也不需要 SDEdit noise sweep。在 H100 上约 1.5 s 出编辑结果。架构教训是：把 stages 合并。

## Further Reading / 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — training-free inpainting。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — text-instruction editing。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，mask source。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention-level editing。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 tooling。
