# Emu3: Next-Token Prediction for Image and Video Generation / Emu3：用 Next-Token Prediction 做图像与视频生成

> BAAI 的 Emu3（Wang et al., 2024 年 9 月）是 2024 年本应终结 diffusion-versus-autoregressive 争论的结果。一个 Llama-style decoder-only transformer，只用 next-token-prediction objective，在 text + VQ image tokens + 3D VQ video tokens 的 unified vocabulary 上训练，就能在图像生成上超过 SDXL，在 perception 上超过 LLaVA-1.6。没有 CLIP loss，没有 diffusion schedule。推理时为了质量使用 classifier-free guidance，但核心训练目标是带 teacher forcing 的 next-token prediction。论文发表在 Nature。本课读取 Emu3 thesis：为什么更好的 tokenizer 加 scale 就足够，并与 diffusion 路线对比。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites / 前置知识：** Phase 12 · 11 (Chameleon)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 解释为什么 Emu3 的 single-loss next-token objective 能工作，尽管长期共识认为高质量图像需要 diffusion。
- 描述 3D video tokenizer：spatiotemporal VQ codebook 是什么，为什么 patch 跨越时间。
- 从 training compute、inference cost、quality ceiling 三个角度比较 Emu3 与 Stable Diffusion XL。
- 说出同一个 Emu3 模型的三个角色：Emu3-Gen（image gen）、Emu3-Chat（perception）、Emu3-Stage2（video gen）。

## The Problem / 问题

2024 年前的传统观点是：图像生成需要 diffusion。理由是 discrete image tokens 丢失太多重建细节，而且 autoregressive sampling 会在数千 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种 diffusion。Chameleon（Lesson 12.11）在小规模上部分反驳了这一点，但质量没有达到 SDXL。

Emu3 正面攻击这个论点。它声称：更好的 visual tokenizer + 足够规模 + next-token loss = 在同一个模型里做出超过 diffusion 的图像生成和 perception。

发布时这个赌注很有争议。两年后，open-source unified-generation family（Emu3、Show-o、Janus-Pro、Transfusion）已成为研究默认路径；生产级 frontier models 很可能也使用某种变体。

## The Concept / 概念

### The Emu3 tokenizer / Emu3 tokenizer

关键材料是 visual tokenizer。Emu3 训练了自定义 IBQ-class tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN family），每个 token 对应 8x8 resolution reduction。512x512 图像变成 64x64 = 4096 tokens，codebook size 32768。

这比 Chameleon 的 512x512 图像 1024 tokens、K=8192 更大，但每 token 更便宜（更小的 codebook lookups、更简单 codec）。关键指标是：reconstruction PSNR 30.5 dB，接近 Stable Diffusion continuous latent space 的 32 dB。

视频方面：3D VQ tokenizer 把 spatiotemporal patch（4x4x4 pixels）编码成一个 integer。4 秒 clip、8 FPS 有 32 帧；256x256，4x spatial 和 4x temporal reduction 时，token count 是 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokens。

Tokenizer quality 是上限。Emu3 的贡献一部分就是“我们训练了一个非常好的 tokenizer”。

### Single-loss training / 单一损失训练

Emu3 只使用一个目标：shared vocabulary 上的 next-token prediction，覆盖 text tokens、2D image tokens 和 3D video tokens。训练时会用 modality-specific factors 调整权重以平衡贡献，但 loss function 相同。

训练混合：

- Image gen：`<text caption> <image> image_tokens </image>`
- Image perception：`<image> image_tokens </image> <question> text_tokens`
- Video gen：`<text caption> <video> video_tokens </video>`
- Video perception：类似。
- Text only：标准 NTP。

模型从数据分布中学习何时输出 image tokens、何时输出 text tokens。生成来自模型在 `<image>` tag 后预测 image tokens。

### Classifier-free guidance and temperature / CFG 与 temperature

Autoregressive image generation 在推理时使用 classifier-free guidance（CFG）会好很多。Emu3 也使用它：生成两次 logits，一次带完整 caption，一次空 caption，然后用 guidance weight 混合（典型 3.0-7.0）。这是 diffusion 的 CFG 技巧，借到 autoregressive setting。

Temperature 也重要：太高会有 artifacts，太低会 mode collapse。Emu3 推荐 perception 使用 1.0，image generation 使用 0.8。

### Three roles, one model / 一个模型，三个角色

Emu3 以三个功能性 API 形式发布，但底层是同一组权重：

- Emu3-Gen。图像生成。输入 text，输出 image tokens。
- Emu3-Chat。VQA 与 captioning。输入 image（tokens），输出 text。
- Emu3-Stage2。视频生成与视频 VQA。输入 text 或 video，输出 text 或 video。

没有 task-specific heads。只是不同 prompt templates。同一个 checkpoint。

### Benchmarks / 基准结果

Emu3 论文（2024 年 9 月）报告：

- Image generation：在 MJHQ-30K FID 上超过 SDXL（5.4 vs 5.6），GenEval overall 基本持平（0.54 vs 0.55），Deep-Eval composite 接近。
- Image perception：VQAv2 上超过 LLaVA-1.6（75.1 vs 72.4），MMMU 上大致持平。
- Video generation：4 秒 clip 质量，在 FVD 上与当时公开 benchmark 的 Sora-era 模型竞争。

数字不总是赢，但“next-token prediction is all you need” 在多模态上变得可辩护。

### Compute cost / 计算成本

Emu3 用 7B 参数模型在约 300B multimodal tokens 上训练。GPU-hours 大致接近 Llama-2-7B 预训练（A100-class silicon 上 2k-4k GPU-years）。Stable Diffusion 3 之类 diffusion model 训练预算相近，但需要独立 text encoders 和更复杂 pipelines。

推理时，Emu3 每张图比 SDXL 慢：512x512 图像 4096 tokens，如果 30 tok/s，约 2 分钟；SDXL 是 2-5 秒。Speculative decoding 和 KV-cache optimization 能缩小差距，但不能完全消除。Autoregressive image generation 计算很重，这是持续的 trade-off。

### Why it matters / 为什么重要

Emu3 的深层贡献是概念性的。如果 next-token prediction 可以在 image generation 上 scale 到 diffusion 质量，那么 unified-model path（一个 loss、一个 backbone、任意 modality）就是可行的。未来模型不一定需要独立 text encoder、独立 diffusion scheduler、独立 VAE。一个 transformer，每个 modality 一个 tokenizer，再 scale。

Show-o、Janus-Pro 和 InternVL-U 都基于或挑战这个 thesis。到 2025 年，中文实验室（BAAI、DeepSeek）在这个方向上发表更积极。

## Build It / 动手构建

本课动手构建两个 toy 组件：一个 2D/3D tokenizer token-count calculator，用来理解图像与视频 token 规模；一个带 classifier-free guidance 的 autoregressive image-token sampler，用来把 Emu3 的推理逻辑拆开。

## Use It / 应用它

`code/main.py` 构建两个 toy pieces：

- 2D vs 3D VQ tokenizer count calculator：给定 resolution、patch、clip_length、FPS，计算 image vs video token counts。
- 带 classifier-free guidance 和 temperature 的 autoregressive image-token sampler。

CFG 实现匹配 Emu3 recipe：用 guidance weight 混合 conditional 和 unconditional logits。

## Ship It / 交付它

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定 generation product spec（image 或 video、target resolution、quality tier、latency budget），它会计算 token counts、inference cost，并在 Emu3-family 与 diffusion 之间做选择。

## Exercises / 练习

1. Emu3 在 8x8 reduction 下，每张 512x512 图像产生 4096 tokens。计算 1024x1024 和 2048x2048 的等价 token count。推理 latency 会怎样？

2. 阅读 Emu3 Section 3.3 关于 video tokenizer。描述 3D VQ patch shape，并解释为什么是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance weight 5.0 vs 3.0 会有什么视觉效果？沿着 `code/main.py` 追踪数学。

4. 计算 Emu3-7B 在 300B tokens 上的 training FLOPs，并与 Stable Diffusion 3 对比。哪个训练更贵？

5. Emu3 在 FID 上超过 SDXL，但 VQAv2 上不如 specialized VLM。解释 unified-loss approach 为什么在不同 benchmark 上表现出不同优势。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Next-token prediction | “NTP” | 标准 autoregressive loss：给定 token[0..i] 预测 token[i+1]；当 modality 被 tokenize 后适用于所有模态 |
| IBQ tokenizer | “Inverse bottleneck quantizer” | 一类 VQ-VAE，使用更大 codebook（32768+）并比 Chameleon reconstruction 更好 |
| 3D VQ | “Spatiotemporal quantizer” | 按 `(time, row, col)` 索引的 codebook；一个 token 覆盖 4x4x4 pixel cube |
| Classifier-free guidance | “CFG” | 用 weight gamma 混合 conditional 与 unconditional logits；推理时提升图像质量 |
| Unified vocabulary | “Shared tokens” | Text + image + video 都来自同一整数空间；模型预测下一个属于任何 modality 的 token |
| MJHQ-30K | “Image gen benchmark” | 30k prompts 的 Midjourney-quality benchmark；Emu3 在这里报告 FID |

## Further Reading / 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
