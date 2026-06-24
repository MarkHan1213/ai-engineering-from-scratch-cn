# Video Generation / 视频生成

> 图像是 2-D tensor。视频是 3-D tensor。理论相同，计算难度高 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明这条路可行。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已能以 1080p 从文本生成生产视频，而 open-weights stack（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）大约落后 12 个月。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释视频生成为什么需要 spatiotemporal compression、temporal coherence 和更大的 compute budget
- 理解 video VAE、patchify、spatiotemporal DiT 和 factorized attention 的分工
- 用 toy sequence 模拟联合 denoising 如何减少 flicker
- 根据质量、开源权重、I2V、角色一致性和音频需求选择 2026 年视频模型

## The Problem / 问题

10 秒 1080p、24fps 的视频有 240 帧，每帧 1920×1080×3 pixels。每个 clip 原始数据约 1.5 GB。Pixel-space diffusion 不可行。你需要：

1. **Spatiotemporal compression / 时空压缩。** 一个编码视频而不是单帧的 VAE，把视频压成 spatial-temporal patches。
2. **Temporal coherence / 时间一致性。** 多帧需要共享内容、光照和 object identity。网络必须建模 motion。
3. **Compute budget / 计算预算。** 在相同 model size 下，视频训练比图像贵 10-100 倍。
4. **Conditioning / 条件控制。** Text、image（first-frame）、audio 或另一个 video。大多数生产模型四种都接收。

解决这个问题的架构是：把 **Diffusion Transformer (DiT)** 应用到 spatiotemporal patches，并在巨大的（prompt, caption, video）数据集上训练。Loss 和 Lesson 06 的 diffusion loss 相同。

## The Concept / 概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify / 切 patch

用 3D VAE（learned spatiotemporal compression）编码视频。Latent 形状是 `[T_latent, H_latent, W_latent, C_latent]`。再切成 `[t_p, h_p, w_p]` 大小的 patches。Sora-style 模型里，`t_p = 1`（per-frame patches）或 `t_p = 2`（每两帧）。10 秒 1080p 视频压缩后大约是 20,000-100,000 patches。

### Spatiotemporal DiT / 时空 DiT

Transformer 处理展平后的 patch 序列。每个 patch 有 3D positional embedding（time + y + x）。Attention 通常 factorized：

- **Spatial attention / 空间注意力**：在每一帧的 patches 内部做 attention。
- **Temporal attention / 时间注意力**：在相同空间位置的跨帧 patches 上做 attention。
- **Full 3D attention / 完整 3D 注意力**：贵 16-100 倍；只在低分辨率或研究中使用。

### Text Conditioning / 文本条件

使用大型 text encoder 做 cross-attention（Sora 用 T5-XXL，CogVideoX-5B 也用 T5-XXL）。长 prompt 很重要：Sora 的训练集使用 GPT 生成 dense re-captions，平均每个 clip 约 200 tokens。

### Training / 训练

在 spatiotemporal latents 上做标准 diffusion loss（ε 或 v prediction）。数据：web video + 约 100M curated clips + synthetic text captions。Compute：即使是小型研究 run，也需要 10,000+ GPU hours；Sora-scale 是 100,000+。

## The 2026 Production Landscape / 2026 年生产格局

| Model / 模型 | Date / 日期 | Max duration | Max res | Open weights? | Notable / 特点 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | 首个在 scale 上展示 world simulator properties 的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | No | 生产版 Sora，推理快 5x |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | 2025 年最高质量 + physics |
| Veo 3 | 2025 Q3 | 15s | 4K | No | Native audio 和更强 camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | 2025 Q1 最强 human motion |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | 上层有专业视频工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | 最强 character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | 首个开放的 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | 2024 年末 open SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | 许可证最宽松 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | 2025 年中最强 open model |

Open weights 在视频空间追赶得比图像更快：到 2026 年中，HunyuanVideo + WAN 2.2 LoRAs 已经支撑大多数开源 workflows。

## Build It / 动手构建

`code/main.py` 模拟核心 spatiotemporal DiT 思路：把一个小 synthetic video patchify，加入 per-patch position embedding，再用 transformer-style attention 在 patches 上对整个序列 denoise。不用 numpy，纯 Python。我们展示即便在 1-D 中，当相邻帧 patches 共享 denoiser 和 position embeddings 时，temporal coherence 也会出现。

### Step 1: patchify a synthetic 1-D "video" / 第 1 步：切分一个 synthetic 1-D “video”

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: position embedding per frame / 第 2 步：每帧 position embedding

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: denoiser sees the whole sequence / 第 3 步：denoiser 看到完整序列

我们的小网络不独立 denoise 每一帧，而是 concat 所有 frame values + position embeddings，并预测所有帧的 noise。

### Step 4: temporal coherence test / 第 4 步：时间一致性测试

训练后 sample 一个 video。测量 frame-to-frame delta。如果模型学到了 temporal structure，deltas 应该小于逐帧独立采样。

## Pitfalls / 常见坑

- **Independent per-frame sampling = flicker。** 如果每帧单独跑 image diffusion，输出会闪，因为每帧 noise 独立。Video diffusion 通过 attention 或 shared noise 绑定帧来解决。
- **Naive 3D attention = OOM。** 在 10 秒 1080p latent 上做 full 3D attention，需要数千亿 operations。要 factorize 成 spatial + temporal。
- **Data captioning matters more than size。** Sora 相比早期工作的主要升级，是用约 10x 更详细 captions 训练（GPT-4 重新标注 clips）。OpenAI 技术报告对此说得很明确。
- **First-frame conditioning。** 大多数生产模型也接收一张 image 作为第一帧。这是 “image-to-video” mode；训练也包含这个变体。
- **Physics drift。** 长 clips（>10s）会积累细微不一致。Sliding-window generation + keyframe anchoring 有帮助。

## Use It / 应用它

| Use case / 使用场景 | 2026 pick |
|----------|-----------|
| Highest-quality text-to-video, hosted | Veo 3 或 Sora |
| Camera-controlled cinematic | Runway Gen-3 with motion brushes |
| Character consistency across clips | Pika 2.0 或 Kling 2.1 |
| Open weights, fast fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| Audio-to-video lip sync | Veo 3（native audio）或 dedicated lip-sync model |
| Video editing | Runway Act-Two、Kling Motion Brush、Flux-Kontext（still-frame） |

在相同质量下，每秒视频的成本从 2024 到 2026 已经下降 20x。

## Ship It / 交付它

保存 `outputs/skill-video-brief.md`。Skill 接收 video brief（duration、aspect ratio、style、camera plan、subject consistency、audio），并输出：model + hosting、prompt scaffolding（camera language、subject description、motion descriptors）、seed + reproducibility protocol 和 frame-level QA checklist。

## Exercises / 练习

1. **Easy / 简单。** 在 `code/main.py` 中比较 (a) independent per-frame sampling、(b) joint sequence sampling 的 frame-to-frame delta。报告 delta 的 mean 和 variance。
2. **Medium / 中等。** 加入 first-frame condition：把 frame 0 固定为给定值，sample 其余帧。测量 pinned value 如何传播。
3. **Hard / 困难。** 用 HuggingFace diffusers 在本地 GPU 跑 CogVideoX-2B。对 6 秒 720p clip 计时 20 inference steps。Profile spatiotemporal attention，定位 bottleneck。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | 把 `(T, H, W, C)` 压缩成 spatiotemporal latent 的 encoder。 |
| Patches | "The tokens" | Latent 的固定大小 3-D blocks；DiT 的输入。 |
| Factorized attention | "Spatial + temporal" | 先在 space 上做 attention，再在 time 上做；跳过 full 3-D attention。 |
| Image-to-video (I2V) | "Animate this photo" | 模型接收 image + text，输出从它开始的视频。 |
| Keyframe conditioning | "Anchor frames" | 固定特定帧来控制视频 arc。 |
| Motion brush | "Directional hint" | UI 输入，用户在图像上绘制 motion vectors。 |
| Re-captioning | "Dense captions" | 用 LLM 重新给训练 clips 标注详细 prompts。 |
| Flicker | "Temporal artifact" | 帧间不一致；通过 coupled denoising 修复。 |

## Production Note: Video Latents Are a Memory-Bandwidth Problem / 生产备注：video latents 是 memory-bandwidth 问题

10 秒 1080p、24 fps 的 clip 是 240 frames × 1920 × 1080 × 3 ≈ 1.5 GB 原始 pixels。经过 4× video VAE compression（`2 × spatial × 2 × temporal`）后，每个 request 的 latent 约 100 MB。把它送进 spatiotemporal DiT，在 batch 1 下跑 30 steps，每 step 都要在 HBM 中移动约 3 GB：瓶颈是 memory bandwidth，不是 FLOPs。

三个生产旋钮都直接来自 production-inference 文献的 inference 章节：

- **TP across the DiT / 对 DiT 做 TP。** Text-to-video 模型经常 ≥10B params。4 张 H100 上 TP=4 是标准；405B-class 模型用 PP=2 × TP=2。直到 all-reduce wall 之前，每步 latency 大致随 TP 线性下降。
- **Frame batching = continuous batching / Frame batching 就是 continuous batching。** 生成时，视频概念上是一批被 attention 连接的 frames。Continuous batching（in-flight scheduling）适用：如果架构支持 sliding-window generation，可以在 frame `t-1` 返回时开始渲染 frame `t+1`。
- **Clip-level prefill cache / Clip 级 prefill cache。** 对 image-to-video，first-frame conditioning 类似 LLM 的 prompt prefill：计算一次，并在 temporal decoder passes 中复用。这本质上是视频的 KV-cache。

## Further Reading / 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中 open SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性 video diffusion 论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的祖先。
