# Evaluation — FID, CLIP Score, Human Preference / 评估：FID、CLIP Score 与人类偏好

> 每个生成模型 leaderboard 都会引用 FID、CLIP score 和某个人类偏好 arena 的 win rate。每个数字都有可以被认真研究者钻空子的失效模式。如果你不知道这些失效模式，就分不清真正提升和刷指标。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分 sample quality、conditioning adherence 和 preference 三类评估目标
- 推导 FID、CLIP score 和 Elo preference aggregation 的计算方式
- 识别小样本 FID、CLIP blind spots、prompt cherry-pick 和 LLM judge hacking 等失效模式
- 设计一个 2026 年生产生成模型评估报告的最小与推荐 protocol

## The Problem / 问题

生成模型要被评价 *sample quality* 和 *conditioning adherence*。两者都没有 closed-form measure。你的模型必须渲染 10,000 张图；某个东西必须给它们打分；你还要相信这些数字能跨 model families、resolutions 和 architectures 比较。2014-2026 年的大浪淘沙后，活下来的三个指标是：

- **FID (Fréchet Inception Distance)。** 在 Inception network feature space 中，比较 real 和 generated 两个分布的距离。越低越好。
- **CLIP score。** 生成图的 CLIP-image embedding 与 prompt 的 CLIP-text embedding 之间的 cosine similarity。越高越好。衡量 prompt adherence。
- **Human preference。** 用同一 prompt 让两个模型 head-to-head，让 humans（或 GPT-4-class model）选更好的，再聚合成 Elo score。

你还会看到：IS（inception score，基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个指标都修正了前一个指标的一个失败点。

## The Concept / 概念

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — Sample Quality / FID：样本质量

Heusel et al.（2017）。步骤：

1. 对 N 张 real images 和 N 张 generated images 提取 Inception-v3 features（2048-D）。
2. 对每个池子拟合 Gaussian：计算 mean `μ_r, μ_g` 和 covariance `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：feature space 中两个 multivariate Gaussians 的 Fréchet distance。越低 = 分布越相似。

失效模式：
- **Biased on small N / 小样本有偏。** FID 是 feature distribution 上的 mean-squared，小 N 会低估 covariance，给出虚假的低 FID。始终使用 N ≥ 10,000。
- **Inception-dependent / 依赖 Inception。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的 domains（人脸、艺术、文字图像）FID 可能无意义。使用 domain-specific feature extractor。
- **Gaming / 可被刷。** 过拟合 Inception prior 可以降低 FID，但视觉质量未必提升。用 CMMD（见下）对抗。

### CLIP Score — Prompt Adherence / CLIP score：prompt 遵循度

Radford et al.（2021）。对 generated image + prompt：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

对 30k generated images 求平均，得到可在模型间比较的 scalar。

失效模式：
- **CLIP's own blind spots / CLIP 自身盲点。** CLIP 组合推理弱（“a red cube on a blue sphere” 经常失败）。模型可以在 CLIP score 上排名很好，但并没有真的遵循复杂 prompts。
- **Short prompt bias / 短 prompt 偏置。** 短 prompts 在野外有更多 CLIP-image matches。长 prompts 的 CLIP score 会机械性变低。
- **Prompt gaming / prompt 刷分。** 在 prompt 里加 “high quality, 4k, masterpiece” 会抬高 CLIP score，但不改善 image-text binding。

CMMD（Jayasumana et al., 2024）修复了其中一部分：使用 CLIP features 而不是 Inception，使用 maximum-mean discrepancy 而不是 Fréchet。更擅长发现微妙质量差异。

### Human Preference — The Ground Truth / 人类偏好：地面真值

取一组 prompts。用 model A 和 model B 生成。把 pairs 展示给 humans（或强 LLM judge）。把 wins 聚合成 Elo 或 Bradley-Terry score。Benchmarks：

- **PartiPrompts (Google)**：1,600 个多样 prompts，12 类。
- **HPSv2**：107k human annotations，广泛用作 automated proxy。
- **ImageReward**：137k prompt-image preference pairs，MIT-licensed。
- **PickScore**：在 Pick-a-Pic 2.6M preferences 上训练。
- **Chatbot-Arena-style image arenas**：https://imagearena.ai/ 等。

失效模式：
- **Judge variance / 评审方差。** 非专家和专家的偏好不同。两者都要用。
- **Prompt distribution / prompt 分布。** Cherry-picked prompts 会偏向某类模型。必须记录。
- **LLM-judge reward hacking / LLM judge 被刷。** GPT-4-judge 会被漂亮但错误的输出欺骗。要和 human triangulate。

## Use Together / 组合使用

生产 eval report 应包括：

1. 在 held-out real distribution 上对 10-30k samples 计算 FID（sample quality）。
2. 在同一批 samples 和 prompts 上计算 CLIP score / CMMD（adherence）。
3. 与上一版模型做 blinded arena win rate（overall preference）。
4. Failure mode analysis：随机抽 50 个 outputs，针对已知问题标注（hand anatomy、text rendering、consistent object count）。

任何单一指标都是谎言。三个互相印证的 metrics + qualitative review 才能构成 claim。

## Build It / 动手构建

`code/main.py` 在 synthetic “feature vectors” 上实现 FID、CLIP-score-like 和 Elo aggregation（我们用 4-D vectors 代替 Inception features）。你会看到：

- 小 N 和大 N 上的 FID computation，以及 bias。
- 用 feature pools 的 cosine similarity 模拟 “CLIP score”。
- 从 synthetic preference stream 做 Elo update。

### Step 1: FID in four lines / 第 1 步：四行 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Step 2: CLIP-style cosine-similarity / 第 2 步：CLIP-style cosine-similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Step 3: Elo aggregation / 第 3 步：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Pitfalls / 常见坑

- **FID at N=1000。** N<10k 时启发式不可靠。报告 low-N FID 的论文是在刷。
- **Comparing FID across resolutions。** Inception 的 299×299 resize 会改变 feature distribution。只在 matched resolution 下比较。
- **Reporting one seed。** 至少跑 3 seeds。报告 std。
- **CLIP score inflation via negative prompts。** 有些 pipeline 通过过拟合 prompt 抬高 CLIP。检查视觉 saturation。
- **Elo bias from prompt overlap。** 如果两个模型训练时都见过 benchmark prompt，Elo 无意义。使用 held-out prompt sets。
- **Human eval paid-crowd skew。** Prolific、MTurk annotators 偏年轻和 tech-friendly。混入招募的 art/design experts。

## Use It / 应用它

2026 年生产 eval protocol：

| Pillar / 支柱 | Minimum / 最小要求 | Recommended / 推荐 |
|--------|---------|-------------|
| Sample quality | 10k vs held-out real 的 FID | + 5k 上的 CMMD + 按 category 子集计算 FID |
| Prompt adherence | 30k 上的 CLIP score | + HPSv2 + ImageReward + VQA-style question answering |
| Preference | 与 baseline 做 200 对 blinded pairs | + 2000 paired human + LLM-judge + Chatbot Arena |
| Failure analysis | 50 个 hand-flagged | 500 个 hand-flagged + automated safety classifier |

四个 pillars 在一份报告里 = claim。任何单独一个 = marketing。

## Ship It / 交付它

保存 `outputs/skill-eval-report.md`。Skill 接收 new model checkpoint + baseline，并输出完整 eval plan：sample sizes、metrics、failure-mode probes、sign-off criteria。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。在同一 synthetic distributions 上比较 N=100 和 N=1000 的 FID。报告 bias magnitude。
2. **Medium / 中等。** 从 synthetic CLIP-style features 实现 CMMD（公式见 Jayasumana et al., 2024）。比较它与 FID 对质量差异的敏感度。
3. **Hard / 困难。** 复现 HPSv2 设置：从 Pick-a-Pic 子集中取 1000 个 image-prompt pairs，在 preferences 上 fine-tune 一个小 CLIP-based scorer，并测量它与 held-out set 的一致性。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 对 real vs gen Inception features 的 Gaussian fits 计算 Fréchet distance。 |
| CLIP score | "Text-image similarity" | CLIP image 和 text embeddings 的 cosine similarity。 |
| CMMD | "FID's replacement" | CLIP-feature MMD；偏差更小，没有 Gaussian assumption。 |
| IS | "Inception score" | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "Learned preference proxies" | 在 human preferences 上训练的小模型，用作 automatic judges。 |
| Elo | "Chess rating" | Pairwise wins 的 Bradley-Terry aggregation。 |
| PartiPrompts | "The benchmark prompt set" | Google curated 的 1,600 prompts，覆盖 12 categories。 |
| FD-DINO | "Self-sup replacement" | 使用 DINOv2 features 的 FD；更适合 out-of-ImageNet domains。 |

## Production Note: Evaluation Is an Inference Workload Too / 生产备注：评估也是 inference workload

在 10k samples 上跑 FID 意味着生成 10k 张图。对单张 L4 上 1024² 的 50-step SDXL base，这是约 11 小时的 single-request inference。评估预算是真实存在的，而且它正是 offline-inference scenario（最大化 throughput，忽略 TTFT）：

- **Batch hard, forget latency / 尽量 batch，忘掉 latency。** Offline eval = 在最大可容纳 memory 下做 static batching。80GB H100 上用 `pipe(...).images` 和 `num_images_per_prompt=8`，wall-clock 比 single-request 快 4-6×。
- **Cache the real features / 缓存真实特征。** Real reference set 上的 Inception（FID）或 CLIP（CLIP-score、CMMD）feature extraction 只跑 *一次*，存成 `.npz`。不要每次 eval 重算。

对 CI / regression gates：每个 PR 在 500-sample subset 上跑 FID + CLIP score（约 30 min）；每晚跑完整 10k FID + HPSv2 + Elo。

## Further Reading / 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — failure-mode survey。
