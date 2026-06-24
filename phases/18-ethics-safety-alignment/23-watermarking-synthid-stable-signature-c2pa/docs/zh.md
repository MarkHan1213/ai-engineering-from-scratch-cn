# Watermarking — SynthID, Stable Signature, C2PA / 水印：SynthID、Stable Signature、C2PA

> 三项技术定义了 2026 年 AI-generated-content provenance。SynthID（Google DeepMind）——2023 年 8 月推出 image watermarking，2024 年 5 月扩展到 text+video（Gemini + Veo），2024 年 10 月通过 Responsible GenAI Toolkit 开源 text，2025 年 11 月与 Gemini 3 Pro 同时发布 unified multi-media detector。Text watermarking 会微妙调整 next-token sampling probabilities；image/video watermarks 能穿过 compression、cropping、filters、frame-rate changes。Stable Signature（Fernandez et al., ICCV 2023, arXiv:2303.15435）——fine-tune latent diffusion decoder，使每个 output 都包含 fixed message；cropped（只剩 10% content）的 generated images 在 FPR<1e-6 下检测率 >90%。后续 “Stable Signature is Unstable”（arXiv:2405.07145，2024 年 5 月）显示 fine-tuning 可以在保持质量的同时移除 watermark。C2PA——cryptographically signed、tamper-evident metadata standard（C2PA 2.2 Explainer 2025）。Watermarking 与 C2PA 互补：metadata 可被剥离但携带更丰富 provenance；watermarks 能穿过 transcoding 但信息量少。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, token-watermark embed + detect)
**Prerequisites / 前置知识：** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 token-level watermarking（SynthID-text style）及其可检测机制。
- 描述 Stable Signature，以及 2024 年移除攻击如何击穿它。
- 说明 C2PA 的作用，以及为什么它与 watermarking 互补。
- 描述关键限制：model-specific signal、paraphrase robustness、meaning-preserving attacks（arXiv:2508.20228）。

## The Problem / 问题

2023-2024 年，deepfakes 与 AI-generated content 大规模进入政治与消费场景。Watermarking 是提出的 technical provenance signal：在生成时标记，之后检测。2025 年证据显示：没有 watermark 是无条件 robust 的，但与 C2PA metadata 分层组合后，可以提供可用的 provenance story。

## The Concept / 概念

### Text watermarking (SynthID-text style) / 文本水印

Kirchenbauer et al. 2023 机制，被 Google 产品化：

1. 每个 decoding step，hash 前 K 个 tokens，把 vocabulary 伪随机划分为 “green” 和 “red” sets。
2. 对 green logits 加 δ，让 sampling 向 green set 偏置。
3. 生成文本中 green tokens 数量超过随机机会。

Detection：重新 hash 每个 prefix，统计 generation 中 green tokens，计算 z-score。Watermarked text 的 z-score >0；human text 约为 0。

性质：
- 对读者不可感知（δ 足够小，quality loss 轻微）。
- 有 vocabulary partition function 时可检测。
- 对 paraphrase 不 robust：重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google Responsible GenAI Toolkit 开源。

### Stable Signature (image) / Stable Signature（图像）

Fernandez et al. ICCV 2023。Fine-tune latent diffusion decoder，让每张 generated image 都在 latent representation 中嵌入 fixed binary message。Detection 由 neural decoder 从 latent 中解码。被裁剪到仅剩 10% content 的图片仍能以 >90% 检出率、FPR<1e-6 被检测。

2024 年 5 月 “Stable Signature is Unstable”（arXiv:2405.07145）：fine-tuning decoder 可以在保持 image quality 的同时移除 watermark。Adversarial post-generation fine-tuning 很便宜；watermark 的 adversarial robustness 有限。

### SynthID unified detector (November 2025) / SynthID 统一检测器

与 Gemini 3 Pro 同时发布：multi-media detector，一个 API 读取 text、image、audio、video 中的 SynthID signals。统一了 Google provenance stack。

### C2PA / C2PA

Coalition for Content Provenance and Authenticity。Cryptographically signed tamper-evident metadata standard。C2PA 2.2 Explainer（2025）。C2PA manifest 记录 provenance claims（谁创建、何时、经过哪些 transformations），并由 creator key 签名。

与 watermarking 互补：
- Metadata 可以被剥离；watermarks 不容易。
- Metadata 信息丰富（完整 provenance chain）；watermarks 只携带 bits。
- C2PA 依赖 platform adoption；watermarks 自动嵌入。

Google 在 Search、Ads 和 “About this image” 中集成二者。

### Limitations / 限制

- **Model-specific。** SynthID 只标记 SynthID-enabled models 的 generations。没有 SynthID signal 不等于 authenticity proof，因为可能来自未启用 SynthID 的模型。
- **Paraphrase。** Text watermarks 不能穿过 meaning-preserving paraphrase。
- **Transformation attacks。** arXiv:2508.20228（2025）展示 meaning-preserving attacks 可以破坏 text watermarks 和许多 image watermarks。
- **Fine-tune removal。** 正如 “Stable Signature is Unstable”，post-generation fine-tuning 可以移除 embedded watermarks。

### EU AI Act Article 50 / EU AI Act 第 50 条

AI-generated content labelling 的 Transparency Code（第一稿 2025 年 12 月，第二稿 2026 年 3 月，按 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content) 预期最终稿 2026 年 6 月）。截至 2026 年 4 月，该 Code 仍是 draft，timeline 可能变化。这是要求 technical layer 的 regulatory layer。Deepfakes 必须被 label。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 22-23 讨论模型 emits 的内容（private data、provenance signal）。Lesson 27 覆盖 training-data governance。Lesson 24 是要求这些 technical measures 的 regulatory framework。

## Build It / 动手构建

本课构建一个 toy text watermark：整数 tokens、hash-defined green set、biased sampling 和 z-score detector。你会观察长度、paraphrase 和 false positive threshold 如何影响可检测性。

## Use It / 应用它

`code/main.py` 构建一个 toy text watermark。Tokens 是 integers 0..N-1；watermarked sampling 会向 hash-defined green set 偏置。Detector 计算 green-token z-score。你可以观察 1000-token generations 的检测结果、看 paraphrase 如何破坏信号，并测量 human text 上的 false-positive rate。

## Ship It / 交付它

本课产出 `outputs/skill-provenance-audit.md`。给定 content deployment 与 provenance claim，它会审计：watermark mechanism（若有）、C2PA signing chain（若有）、各自 adversarial robustness，以及每种 modality 的 coverage。

## Exercises / 练习

1. 运行 `code/main.py`。报告 watermarked 1000-token generation 与 human-authored text 的 z-scores。识别 95% confidence threshold 下的 false-positive rate。

2. 实现一个 paraphrase attack，用 synonyms 替换 30% tokens。重新测量 z-score。

3. 阅读 Kirchenbauer et al. 2023 Section 6 on robustness。为什么 text watermarks 在 paraphrase 下失败，而 image watermarks 能穿过 cropping？

4. 设计一个使用 SynthID-text + C2PA metadata 的 deployment。描述 consumer 看到的 provenance chain。分别指出每个组件的一个 failure mode。

5. 2024 “Stable Signature is Unstable” 结果显示 fine-tuning 会移除 image watermark。设计一个 deployment control 限制这种攻击，例如要求 fine-tuned checkpoints 的 signed releases。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| SynthID | “Google's watermark” | cross-modal provenance signal；text、image、audio、video |
| Token watermark | “Kirchenbauer-style” | biased-sampling text watermark，可通过 green-token z-score 检测 |
| Stable Signature | “image watermark” | fine-tuned-decoder watermark；ICCV 2023 |
| C2PA | “the metadata standard” | cryptographically signed tamper-evident provenance metadata |
| Paraphrase robustness | “does rewording break it” | text watermark property；当前有限 |
| Fine-tune removal | “adversarial unwatermark” | 通过 decoder fine-tuning 移除 image watermark 的 attack |
| Cross-modal detector | “unified SynthID” | 2025 年 11 月跨 modalities 的统一 API |

## Further Reading / 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token-watermark mechanism。
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — image watermark paper。
- [“Stable Signature is Unstable” (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — removal attack。
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — cross-modal watermark。
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — metadata standard。
