# Open-Weight VLM Recipes: What Actually Matters / Open-Weight VLM 配方：真正重要的是什么

> 2024-2026 年的 open-weight VLM 论文像一片 ablation table 森林。Apple 的 MM1 测了 13 种 image encoder、connector 和 data mix 组合。Allen AI 的 Molmo 证明详细人工 caption 胜过 GPT-4V distillation。Cambrian-1 做了 20+ encoder 对比。Idefics2 形式化了五轴 design space。Prismatic VLMs 在受控 benchmark 上比较了 27 个训练 recipe。噪声很多，但跨论文稳定成立的结果很少：image encoder 比 connector architecture 更重要，data mixture 比两者都重要，详细人工 caption 在同 token count 下胜过 distilled synthetic data。本课替你读这些表。

**Type / 类型：** Learn + lab / 学习 + 实验
**Languages / 语言：** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites / 前置知识：** Phase 12 · 05 (LLaVA baseline)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 说出 VLM design space 的五个轴：image encoder、connector、LLM、data mix、resolution schedule。
- 阅读 MM1 / Idefics2 / Cambrian-1 的 ablation table，并预测哪个 knob 会移动某个 benchmark。
- 在给定 compute budget 和 task mix 时，为新 VLM 选择 recipe（encoder、connector、data、resolution）。
- 解释为什么详细人工 caption 在相同 token count 下胜过 GPT-4V distillation。

## The Problem / 问题

开源 VLM 已有数百个。从“能用”到“state-of-the-art”的差距，大多不在架构，而在数据、resolution schedule 和 encoder choice。知道模型表现不佳时先拧哪个 knob，能避免一次 5-million-GPU-hour 的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）使用 caption-pair pretraining + LLaVA-Instruct-150k。是不错 baseline，但 MMMU 大约封顶在 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了大量 ablations。结果出人意料，而且非常实用。

## The Concept / 概念

### The five-axis design space / 五轴设计空间

Idefics2（Laurençon et al., 2024）命名了这些轴：

1. Image encoder。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。差异在 patch size、resolution 和 pretraining objective。
2. Connector。MLP（2-4 layers）、Q-Former（32 queries + cross-attn）、Perceiver Resampler（64 queries）、C-Abstractor（convolutional + bilinear pooling）。
3. Language model。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM size 是参数成本的主导。
4. Training data。Caption pairs（CC3M、LAION）、interleaved（OBELICS、MMC4）、instruction（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. Resolution schedule。固定 224/336/448、AnyRes、native dynamic。训练中 ramped 或 constant。

每个 production VLM 都会在这五个轴上做选择。MMMU score 的多数方差由 axes 1、4、5 解释，而不是 connector。

### Axis 1: encoder > connector / 轴 1：encoder 大于 connector

MM1 Section 3.2 显示：把 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 增加 3+ 点；把 connector 从 MLP 换成 Perceiver Resampler，提升不到 1 点。Idefics2 复现了这一点：SigLIP > CLIP；在 token count 相同情况下，Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的 “Cambrian Vision Encoders Match-Up”（Tong et al., 2024）在 vision-centric benchmark（CV-Bench）上测了 20+ encoders。榜首混合了 DINOv2 和 SigLIP；CLIP 在中游；ImageBind 和 ViT-MAE 更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14，CV-Bench 约有 5-7 点差距。

2026 年 open VLM 的默认 encoder 是 SigLIP 2 SO400m/14，面向 semantic + dense features；如果需要 segmentation/grounding，有时会拼接 DINOv2 ViT-g/14 features（Cambrian 的 “Spatial Vision Aggregator” 就这么做）。

### Axis 2: connector design is a wash / 轴 2：connector 设计差异很小

MM1、Idefics2、Prismatic 和 MM-Interleaved 得出同样结论：在固定 visual-token count 下，connector architecture 几乎不重要。对 mean-pooled patches 使用 2-layer MLP，与同 token budget 的 32-query Q-Former 相差不到 1 点。

真正重要的是 token count。更多 visual tokens = 更多 LLM compute = 性能提升，直到边际收益递减。每张图 64 tokens 对 OCR 太少。576-1024 tokens 是多数 open VLM 的最佳折中区间。2048+ 主要帮助 documents 和 charts。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 无论 image resolution 如何都限制在 32-64 tokens；MLP 输出全部 patch tokens。高分辨率输入时 Q-Former 节省 LLM context；低分辨率时差异不大。

### Axis 3: LLM size sets the ceiling / 轴 3：LLM size 决定上限

把 LLM 从 7B 翻到 13B，几乎所有 VLM 论文里 MMMU 都稳定增加 2-4 点。到 70B 时，大多数 benchmark 接近饱和。VLM 的 multimodal reasoning 上限就是 LLM 的 text reasoning 上限；visual encoder 只能给它输入，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro、ScreenSpot-Pro 上碾压：语言大脑足够大。7B VLM 不能靠聪明 connector 设计替代 70B VLM。

### Axis 4: data — detailed human captions beat distillation / 轴 4：数据，详细人工 caption 胜过蒸馏

Molmo + PixMo（Deitke et al., 2024）是每个人都该读的 2024 结果。Allen AI 让人工标注员用 1-3 分钟 dense speech-to-text 描述图像，得到 712K 张 dense-captioned images。训练数据里没有任何 GPT-4V distillation。

Molmo-72B 在 11/11 个 benchmark 上击败 Llama-3.2-90B-Vision。差异不在 architecture，而在 caption quality。详细人工 caption 每张图包含的信息量比短网页 caption 多 5-10 倍，并且比 GPT-4V distillation 更少 hallucination。

ShareGPT4V（Chen et al., 2023）和 Cauldron（Idefics2）也沿着 mixed human + GPT-4V captions 方向走。趋势明确：2026 年前沿中，caption density > caption quantity > distillation convenience。

### Axis 5: resolution and its schedule / 轴 5：resolution 与 schedule

Idefics2 ablations：384 -> 448 增加 1-2 点。448 -> 980，加 image splitting（AnyRes），在 OCR benchmarks 上再增加 3-5。固定 resolution training 会在中等精度 plateau；resolution ramping（从 224 开始，最后到 448 或 native）训练更快，最终更高。

Cambrian-1 做了 resolution vs tokens trade-off：固定 compute 下，可以用更多 token 但较低 resolution，或更少 token 但更高 resolution。OCR 上高 resolution 胜出；一般 scene understanding 上，低分辨率更多 token 胜出。

2026 生产 recipe：Stage 1 在 384 fixed 训练，Stage 2 对 OCR-heavy tasks 使用 dynamic resolution up to 1280。

### The Prismatic controlled comparison / Prismatic 的受控比较

Prismatic VLMs（Karamcheti et al., 2024）控制了所有轴。相同 13B LLM、相同 instruction data、相同 evaluation，每次只改变一个轴。结果：

- Per-image visual-token count 解释约 60% 方差。
- Encoder choice 解释约 20%。
- Connector architecture 解释约 5%。
- 其他（data mix、scheduler、LR）解释剩下约 15%。

这只是粗略分解，但它是文献中对“先 ablate 什么”最干净的答案。

### A picker for 2026 / 2026 年选择器

基于这些证据，2026 年新项目的默认 open-VLM recipe：

- Encoder：native resolution + NaFlex 下的 SigLIP 2 SO400m/14；如果需要 segmentation/grounding，再拼接 DINOv2 ViT-g/14。
- Connector：patch tokens 上的 2-layer MLP。除非 token-constrained，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2；成本优先选 7B，质量优先选 70B，按目标 latency 取舍。
- Data：PixMo + ShareGPT4V + Cauldron，再补 task-specific instruction data。
- Resolution：dynamic（min 256, max 1280 pixels per long side）。
- Schedule：Stage 1 alignment（projector-only），Stage 2 full fine-tune，Stage 3 task-specific fine-tune。

这些默认值都能在本课结尾引用的论文 ablation 中找到依据。

## Build It / 动手构建

本课构建一个 ablation table parser 与 recipe picker。它把论文中的定性结论压成可查询规则：给定任务、预算、LLM 大小和 encoder choice，输出最值得优先尝试的 recipe，以及应该先做哪个 ablation。

## Use It / 应用它

`code/main.py` 是一个 ablation table parser 与 recipe picker。它编码了 MM1 和 Idefics2 的 condensed ablation tables，并允许查询：

- “Given budget X and task Y, what recipe wins?”
- “If I swap SigLIP for CLIP on a 7B Llama, what is the expected MMMU delta?”
- “Which axis should I ablate first for an 80% confidence answer?”

输出是带 expected benchmark deltas 的 ranked recipe list，以及一个 “ablate first” recommendation。

## Ship It / 交付它

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定 target task mix、compute budget 和 latency target，它会输出完整 recipe（encoder、connector、LLM、data mix、resolution schedule），并为每个选择附上对应 ablation 引用。它能避免工程师每次新 VLM 项目都重新发明 Idefics2 ablation table。

## Exercises / 练习

1. 阅读 MM1 Section 3.2。在固定 2B LLM、预算 50M images 下，哪个 encoder 胜出？如果换成 13B LLM，答案会翻转吗？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在 vision-centric benchmarks 上优于任意单独 encoder，但在 MMMU 上没有新增信号。预测哪些 benchmark 会提升，哪些保持不变。

3. 目标是 2B LLM 上的 mobile UI agent。选择 encoder、connector、resolution 和 data mix。用具体 ablation table 为每个选择辩护。

4. Molmo 有 4B 和 72B 模型。4B 能与 closed 7B VLM 竞争；72B 在 11/11 个 benchmark 上击败 Llama-3.2-90B-Vision。这对 LLM-size plateau hypothesis 说明了什么？

5. 设计一个 ablation table，在 7B VLM 上隔离 data-mix quality 与 encoder quality。最少需要多少次 training runs？提出四个 axis settings。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Ablation | “Turning one knob” | 训练多个 run，每次只改变一个 design-space axis，其他保持不变 |
| Connector | “Bridge” / “projector” | 把 vision encoder output 映射到 LLM token space 的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | “Dense caption” | 多句人工描述（通常 80-300 tokens），比网页 alt text 信息更丰富 |
| Distillation | “GPT-4V captions” | 由更强 proprietary VLM 生成的训练数据；方便但容易继承 hallucination |
| AnyRes / dynamic res | “High-res path” | 通过 tiling 或 M-RoPE 输入大于 encoder native resolution 的图像 |
| Resolution ramp | “Curriculum” | 从低 resolution 开始并逐步提高的训练 schedule，加速 alignment learning |
| Vision-centric bench | “CV-Bench / BLINK” | 强调细粒度视觉感知而非语言-heavy reasoning 的评测 |
| PixMo | “Molmo's data” | Allen AI 的 712K dense-captioned image dataset；人工语音转写成 dense captions |

## Further Reading / 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
