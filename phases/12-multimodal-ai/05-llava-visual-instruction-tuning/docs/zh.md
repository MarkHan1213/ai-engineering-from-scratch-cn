# LLaVA and Visual Instruction Tuning / LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的 multimodal architecture。它把 BLIP-2 的 Q-Former 换成 2-layer MLP，把 Flamingo 的 gated cross-attention 换成朴素的 token concatenation，并在 GPT-4 根据 text-only captions 生成的 158k visual-instruction turns 上训练。2023 到 2026 年间，几乎每个构建 VLM 的实践者都做过某种 LLaVA 变体。LLaVA-1.5 加入 AnyRes。LLaVA-NeXT 提高 resolution。LLaVA-OneVision 把 image、multi-image 和 video 统一到一个 recipe。本课读取这个 recipe，实现 projector，并解释为什么“更简单”赢了。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, projector + instruction-template builder)
**Prerequisites / 前置知识：** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 构建一个 2-layer MLP projector，把 ViT patch embeddings（dim 1024）映射到 LLM embedding dim（dim 4096）。
- 走通 LLaVA 两阶段 recipe：(1) 在 558k caption pairs 上做 projector alignment，(2) 在 158k GPT-4-generated turns 上做 visual instruction tuning。
- 构造 LLaVA-format prompt，包含 image token placeholder、system prompt 和 user/assistant turns。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token budget 上有优势。

## The Problem / 问题

BLIP-2 的 Q-Former（Lesson 12.03）把一张图压缩到 32 个 token。干净、高效、benchmark 上表现好。但它有两个问题。

第一，Q-Former 可训练，但它的 loss 不是最终任务。Stage 1 训练 ITC+ITM+ITG。Stage 2 训练 LM loss。Query 学到的是某种中间表示，LLM 还必须把它解码出来。瓶颈中会丢信息。

第二，Q-Former 有 188M 参数。在 LLaVA 的 2023 年规模下，你必须围绕目标 LLM 共同设计它。换 LLM，要重训 Q-Former；换 vision encoder，也要重训。每个组合都是一个单独的 R&D 项目。

LLaVA 的答案简单到有点尴尬：取 ViT 的 576 个 patch tokens，把每个 token 通过 2-layer MLP（`1024 → 4096 → 4096`），然后把全部 576 个 token 倒进 LLM 输入序列。没有 bottleneck。没有怪异 objective 的 stage 1 pretraining。只用 direct LM loss 训练 MLP。

数据从哪里来？LLaVA 的第二个洞察：用 GPT-4（text-only）生成 instruction data。把一张图的 COCO caption 和 bounding-box 数据喂给 GPT-4，让它产出 conversations、descriptions 和 complex reasoning questions。免费得到 158k instruction-response turns。没有人工标注。

结果是：一个 VLM 在 8 张 A100 上跑一天，MMMU 上超过 Flamingo，并发布了社区能扩展的 open checkpoint。到 2023 年底，它已经催生 50+ forks。

## The Concept / 概念

### The architecture / 架构

LLaVA-1.5 13B：

- Vision encoder：CLIP ViT-L/14 @ 336（stage 1 frozen，stage 2 可选 unfreeze）。
- Projector：带 GELU activation 的 2-layer MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本 prompt 的 forward pass：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM context 的 576 个 token。2048 context 下还剩 1472 token 给文本。32k context 下，这几乎只是零头。

### Stage 1: projector alignment / Stage 1：projector 对齐

Freeze ViT。Freeze LLM。只训练 2-layer MLP。数据集：558k image-caption pairs（LAION-CC-SBU）。Loss：在 projected image tokens 条件下，对 caption 做 language modeling。

Batch 128 单 epoch 几小时完成。Projector 学会把 ViT-space 映射到 LLM-space。不需要任务特定监督。

### Stage 2: visual instruction tuning / Stage 2：视觉指令微调

Projector 继续可训练。LLM 也 unfreeze（通常 full，有时 LoRA）。在 158k visual-instruction turns 上训练。

Instruction data 是关键。Liu 等人这样生成：

1. 取一张 COCO image。
2. 提取文本描述（5 条人工 caption + bounding-box list）。
3. 用三个 prompt template 发给 GPT-4：
   - Conversation：“Generate a back-and-forth dialogue between a user and assistant about this image.”
   - Detailed description：“Give a rich, detailed description of the image.”
   - Complex reasoning：“Ask a question that requires reasoning about the image, then answer it.”
4. 把 GPT-4 输出解析成 `(instruction, response)` pairs。

整个过程不直接看图，只看文本描述。GPT-4 会 hallucinate 一些合理图像内容。有噪声，但有效：158k turns 足以打开 dialogue 能力。

### Why the community copied this / 社区为什么复制它

- 没有 stage-1-specific losses 要调。全程都是 LM loss。
- Projector 几小时训练完，不是几天。
- LLM 可替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训 projector。
- Visual-instruction data pipeline 用 GPT-4，便宜且可为新领域再生成。

### LLaVA-1.5 and LLaVA-NeXT / LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）加入：

- Academic-task data（VQA、OKVQA、RefCOCO）混入 instruction tuning。
- 更好的 system prompt。
- 2048 → 32k context。

LLaVA-NeXT（2024 年 1 月）加入：

- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 crops，再加一个 global low-res thumbnail。每个 crop 是 576 token，总计约 2880 visual tokens/image。OCR 和 chart tasks 大幅提升。
- 更好的 instruction data mixture，引入 ShareGPT4V（高质量 GPT-4V captions）。
- 更强 base LLMs（Mistral-7B、Yi-34B）。

### LLaVA-OneVision / LLaVA-OneVision

Lesson 12.08 会深入讲 OneVision。短版：还是同一个 projector，但用一个 curriculum 在同一个模型中覆盖 single-image、multi-image 和 video，并共享 visual-token budget。

### The comparison to Q-Former / 与 Q-Former 对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Visual tokens per image | 32 | 576 (base) or 2880 (AnyRes) |
| Trainable params | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | LM only |
| LLM drop-in | Requires retrain | Swap with minimal retrain |
| Multi-image | Awkward | Natural (concat) |
| Video | Awkward | Natural (per-frame concat) |
| Token budget | Small | Large |

MLP 赢在简单性和 token flexibility。Q-Former 赢在 token budget。到 2023 年末，token budget 不再是绑定约束（LLM contexts 增长到 32k-128k+），简单性占了上风。

### The prompt format / Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是 placeholder token。Tokenization 之前，它会被 576 个 visual tokens（或 AnyRes 下 2880 个）替换。Tokenizer 看到的序列比预训练时略长，但 stage 1 已经教会 LLM 处理这种新输入。

### Parameter economy / 参数经济性

LLaVA-1.5-7B 拆解：

- CLIP ViT-L/14 @ 336：303M（stage 1 frozen，stage 2 常常 unfreeze）。
- Projector（2x linear）：约 22M trainable。
- Llama-7B：7B。
- 总计：7.3B 参数。Stage 2 训练：完整 7B + 22M projector。

Stage 2 训练成本：8xA100 上约 20 小时。这个数字是关键：一天、一个节点、可复现。这就是 LLaVA 扩散开的原因。

## Build It / 动手构建

本课动手实现 LLaVA 的最小 recipe：2-layer MLP projector、`<image>` placeholder 替换逻辑、以及 visual block 在 LLM context 中的预算可视化。你会看到 LLaVA 为什么能用直接 LM loss 把 ViT patch space 接到 LLM embedding space，而不需要额外 Q-Former objectives。

## Use It / 应用它

`code/main.py` 实现：

1. Pure Python 的 2-layer MLP projector（toy scale：dim 16 → 32 → 32）。
2. Prompt-building pipeline：system prompt + `<image>` 替换成 N 个 projected tokens + user turn + assistant generation placeholder。
3. 一个 visualizer，展示 576-token visual block 在 LLM context 中占多少比例（2k / 32k / 128k context）。

## Ship It / 交付它

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA-family checkpoint，它会运行 10-prompt vibes-eval suite（3 个 captioning、3 个 VQA、2 个 reasoning、2 个 refusal），并报告 human-readable scorecard。它不是 benchmark，而是 smoke test，用来确认 projector 与 LLM 连接良好。

## Exercises / 练习

1. 计算 `1024 → 4096 → 4096` 的 2-layer MLP projector 的 trainable-parameter count。带 GELU 和 bias 时，它占 LLaVA-13B 的比例是多少？

2. 为一个 “refusal” case 构造 LLaVA prompt：图像中包含私人个体。写出期望 assistant response。为什么 LLaVA 应该 zero-shot 拒绝？需要什么训练数据强化这种拒绝？

3. 阅读 LLaVA-NeXT blog 的 AnyRes section。计算 1344x672 图像在 AnyRes 下的 visual token count。与 336x336 的 base 576 tokens 对比。

4. LLaVA stage-1 projector 用 caption 上的 LM loss 训练。如果跳过 stage 1，直接进入 stage 2（visual instruction tuning），会发生什么？引用 Prismatic VLMs ablation（arXiv:2402.07865）回答。

5. LLaVA-Instruct-150k 用 GPT-4 与 COCO captions 生成 instructions。对于新领域（medical X-rays、satellite imagery），描述生成 domain instructions 的四步 data pipeline。每一步可能出什么问题？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Projector | “MLP bridge” | 带 GELU 的 2-layer MLP，把 ViT dim 映射到 LLM dim |
| Image token | “<image> placeholder” | prompt marker，推理前被 N 个 projected visual tokens 替换 |
| Visual instruction tuning | “LLaVA stage 2” | 在 GPT-4-generated `(image, instruction, response)` triplets 上训练 |
| Stage 1 alignment | “Projector pretraining” | 冻结 ViT 和 LLM，用 captions 上的 LM loss 训练 projector |
| AnyRes | “Multi-crop tiling” | 把高分辨率图像切成 tile grid，并拼接每个 tile 的 visual tokens |
| LLaVA-Instruct | “GPT-4-generated” | 从 COCO captions + GPT-4 合成的 158k instruction-response pairs |
| Vision encoder freeze | “Backbone locked” | CLIP 权重在 stage 1 不更新，有时 stage 2 也不更新 |
| ShareGPT4V | “Better captions” | GPT-4V 生成的 1M dense captions，用于更高质量 alignment |
| VQA | “Visual question answering” | 回答关于图像的自由形式问题 |
| Prismatic VLMs | “Design-space paper” | Karamcheti 2024 ablation，系统测试 projector 和 data choices |

## Further Reading / 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense captions dataset。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — design-space ablations。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — unified single-image、multi-image、video。
