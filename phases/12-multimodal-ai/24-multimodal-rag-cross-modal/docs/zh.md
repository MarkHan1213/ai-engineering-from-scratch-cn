# Multimodal RAG and Cross-Modal Retrieval / 多模态 RAG 与跨模态检索

> Vision-native document RAG 只是一个切片。生产级 multimodal RAG 更宽：跨 text、images、audio 和 video 检索，支撑 trip planning（“find me a quiet vegan brunch with natural light”）、medical triage（“what injury matches this photo + these notes”）、e-commerce（“outfits similar to this selfie, in my size”）、field service（“diagnose this engine sound plus photo of the part”）。2025 年三篇 survey（Abootorabi et al., Mei et al., Zhao et al.）梳理了子问题：cross-modal retrieval、retrieval fusion、generation grounding、multimodal evaluation。本课读取这些 survey，并设计 production pipeline。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites / 前置知识：** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 设计 cross-modal retrieval：text → image、image → text、audio → video 等。
- 比较三种 fusion strategies：score fusion、attention-based fusion、MoE fusion。
- 解释 generation grounding：当 sources 混合多种 modality 时，“cite your sources” 长什么样。
- 说出 2025 年三篇 canonical multimodal RAG surveys 及其 sub-problem taxonomy。

## The Problem / 问题

Single-modality RAG 是成熟模式：embed query、embed chunks、retrieve、塞进 LLM。Multimodal RAG 需要：

1. 多个 retrieval heads（每个 modality 需要兼容空间中的 embeddings）。
2. 跨 modality 融合 retrieval results。
3. 能引用多模态 sources 的 generation grounding。
4. 覆盖 cross-modal signal 的 evaluation metrics。

2025 年 surveys 都收敛到同一 taxonomy。

## The Concept / 概念

### Cross-modal retrieval / 跨模态检索

给定 modality A 的 query，检索 modality B 的 documents。三种模式：

1. Shared embedding space。CLIP 和 CLAP 产出 text + image / text + audio 的共享 embedding。跨模态直接用 cosine similarity。受限于 CLIP 训练过的 pairs。

2. Per-modality encoder + translation。Text encoder + image encoder + 小 translator module，把一个空间映射到另一个。Gupta et al. 的 Sen2Sen 和其他 2024 设计。灵活但复杂。

3. VLM as encoder。使用 VLM hidden states 作为 retrieval representation。VLM 支持的任意 modality 都能用。质量更高，成本更高。

选择：text+image 用 CLIP / SigLIP 2；text+audio 用 CLAP；frontier quality 的 cross-modal 用 VLM-hidden-states。

### Fusion strategies / 融合策略

你检索到 10 个结果：5 张图、3 段文本、2 个音频 clips。如何合并？

Score fusion（最便宜）。每个 modality 有自己的 retriever，并返回 scores。先在 modality 内 normalize scores，再求和。简单，常常有效。

Attention-based fusion。把所有 retrieved items concatenate，让一个小 attention network 给权重。需要训练。

MoE fusion。Gating network 路由到 modality-specific experts。不同 query types 权重不同；视觉问题会更信任 images。

生产默认：score fusion，并略微偏向 query 的 dominant modality。只有 A/B 测试在领域内明确赢时，再升级到 MoE。

### Generation grounding / 生成 grounding

LLM 应该引用每个 claim 由哪个 retrieved item 支撑。多模态下：

- Text source：标准 citation `[1]`。
- Image source：`[img 3]` 加短 caption。
- Audio：`[audio 2 at 0:34]`。

用 grounding-aware data 训练 generator：training target 中每个 claim 都带 source index。推理时模型自然输出 citations。

### The 2025 surveys / 2025 年 surveys

Abootorabi et al.（arXiv:2502.08826, “Ask in Any Modality”）：multimodal RAG taxonomy。覆盖 retrieval、fusion、generation。范围最广。

Mei et al.（arXiv:2504.08748, “A Survey of Multimodal RAG”）：聚焦 sub-task benchmarks 和 failure modes。适合 evaluation design。

Zhao et al.（arXiv:2503.18016）：vision-focused survey。对 ColPali-family 工作覆盖较强。

三篇一起读，可以得到 2025 年春季的 state of the art。多数子问题仍未解决。

### MuRAG — the foundational paper / MuRAG：奠基论文

MuRAG（Chen et al., 2022）是第一篇 multimodal RAG。它从 multimodal KB 检索 image + text，并生成答案。在 VLM 浪潮前证明了可行性。现代系统（REACT、VisRAG、M3DocRAG）都在此基础上发展。

### A production trip-planner example / 生产级 trip-planner 示例

Query：“find me a quiet vegan brunch with natural light.”

Pipeline：

1. Decompose query。“quiet” → audio/review keyword；“vegan brunch” → menu item；“natural light” → image feature。
2. 按 modality 检索：
   - Reviews 上 text retrieval：“vegan brunch, quiet ambiance.”
   - Restaurant photos 上 image retrieval：“natural light, airy.”
   - Ambient-sound clips 上 audio retrieval：“low decibel, no music.”
3. Fuse scores。每家餐厅得到 composite score。
4. Top-k restaurants → VLM generator with all evidence → 带 citations 的答案。

这远超 text-RAG。每个 modality 提供 text alone 缺失的信号。

### Agentic multimodal RAG / Agentic multimodal RAG

Multi-hop：如果第一次 retrieval 没有高置信答案，LLM reformulate 并再次检索。Phase 14 的 Agentic RAG patterns 在这里适用。示例：

- Retrieve initial top-10 → LLM asks “too noisy, filter for <40 dB” → re-retrieve。
- Retrieve images → LLM 看到其中一张有 menu → retrieve menu text → answer。

它增加复杂度，但能处理 single-shot retrieval 无法处理的 query。

### Evaluation / 评估

Cross-modal evaluation 仍不成熟。常见 proxy：

- Recall@k per modality。
- Fused top-k accuracy。
- Human-judged end-to-end satisfaction。
- Task-specific（bookings completed、purchases made）。

没有标准 benchmark 覆盖所有 modality。多数论文在 domain-specific tasks 上评估。

## Build It / 动手构建

本课构建一个 mock multimodal RAG：三个 retrievers（text、image、audio）在共享 corpus 上返回分数，score fusion 合并结果，generator stub 输出带 citations 的答案，并在低置信时触发一次 agentic reformulation。

## Use It / 应用它

`code/main.py`：

- 三个 mock retrievers（text、image、audio），作用于 restaurants 共享 corpus。
- Score fusion，用 configurable weights 组合 modality scores。
- Generator stub，输出带 citations 的 final answer。
- 简单 agentic loop，在 confidence low 时 reformulate query。

## Ship It / 交付它

本课产出 `outputs/skill-multimodal-rag-designer.md`。给定一个带 multimodal query flow 的 product spec，它会设计 retrievers、fusion、generator 和 evaluation。

## Exercises / 练习

1. 提出一个 medical-triage multimodal RAG：query = injury photo + text symptoms。哪些 modality 从哪些 KB 检索？

2. Score fusion 是简单 weighted sum。它有什么 failure mode 是 MoE fusion 可以避免的？

3. 阅读 Abootorabi et al. taxonomy（Section 3）。三个 canonical sub-problems 是什么？它们如何映射到你选择的产品？

4. 为 trip-planner multimodal RAG 设计 eval spec。哪些 metrics 覆盖 image recall、audio recall 和 composite correctness？

5. Agentic multi-hop RAG 每轮都有 latency tax。什么 query 难度下，accuracy gain 足以 justify latency？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Cross-modal retrieval | “Query one modality, retrieve another” | Text query 检索 images；image query 检索 text；需要 shared space 或 translator |
| Score fusion | “Combine scores” | Per-modality retrieval scores 的 weighted sum；最简单融合方式 |
| MoE fusion | “Modality-routed experts” | Gating network 按 query 选择更该信任哪个 modality 的 scores |
| Grounded generation | “Cite your sources” | 答案中每个 claim 都标记 source index |
| MuRAG | “First multimodal RAG” | 2022 年建立 multimodal RAG pattern 的论文 |
| Agentic multi-hop | “Reformulate and retry” | 当 first-pass confidence low 时，LLM 重新查询 retrievers |

## Further Reading / 延伸阅读

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
