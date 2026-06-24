# Document and Diagram Understanding / 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表单包含 layout、tables、diagrams、footnotes、headers 和 semantic structure，普通图像理解无法完整捕捉。VLM 之前的 stack 是 pipeline：Tesseract OCR + LayoutLMv3 + table-extraction heuristics。VLM 浪潮把它替换成 OCR-free models：Donut（2022）、Nougat（2023）、DocLLM（2023），直接输出 structured markup。到 2026 年，frontier 就是“把 page image 以 2576px native 喂给 Claude Opus 4.7”，structured-markup output 顺带得到。本课读取 document AI 的三个时代。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites / 前置知识：** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 解释 document AI 的三个时代：OCR pipeline、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三路输入：text、layout（bbox）、image patches，以及 unified masking。
- 比较 Donut（OCR-free, image → markup）、Nougat（scientific paper → LaTeX）、DocLLM（layout-aware generative）、PaliGemma 2（VLM-native）。
- 为新任务（invoices、scientific papers、handwritten forms、Chinese receipts）选择 document model。

## The Problem / 问题

“Understand this PDF” deceptively hard。信息存在于：

- Text content（90% 信号）。
- Layout（headers、footnotes、sidebars、two-column format）。
- Tables（rows、columns、merged cells）。
- Figures and diagrams。
- Handwritten annotations。
- Fonts and typography（title vs body）。

Raw OCR 只导出文字，丢掉其他信号。一个发票系统必须知道 “Total: $1,245” 来自 bottom-right，而不是 footnote。

## The Concept / 概念

### Era 1 — OCR pipeline (pre-2021) / 时代 1：OCR pipeline

经典 stack：

1. PDF → 每页 image。
2. Tesseract（或 commercial OCR）提取 text 与 per-word bounding boxes。
3. Layout analyzer 识别 blocks（header、table、paragraph）。
4. Table structure recognizer 解析 tables。
5. Domain rules + regex 提取 fields。

对干净印刷文本有效。对 handwriting、skewed scans、complex tables、non-English scripts 容易失败。每个 failure mode 都需要 custom exception path。

### TrOCR (2021) / TrOCR（2021）

TrOCR（Li et al., arXiv:2109.10282）把 Tesseract 的 classic CNN-CTC 替换成在 synthetic + real text images 上训练的 transformer encoder-decoder。手写和多语文本上明显提升。仍然是 pipeline（detector -> TrOCR -> layout），但 OCR step 大幅改进。

### Era 2 — OCR-free (2022-2023) / 时代 2：OCR-free

第一批 OCR-free models 说：跳过 detection，直接把 image pixels 映射到 structured output。

Donut（Kim et al., arXiv:2111.15664）：

- Encoder-decoder transformer，encoder 是 Swin-B。
- 输出可以是 form understanding 的 JSON、summarization 的 markdown，或任意 task-specific schema。
- 无 OCR、无 layout、无 detection。

Nougat（Blecher et al., arXiv:2308.13418）：

- 专门在 scientific papers 上训练。
- 输出 LaTeX / markdown。
- 处理 equations、multi-column layout、figures。
- 几乎每个 arXiv-parser 都会调用它。

这些是 specialists，不是 generalists。Donut 处理科学论文会失败；Nougat 处理发票会失败。

### LayoutLMv3 (2022) / LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang et al., arXiv:2204.08387）保留 OCR，但加入 layout understanding：

- 三路输入：OCR text tokens、per-token 2D bounding boxes、image patches。
- 对三种 modality 做 masked training objective（masked text、masked patches、masked layout）。
- 下游：classification、entity extraction、table QA。

LayoutLMv3 是 OCR-based document understanding 的高峰。Forms 和 invoices 上很强。需要上游 OCR。在标准 document benchmarks 上是 VLM 前最强 accuracy。

### DocLLM (2023) / DocLLM（2023）

DocLLM（Wang et al., arXiv:2401.00908）是 LayoutLM 的 generative sibling。基于 layout tokens 生成 free-form answers。更适合 document QA；仍依赖 OCR input。

### Era 3 — VLM-native (2024+) / 时代 3：VLM-native

2024 年的 VLM 已经足够好，可以替换整个 pipeline。把 full page image 以高 resolution 喂给 VLM，提问，得到答案。

- LLaVA-NeXT 336-tile AnyRes 可处理小文档。
- Qwen2.5-VL dynamic-resolution 原生处理 2048+ pixels。
- Claude Opus 4.7 支持 2576px documents。
- PaliGemma 2（2025 年 4 月）专门为 documents + handwriting 训练。

VLM-native 与 OCR-pipeline 的差距迅速收敛。到 2026 年，VLM-native 在这些场景胜出：

- Scene text（手写 + 印刷、混合脚本）。
- Complex tables with merged cells。
- 嵌入正文中的 math equations。
- 带 text annotations 的 figures。

OCR pipelines 仍然胜出于：

- 大规模纯扫描 workload，每页 latency 很关键。
- Pipeline reliability（deterministic failures vs VLM hallucinations）。
- 受监管环境，需要可审计 OCR output。

### The Claude 4.7 / GPT-5 frontier / Claude 4.7 / GPT-5 前沿

2576-pixel native input 下，frontier VLM 的 document understanding 接近人类。2026 年初 benchmark：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，pipelined LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

Closed-model gap 主要来自 resolution 和 base-LLM scale。Open 7B models 落后几个点，但在追赶。

### Math equations and LaTeX output / 数学公式与 LaTeX 输出

科学论文需要精确 LaTeX 输出。Nougat 专门为此训练。用 LaTeX targets 训练的 VLM（Qwen2.5-VL-Math、Nougat derivatives）能输出可用 LaTeX。没有显式 LaTeX training 的 VLM 会给出可读但不够精确的 transcription。

2026 年科学论文 pipeline：PDF 上跑 Nougat，再让 VLM 处理 tricky pages。

### Handwriting / 手写

这仍是最难子任务。混合印刷 + 手写（医生笔记、填写表单）是 OCR pipelines 在成本上仍胜过 VLM 的地方。Handwritten-only VLMs 在变强（Claude 4.7、PaliGemma 2）。

### 2026 recipe / 2026 配方

新 document-AI 项目：

- 大规模纯印刷 invoices：LayoutLMv3 + rules，cost-efficient。
- 混合文档（scientific + handwritten + forms）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv ingestion：Nougat 处理 math，VLM 处理 figures。
- Regulatory：OCR pipeline + VLM validator 做 cross-check。

## Build It / 动手构建

本课构建 layout-aware document parser skeleton：把 `(text, bbox)` pairs 转成 LayoutLMv3-style inputs，并为 Donut-style task 生成 JSON schema。你会把 OCR-pipeline 与 VLM-native 的 token 和结构成本放在一起比较。

## Use It / 应用它

`code/main.py`：

- Toy layout-aware tokenizer：给定 `(text, bbox)` pairs，生成 LayoutLMv3-style input。
- Donut-style task schema generator：forms 的 JSON template。
- 对比 OCR-pipeline、Donut、Nougat 和 VLM-native 每页 token budgets。

## Ship It / 交付它

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定 document-AI project（domain、scale、quality、regulatory），它会在 OCR pipeline、OCR-free specialist 和 VLM-native 之间选择。

## Exercises / 练习

1. 项目每天处理 10M invoices。哪个 stack 能最小化 cost-per-page 且不牺牲 accuracy？

2. 为什么 LayoutLMv3 在 form QA 上超过 pure-CLIP-VLMs，但在 scene-text 上落后？bbox stream 带来和放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM-native 在 LaTeX fidelity 上超过 Nougat 的 test case，以及一个 Nougat 胜出的 case。

4. 阅读 PaliGemma 2 paper（Google, 2024）。相对 PaliGemma 1，提升 document accuracy 的关键训练数据新增是什么？

5. 设计一个 regulatory-safe hybrid：OCR pipeline 作为 primary，VLM 作为 secondary cross-check。如何处理 disagreement？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| OCR pipeline | “Tesseract-style” | 阶段式 stack：detect -> OCR -> layout -> rules；确定性强但脆弱 |
| OCR-free | “Donut-style” | 跳过显式 OCR 的 image-to-output transformer；单模型 |
| Layout-aware | “LayoutLM” | 输入包含 per-token bbox coordinates；跨 modalities 做 unified masking |
| VLM-native | “Frontier VLM” | 直接把 page image 高分辨率喂给 Claude/GPT/Qwen VLM；无 pipeline |
| DocVQA | “Doc benchmark” | Document VQA standard；最常被引用的分数 |
| Markup output | “LaTeX / MD” | 结构化输出格式，而非 free-form text；便于下游自动化 |

## Further Reading / 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
