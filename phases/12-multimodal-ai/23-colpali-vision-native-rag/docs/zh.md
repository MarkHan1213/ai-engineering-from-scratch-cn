# ColPali and Vision-Native Document RAG / ColPali 与视觉原生文档 RAG

> 传统 RAG 会把 PDF 解析成文本、切 chunk、embed chunk、存向量。每一步都丢信号：OCR 丢 chart data，chunking 打断 table rows，text embeddings 忽略 figures。ColPali（Faysse et al., 2024 年 7 月）问了一个更简单的问题：为什么要提取文本？直接通过 PaliGemma embed page image，使用 ColBERT-style late interaction 做检索，保留文档承载的 layout、figures、fonts 和 formatting signal。公开 benchmark 显示，在视觉丰富文档上端到端 accuracy 比 text-RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这个模式。本课读取 vision-native RAG thesis，并构建一个 tiny ColPali-like indexer。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites / 前置知识：** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 解释 bi-encoder retrieval（每文档一个向量）与 late-interaction retrieval（每文档多个向量）的差异。
- 描述 ColBERT 的 MaxSim operation，以及 ColPali 如何把它从 text tokens 泛化到 image patches。
- 构建一个 tiny ColPali-like indexer：page → patch embeddings → query-term embeddings 上的 MaxSim → top-k pages。
- 对比 ColPali + Qwen2.5-VL generator 与 text-RAG + GPT-4 在 invoices / financial reports 场景中的表现。

## The Problem / 问题

PDF 上的 text-RAG 会扔掉大部分文档。一份财报的 Q3 revenue growth 通常在图表里；医学报告的 findings 在 annotated images 里；法律合同的 signature block 是 layout fact，不只是 text fact。

Text-RAG pipeline：

1. PDF → 通过 OCR / pdftotext 变成 text。
2. Text → 300-500 token chunks。
3. Chunk → bi-encoder embedding（一个向量）。
4. User query → embedding → cosine similarity → top-k chunks。
5. Chunks + query → LLM。

五个有损步骤。Charts 捕捉不到。Tables 跨 chunk 断裂。Multi-column layout 被拉平。Figure annotations 消失。

ColPali 的修复：跳过 OCR，直接 embed page image。使用 ColBERT-style late interaction 做 retrieval，让模型在 query time attend 到细粒度 patches。

## The Concept / 概念

### ColBERT (2020) / ColBERT（2020）

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是文本检索方法。它不是每文档一个向量，而是每个 token 一个向量。Query time：

- Query tokens 有自己的 embeddings（N_q vectors）。
- Document tokens 有 embeddings（N_d vectors，通常缓存）。
- Score = sum over query tokens of max over document tokens of cosine similarity：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim。每个 query token “选择”最匹配的 document token。最终分数是求和。

优点：recall 强，处理 term-level semantics。缺点：每个文档 N_d vectors，存储昂贵。

### ColPali / ColPali

ColPali（Faysse et al., arXiv:2407.01449）把 ColBERT pattern 用到图像上。

- 每页由 PaliGemma（ViT + language）编码成 patch embeddings：每页 N_p vectors。
- 用户 query（text）编码成 query-token embeddings：N_q vectors。
- Score = Σ_i max_j cos(q_i, p_j)，即 query-text-tokens 与 page-image-patches 之间的 MaxSim。
- 按总分检索 top-k pages。

文档 ingestion 时：用 PaliGemma embed 每一页，存所有 patch embeddings。Query time：embed query tokens，对所有 page embeddings 做 MaxSim，返回 top-k pages。

优点：视觉丰富文档上端到端比 text-RAG 高 20-40%。每个 patch-vector 捕捉局部 layout 和 content。

缺点：每页 N_p patches × 4-byte floats × D-dim vectors，存储增长很快。用 PQ / OPQ quantization 缓解。

### ColQwen2 and ColSmol / ColQwen2 与 ColSmol

ColQwen2（illuin-tech, 2024-2025）把 PaliGemma 换成 Qwen2-VL。Base encoder 更好，retrieval 更好。

ColSmol 是本地/边缘使用的小规模变体。约 1B 参数的 ColSmol retriever 可在消费级 GPU 上运行。

### VisRAG / VisRAG

VisRAG（Yu et al., arXiv:2410.10594）是不同变体：不是在 patches 上做 MaxSim，而是用 VLM 把每页 pool 成单向量，再 bi-encoder retrieve。Indexing 更快、存储更小、recall 较弱。

质量-成本 trade-off：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG / M3DocRAG

M3DocRAG（Cho et al., arXiv:2411.04952）把 multi-modal retrieval 扩展到 multi-page multi-document reasoning。跨文档检索 pages，为 VLM 组合 multi-page context。

### ViDoRe — the benchmark / ViDoRe benchmark

ColPali 的 companion benchmark。Visual Document Retrieval Evaluation。任务包括 financial reports、scientific papers、administrative documents、medical records、manuals。Metric 是 nDCG@5。

ColPali-v1 在 ViDoRe 上约 80% nDCG@5；同文档 text-RAG 约 50-60%。

### The end-to-end RAG pipeline / 端到端 RAG pipeline

Vision-native RAG：

1. Ingest：PDF → page images → PaliGemma encoding → store all patch embeddings。
2. Query：user text → query-token embeddings → MaxSim against all indexed pages → top-k pages。
3. Generate：top-k page images + query → VLM（Qwen2.5-VL 或 Claude）→ answer。

全程无 OCR。Figures、charts、fonts、layout 都进入答案。

### Storage math / 存储计算

50 页 financial report，每页 729 patches，128-dim embeddings：

- ColPali：50 * 729 * 128 * 4 bytes = ~18 MB raw，PQ 后约 4 MB。
- Text-RAG：50 chunks * 768-dim * 4 bytes = ~150 kB。

ColPali 每文档存储约 30x。大规模下，OPQ / PQ 可降到约 5-10x，通常可接受。

### When text-RAG still wins / Text-RAG 仍然胜出的场景

- 没有 layout signal 的纯文本 documents（wiki articles、chat logs）。Text-RAG 更简单、存储更便宜。
- 多百万页 archives，存储主导成本。
- 严格监管要求：检索时必须保留 extractable OCR text。

除此之外，到 2026 年，financial reports、scientific papers、legal contracts、medical records、UX documentation 上 vision-native RAG 更强。

## Build It / 动手构建

本课构建 multi-vector indexer 与 MaxSim scorer：把 toy page 编码成 patch embedding set，把 query 编码成 token embedding set，然后用 ColBERT-style MaxSim 排序 top-k pages。重点是 late interaction，而不是训练真实 PaliGemma。

## Use It / 应用它

`code/main.py`：

- Toy patch encoder：把 “page”（小 feature vector grid）映射成 patch embeddings array。
- MaxSim scorer：计算 query token embedding set 与 page patch set 的 ColBERT-style score。
- 索引 5 个 toy pages，运行 3 个 queries，返回 top-k 和 scores。

## Ship It / 交付它

本课产出 `outputs/skill-vision-rag-designer.md`。给定 document-RAG project，它会在 ColPali / ColQwen2 / VisRAG / text-RAG 之间选择，并 sizing storage。

## Exercises / 练习

1. 200 页 annual report，每页 729 patches，128-dim emb，4-byte floats。计算 raw storage 和 PQ-compressed（8x）storage。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。它捕捉了 simple mean similarity 捕捉不到的什么？

3. ColPali 按 patch sets 索引 pages。如果改成 word level（像 ColBERT），会发生什么变化？Trade-offs？

4. 为 1M-page corpus 设计端到端 pipeline，latency budget 是每 query 500ms。选择 ColQwen2 / VisRAG 并说明理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述 multi-page attention pattern，以及它与 single-page ColPali retrieval 的差异。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction | “ColBERT-style” | 使用 per-token 或 per-patch embeddings + MaxSim 的检索，而不是单个 doc vector |
| MaxSim | “Max-over-patches” | 对每个 query token，选择相似度最高的 document token，并对 query 求和 |
| Bi-encoder | “Single-vector” | 每个文档一个向量；更快但丢 granularity |
| Multi-vector | “Many-vectors-per-doc” | 每个 document / page 存 N_p vectors；存储增加但 recall 提升 |
| Patch embedding | “Page feature” | VLM encoder 每个 image patch 的向量，按页缓存 |
| ViDoRe | “Vision doc bench” | ColPali 的 visual document retrieval benchmark suite |
| PQ quantization | “Product quantization” | 在保持 vector similarity 的同时压缩存储，通常约 8x |

## Further Reading / 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
