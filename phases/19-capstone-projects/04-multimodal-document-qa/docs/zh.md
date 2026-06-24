# Capstone 04 — Multimodal Document QA (Vision-First PDF, Tables, Charts) / 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年文档 QA 的前沿已经从 OCR-then-text 转向 vision-first late interaction。ColPali、ColQwen2.5 和 ColQwen3-omni 把每个 PDF 页面当作图像，用 multi-vector late interaction 做 embedding，让 query 直接关注 patches。在金融 10-K、科学论文和手写笔记上，这种模式显著优于 OCR-first。本 capstone 要你在 10k pages 上端到端构建 pipeline，并发布与 OCR-then-text 的 side-by-side 对比。

**类型：** 综合项目
**语言：** Python（pipeline）, TypeScript（viewer UI）
**前置知识：** 第 04 阶段（computer vision）, 第 05 阶段（NLP）, 第 07 阶段（transformers）, 第 11 阶段（LLM engineering）, 第 12 阶段（multimodal）, 第 17 阶段（infrastructure）
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## Learning Objectives / 学习目标

- 构建 vision-first PDF retrieval pipeline，把页面渲染为图像并生成 multi-vector embeddings
- 理解 late interaction / MaxSim 如何让 query tokens 与 page patches 细粒度匹配
- 在视觉检索、OCR-then-text 和 hybrid 方案之间做内容类型维度的评估
- 实现证据区域引用和 viewer overlay，让答案可回溯到页面位置
- 用 ViDoRe v3、M3DocVQA、storage 和 latency 指标评估系统

## Problem / 问题

企业持有大量会被 OCR pipeline 搞坏的 PDF：扫描版 10-K 中旋转的表格、包含密集公式的科学论文、只有看图才有意义的 charts、手写批注。把这些文档当成 text-first，会损失一半信号。2026 年的答案是对原始页面图像做 late-interaction multi-vector retrieval。ColPali（Illuin Tech）开创了这个方向；ColQwen2.5-v0.2 和 ColQwen3-omni 继续提升准确率。在 ViDoRe v3 上，vision-first retrieval 明显高于 OCR-then-text，且差距会在图表、表格和手写内容上扩大。

代价是 storage 和 latency。ColQwen embedding 每页约 2048 个 patch vectors，而不是单个 1024-dim vector。原始存储会膨胀。DocPruner（2026）可以在几乎不损失准确率的前提下剪掉 50%。你将索引 10k pages，测量 ViDoRe v3 nDCG@5，在 2s 内服务答案，并与 OCR-then-text baseline 直接对比。

## Concept / 概念

Late interaction 的意思是：每个 query token 都与每个 patch token 打分，然后对每个 query token 取最大分再求和。这样无需把文档压成单个 pooled vector，也能获得细粒度匹配。multi-vector index（Vespa、Qdrant multi-vector 或 AstraDB）存储 per-patch embeddings，并在检索时运行 MaxSim。

answerer 是一个 vision-language model，输入 query 和 top-k retrieved pages 的图像，输出带 evidence regions（bounding boxes 或 page references）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。遇到公式和科学记号时，可以把 OCR fallback（Nougat、dots.ocr）作为可选 text channel 拼进去。

评估是二维矩阵。一条轴是 content type（plain text paragraphs、dense tables、bar/line charts、handwritten notes、equations）。另一条轴是 retrieval approach（vision-first late interaction、OCR-then-text、hybrid）。每个单元格都记录 nDCG@5 和 answer accuracy。报告本身就是交付物。

## Architecture / 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Stack / 技术栈

- Page rendering: PyMuPDF (fitz)，180 DPI，portrait-normalized
- Late-interaction model: ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore team）
- Index: Vespa multi-vector field，或 Qdrant multi-vector，或支持 MaxSim 的 AstraDB
- Pruning: DocPruner 2026 policy（保留高方差 patches，50% compression，准确率损失 < 0.5%）
- OCR fallback（equations / dense tables）: dots.ocr 或 Nougat
- VLM answerer: self-hosted Qwen3-VL-30B 或 hosted Gemini 2.5 Pro；InternVL3 作为 fallback
- Evaluation: ViDoRe v3 benchmark，M3DocVQA 用于 multi-page reasoning
- Viewer UI: Next.js 15，使用 canvas overlay 展示 evidence regions

## Build It / 动手构建

1. **Ingest.** 遍历一个包含 10-K、科学论文、扫描文档的 10k PDF pages corpus。把每页渲染成 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **Embed.** 对每个 page image 运行 ColQwen2.5-v0.2。输出形状约为 2048 个 patch embeddings，每个 dim 128。应用 DocPruner 保留信号最高的一半。写入 Vespa multi-vector field 或 Qdrant multi-vector。

3. **Query.** 对每个 incoming query，用 query tower 做 embedding（token-level embeddings）。对 index 运行 MaxSim：每个 query token 都在 page patch embeddings 上取最大 dot-product，再求和。返回 top-k pages。

4. **Synthesize.** 用 query 和 top-5 page images 调用 Qwen3-VL-30B。Prompt: "Answer using only the supplied pages. Cite each claim by (doc_id, page) and name the region (figure, table, paragraph)."

5. **Evidence regions.** 对答案做 post-process，抽取 cited regions。如果 VLM 输出 bounding boxes（Qwen3-VL 支持），就把它们渲染为 viewer overlay。

6. **OCR fallback.** 对被识别为 equation-dense 的页面（基于 image variance 的 heuristic），运行 Nougat 或 dots.ocr，并把 OCR text 作为额外 channel 与图像一起传入。

7. **Eval.** 运行 ViDoRe v3（retrieval nDCG@5）和 M3DocVQA（multi-page QA accuracy）。在同一 corpus 上用同一个 synthesizer 跑 OCR-then-text pipeline。产出 content-type × approach 矩阵。

8. **UI.** 先做 Streamlit prototype，再做 Next.js 15 production viewer，支持逐页 evidence-region overlay。

## Use It / 应用它

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Ship It / 交付它

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定 corpus 调优的 vision-first multimodal document QA system，并在 ViDoRe v3 上与 OCR-then-text baseline 对比评估。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline and published leaderboard |
| 20 | Evidence-region grounding | cited regions 真正包含 answer span 的比例 |
| 20 | Storage and latency engineering | DocPruner compression ratio、index p95、answer p95 |
| 20 | Multi-page reasoning | 手工标注 100-question multi-page set 上的 accuracy |
| 15 | Source-inspection UX | Viewer clarity、overlay fidelity、side-by-side comparison tools |
| **100** | | |

## Exercises / 练习

1. 在同一 corpus 上测量 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面一个答对、另一个错过？给 index 添加 "content class" tag，并按类型路由。

2. 激进剪枝 embeddings（75%、90%）。找到 compression cliff：ViDoRe nDCG@5 低于 OCR baseline 的拐点。

3. 构建 hybrid：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，再用 cross-encoder rerank。hybrid 是否超过任一单独方案？它最擅长哪类内容？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。测量 accuracy-per-dollar curve。

5. 添加 handwritten-note support。渲染 handwriting corpus，用 ColQwen 做 embedding 并测量 retrieval。与 handwriting OCR pipeline 对比。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction | “ColPali-style retrieval” | Query tokens 独立对 page patches 打分；MaxSim 聚合 |
| Multi-vector | “Per-patch embedding” | 每个 document 有很多 vectors，而不是一个 pooled vector |
| MaxSim | “Late-interaction scoring” | 对每个 query token，在 document vectors 上取最大相似度后求和 |
| DocPruner | “Patch compression” | 2026 pruning 方法，保留 50% patches，准确率几乎不损失 |
| ViDoRe v3 | “Document-retrieval benchmark” | 2026 年衡量 visual-document retrieval 的标准 benchmark |
| Evidence region | “Cited bounding box” | 源页面上的 bbox，用于定位 answer span |
| OCR fallback | “Equation channel” | 对公式或表格密集页面，与 vision 并行使用的 text pipeline |

## Further Reading / 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — late-interaction doc retrieval reference
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — foundational method paper
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — production-ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — reference serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternate index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternate managed index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — equation-capable OCR fallback
