# Chunking Strategies, Compared / Chunking 策略对比

> Chunking 决定了你的 retriever 最多能检索出什么。边界切错了，下游再好的 embedding model、reranker、LLM 也修不回来。

**类型：** 构建
**语言：** Python
**前置知识：** 第 11 阶段第 04 课（embeddings）, 06（RAG）, 07（advanced RAG）; 第 19 阶段 Track B 基础（第 20-29 课）
**时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 从零实现五种 chunking strategies：fixed-window、sentence、recursive-split、semantic clustering、structural markdown headers。
- 在带 gold-labeled answer spans 的 fixture corpus 上测量 recall@k，并解释为什么一种策略在 prose 上胜出，另一种策略在 technical documents 上胜出。
- 读取 chunk-length distribution，识别每种策略注入的 failure modes：orphan sentences、mid-symbol cuts、header-only chunks、semantic drift。
- 不跑 benchmark 时，也能通过三个属性为新 corpus 选择默认策略：document type、average paragraph length、format 是否携带 explicit structure。

## The Problem / 问题

每个 RAG pipeline 都从切分 source documents 开始。切片既要小到 embedding model 能处理，又要大到每个片段能承载一个自洽 idea。切在哪里不是一个普通 hyperparameter，而是 retriever 未来能返回内容的上限。

如果 query 问的是 “what does the budget abort threshold look like”，只有包含 abort threshold 的 chunk 可达，系统才有机会成功。若 fixed-window splitter 把 threshold value 和周围 context 切开，embedding 会移到不同 cluster，BM25 score 降低，rerankers 看到的是 noise，LLM 生成的答案也会错。2024 年论文 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" 测到，仅 chunking choice 就能让 retrieval recall 产生 35 个百分点的绝对波动。2025 年 contextual chunk headers 的后续工作缩小了差距，但没有消除它。

本课把五种策略并排实现，在带 gold-labeled answer spans 的 fixture corpus 上运行，让你直接读 recall numbers。

## The Concept / 概念

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Fixed-window / 固定窗口

暴力 baseline。每 N 个 characters 切一次。可以加 overlap，让在位置 N 被切断的句子，完整出现在从 `N - overlap` 开始的 chunk 中。它快速、deterministic，但边界很差。把它当 control，不要当默认策略。

### Sentence / 句子切分

用 regex 或简单 state machine 在 sentence boundaries 上切分。把一个或多个 sentences 打包进 chunk，直到接近 target character budget。它不再切断 word，但仍可能切断 paragraph 和 section。很多早期 RAG pipelines 把它当默认；对没有其他结构的 prose 来说，它仍然合理。

### Recursive split / 递归切分

这是 2023 年左右 libraries 推广的 hierarchy strategy。先尝试最强 separator（double newline、paragraph），不合适就回退到下一层（single newline），再回退到 sentences，最后到 characters。chunk fits budget 时 recursion 终止。对结构不一致的 documents 很强，因为它能按 region 自适应。

### Semantic clustering / 语义聚类

先 embed 每个 sentence。把相邻且共享 topic centroid 的 sentences 聚成 cluster。当 running similarity to centroid 低于 threshold 时切开。边界反映 meaning，而不是 characters。它构建更慢，且依赖 embedding model，但对 paragraph 内部切换主题的 documents 更稳。

### Structural markdown headers / 结构化 Markdown 标题

对携带 explicit structure 的 documents（markdown、reStructuredText、RFC-style numbered sections），按 heading boundaries 切。每个 chunk 是 heading 加其下内容，直到遇到同级或更高级 heading。它能得到每个 topic 最小的 chunks，但只有 corpus 格式良好时才可用。

### How recall@k measures the boundary choice / recall@k 如何衡量边界选择

一个 gold-labeled query 会携带 source document 中 answer span 的精确 character offsets。chunking 之后，你问：retriever 返回的 top-k chunks 中，是否有任意 chunk 与 gold span 重叠？有则该 query 的 recall@k 为 1，没有则为 0。对 query set 求平均。对每种 strategy 跑同一评估，分差会显示哪种 boundary policy 经得住你的 corpus。

## Build It / 动手构建

`code/main.py` implements:

- `fixed_window(text, size, overlap)` - the baseline.
- `sentence_chunks(text, target)` - simple sentence packer.
- `recursive_split(text, separators, target)` - hierarchical recursion.
- `semantic_chunks(text, similarity_threshold)` - centroid-based clustering on top of a deterministic mock embedding.
- `structural_markdown(text)` - header-aware splitter.
- `mock_embed(text, dim)` - a hash-based embedding so the loop runs offline.
- `DenseIndex` - the same shape used in Phase 19 Track B's hybrid retrieval lesson.
- `eval_recall(strategy, corpus, queries, k)` - the comparison loop.
- A `main()` that runs every strategy on the fixture corpus and prints a recall@k table.

Run it:

```bash
python3 code/main.py
```

输出是一张小表，每个 strategy 一行，每个 k 一列。sentence 在 structured fixture 上会输。structural-markdown 在 markdown fixture 上胜出。recursive 在 mixed fixture 上表现稳定，因为 recursion 会自适应。semantic clustering 在没有可用 structural cues 的 prose fixture 上胜出。

## Failure modes the table will not hide / 表格不会隐藏的失败模式

**Orphan sentences.** Sentence packing 可能产出缺少 topic sentence 的 chunks。此时 embedding 会指向错误 cluster。

**Mid-symbol cuts.** Fixed-window 在 code 或 YAML 中会把 identifier 切成两半，两半都会 embed 成 noise。

**Header-only chunks.** Structural markdown 可能输出只包含 `## Title` 的 chunk。要过滤掉，或者附上下一 chunk 的第一段。

**Semantic drift.** 当 corpus 整体都在同一 topic 上时，semantic clustering 可能切得太少。一个 5000-character chunk 会把许多具体答案塞进一个 diffuse embedding。把 semantic 与 hard character cap 组合起来。

**Stale embeddings.** Semantic clustering 使用 embedding model。如果换 model，chunks 也会变。单独 pin chunk model，或者与 retrieval model 一起重建 index。

## Choosing a default without running the benchmark / 不跑 benchmark 时如何选择默认策略

新 corpus 的默认 chunker 由三个属性决定。

| Property | Value | Default |
|----------|-------|---------|
| Document type | Prose with no structure | Recursive split, target 800 |
| Document type | Markdown / RFC / API docs | Structural markdown |
| Document type | Code | AST-aware (out of scope; see Phase 19 lesson 02) |
| Paragraph length | Long, single topic | Sentence, target 500 |
| Paragraph length | Short, mixed topics | Semantic, threshold 0.6 |

拿不准时，选择 recursive split。它是最强的单策略 baseline。

## Use It / 应用它

Production patterns:

- 发布新 pipeline 前先跑 eval；不要盲信 library 的默认 strategy。
- 每当 embedding model 或 corpus mix 改变，都要重跑 eval；赢家取决于 corpus。
- 在每个 chunk metadata 中持久化 strategy name，便于之后归因 regressions。

## Ship It / 交付它

第 69 课的 Track F end-to-end RAG system 会把这里选出的 chunker 作为第一阶段。第 68 课的 eval harness 会读取与本课 `eval_recall` 相同形状的 recall@k。选择在你的 corpus 上胜出的 strategy，并把它传给后续链路。

## Exercises / 练习

1. 增加第六种策略：使用 `tiktoken` 的 token-window，而不是 character counts。在同一 fixture 上与 fixed-window 对比。
2. 向 prose fixture 注入 30% 的 code blocks。重跑表格。解释为什么除了 structural markdown 之外的策略都会损失 recall。
3. 用项目真实 provider 的 embedding 替换 deterministic embedding。测量 semantic-clustering recall delta，并报告 strategies 之间的差距是扩大还是缩小。
4. 给每个 chunk 增加 `summary` 字段：一句 centroid description。把 summary 拼到 chunk body 后重跑 eval，测量 recall lift。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Recall@k | "Did we get the right chunk?" | top-k chunks 中任意一个与 gold answer span 重叠的 query 比例 |
| Chunk overlap | "Sliding window" | 将前一个 chunk 末尾 N 个 characters 重新包含到下一个 chunk |
| Structural splitter | "Header-aware chunks" | 按 H1/H2/H3 边界切分，heading text 是 chunk 的一部分 |
| Semantic chunker | "Topic-aware chunks" | embed sentences，按 centroid similarity 聚类，并在 drift 时切开 |
| Centroid drift | "Topic shift" | running mean 与下一句之间的 cosine similarity 低于 threshold |

## Further Reading / 延伸阅读

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 lesson 06 - RAG fundamentals
- Phase 11 lesson 07 - advanced RAG
- Phase 19 lesson 65 - hybrid retrieval that ranks the chunks produced here
- Phase 19 lesson 68 - the eval harness that scores the strategy choice in production
