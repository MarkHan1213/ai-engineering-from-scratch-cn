# Capstone 02 — RAG over Codebase (Cross-Repo Semantic Search) / 面向代码库的 RAG（跨仓库语义搜索）

> 到 2026 年，严肃的工程组织都会运行内部代码搜索，它理解语义，而不只是字符串。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的 enterprise graph、Aider 的 repomap、Pinterest 内部 MCP，本质形态一致：摄取多个 repo，用 tree-sitter 解析，按函数和类级别 chunk 做 embedding，混合搜索，重排序，再带引用回答。本 capstone 要你构建一个能处理 10 个 repo、200 万行代码，并能在每次 git push 后增量重建索引的系统。

**类型：** 综合项目
**语言：** Python（ingestion）, TypeScript（API + UI）
**前置知识：** 第 05 阶段（NLP foundations）, 第 07 阶段（transformers）, 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 17 阶段（infrastructure）
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## Learning Objectives / 学习目标

- 构建 AST-aware ingestion pipeline，把代码按函数和类节点切分
- 同时生成 dense embedding、BM25 sparse terms 和自然语言 summary 三种可检索表示
- 实现跨 repo 的 hybrid retrieval、cross-encoder rerank 和带 file:line citation 的回答
- 设计增量 re-index 流程，让 push 后只重算受影响 chunk 与 symbol edge
- 用 MRR@10、nDCG@10、citation faithfulness 和 latency 评估代码 RAG

## Problem / 问题

到 2026 年，每个 frontier coding agent 都会带一个 codebase retrieval layer，因为 context window 本身不能解决跨 repo 问题。Claude 的 1M-token context 有帮助，但不会消除 ranked retrieval 的必要性。直接对原始 chunk 做 naive cosine search，会在生成代码、monorepo 重复代码、很少被 import 的长尾 symbol 上污染结果。生产答案是：基于 AST-aware chunks 的 hybrid（dense + BM25）search、re-ranker，以及 symbol references graph。

你要通过索引一组真实 repo，而不是一个教程仓库，来学习这件事。测量 MRR@10、citation faithfulness 和 incremental freshness。失败模式主要是基础设施问题：100k-file monorepo，一次 push 改动半数文件，一个 query 需要跨四个 repo 才能正确回答。

## Concept / 概念

AST-aware ingestion pipeline 使用 tree-sitter 解析每个文件，抽取 function 和 class nodes，并在节点边界切 chunk，而不是用固定 token 窗口。每个 chunk 有三种表示：dense embedding（Voyage-code-3 或 nomic-embed-code）、sparse BM25 terms，以及一条简短的自然语言 summary。summary 增加第三种可检索模态：用户问 “how is X authorized”，summary 里可能提到 “authz”，即便代码里只有 `check_permission`。

Retrieval 是 hybrid 的。一个 query 同时触发 dense 和 BM25 search，合并 top-k，再交给 cross-encoder re-ranker（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序结果送入 long-context synthesizer（带 prompt caching 的 Claude Sonnet 4.7，或自托管 Llama 3.3 70B），并要求每个 claim 都用文件和行号范围引用。没有 citation 的回答会被 post-filter 拒绝。

增量新鲜度是基础设施问题。Git push 触发 diff：哪些文件变了，哪些 symbol 变了。只有文本变化的 chunk 需要重新 embedding。受影响的跨文件 symbol edges（imports、method calls）会重新计算。这样每次提交后索引保持一致，而不需要重新处理 200 万行代码。

## Architecture / 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Stack / 技术栈

- Parsing: tree-sitter，包含 17 种 language grammars（Python、TS、Rust、Go、Java、C++ 等）
- Dense embeddings: Voyage-code-3（hosted）或 nomic-embed-code-v1.5（self-host），bge-code-v1 fallback
- Sparse index: Tantivy (Rust)，BM25F，按 symbol name 和 body 设置 field weight
- Vector DB: Qdrant 1.12 with hybrid search；或 pgvector + pgvectorscale（适合低于 50M vectors 的团队）
- Chunk summary model: Claude Haiku 4.5 或 Gemini 2.5 Flash，使用 prompt caching
- Re-ranker: Cohere rerank-3 或自托管 bge-reranker-v2-gemma-2b
- Orchestration: ingestion 用 LlamaIndex Workflows，query agent 用 LangGraph
- Synthesizer: Claude Sonnet 4.7（1M context）with prompt caching
- Symbol graph: Neo4j（managed）或 kuzu（embedded），用于 import 和 call edges
- Observability: 每个 retrieval + synthesis step 写 Langfuse spans

## Build It / 动手构建

1. **Ingestion walker.** 在每个 push hook 上遍历 git history。收集 changed files。对每个文件用 tree-sitter 解析，抽取 function 和 class nodes 及其完整 source span。输出 chunk records `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk summarizer.** 把 chunks 批量送入 Haiku 4.5，system preamble 使用 prompt caching。Prompt: "Summarize this function in one sentence, naming its public contract and side effects." 把 summary 与 chunk 一起保存。

3. **Embedding pool.** 两个并行队列：dense（Voyage-code-3 batch 128）和 summary（同一模型，但输入 summary string）。把 vectors 写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 index.** Tantivy field-weighted index：symbol name weight 4，symbol body weight 1，summary weight 2。这样既能回答 “find the function named X”，也能回答 “find the function that does X”。

5. **Symbol graph.** 对每个 chunk 记录 edges：imports（本文件使用 repo Z 的 symbol Y）、calls（本函数调用 class C 的 method M）、inheritance。存入 kuzu。查询时用它跨 repo 边界扩展 retrieval。

6. **Query agent.** 三节点 LangGraph。`retrieve` 并行触发 dense + BM25，并按 (repo, path, symbol) 去重。`rerank` 对 top-50 跑 cross-encoder，保留 top-10。`synth` 用 reranked chunks 作为上下文调用 Claude Sonnet 4.7，缓存 system prompt，并要求 file:line citations。

7. **Citation enforcement.** 解析模型输出；任何没有 `(repo/path:start-end)` anchor 的 claim 都标记为 re-ask 或直接丢弃。只把带 citation 的回答返回给用户。

8. **Incremental re-index.** 每个 webhook 计算 symbol-level diff。只有文本变化的 chunks 重新 embedding。只有 imports 变化的 chunks 重新计算 symbol edges。目标：一个 2M-LOC fleet 中，50-file push 在 60 秒内完成 re-index 并可搜索。

9. **Eval.** 标注 100 个跨 repo 问题及其 gold file:line answers。测量 MRR@10、nDCG@10、citation faithfulness（可验证 anchor 的 claim 占比）和 p50/p99 latency。

## Use It / 应用它

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It / 交付它

交付物是 `outputs/skill-codebase-rag.md`。给定一组 repos，它会搭起 ingestion pipeline、hybrid index 和 query agent，并对任何跨 repo 问题返回带引用的答案。评分标准：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Retrieval quality | 100-question held-out set 上的 MRR@10 和 nDCG@10 |
| 20 | Citation faithfulness | 回答中带可验证 file:line anchors 的 claim 占比 |
| 20 | Latency and scale | 已索引语料规模上 10k QPS 的 p95 query latency |
| 20 | Incremental indexing correctness | 50-file commit 从 git push 到可搜索的时间 |
| 15 | UX and answer formatting | Citation clickability、snippet previews、follow-up affordance |
| **100** | | |

## Exercises / 练习

1. 把 Voyage-code-3 换成自托管 nomic-embed-code。测量 MRR@10 delta。报告启用 re-ranking 后差距是否缩小。

2. 向语料注入 20% generated code（LLM-produced boilerplate）并重新评估。观察 retrieval poisoning。给 payload 添加 "generated" flag，并下调这些 hits 的权重。

3. 在你的语料规模下 benchmark Qdrant hybrid search 与 pgvector + pgvectorscale。报告 batch size 1 的 p99。

4. 添加 sampling-based drift check：每周重跑 100-question eval。若 MRR@10 下降超过 5% 则告警。

5. 扩展到跨语言 symbol resolution：一个 Python 函数通过 gRPC 调用 Go service。用 symbol graph 把它们连接起来。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | “Function-level splits” | 在 tree-sitter node 边界切代码，而不是固定 token 窗口 |
| Hybrid search | “Dense + sparse” | 并行运行 BM25 和 vector search，合并 top-k 后 rerank |
| Cross-encoder rerank | “Second-stage rank” | 把 (query, candidate) 成对打分的模型，比 cosine 更准确 |
| Prompt caching | “Cached system prompt” | 2026 Claude / OpenAI 功能，可让重复 prefix tokens 最高打 90% 折扣 |
| Symbol graph | “Code graph” | 跨文件和 repo 的 imports、calls、inheritance edges |
| Citation faithfulness | “Grounded answer rate” | 用户点击 anchor 并阅读引用 span 后能验证的 claim 占比 |
| Incremental re-index | “Push-to-search time” | 从 git push 到 changed symbols 可查询的 wall-clock 时间 |

## Further Reading / 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨 repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 capstone 的 reference deep-dive
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业 symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal-platform reference
