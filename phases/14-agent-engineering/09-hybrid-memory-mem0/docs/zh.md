# Hybrid Memory: Vector + Graph + KV (Mem0) / Hybrid Memory：Vector + Graph + KV（Mem0）

> Mem0 (Chhikara et al., 2025) 把 memory 视为三个并行 store：vector 负责 semantic similarity，KV 负责快速事实查询，graph 负责 entity-relationship reasoning。检索时由 scoring layer 融合三者。这是 2026 年生产外部记忆的标准形态。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么单一 store（vector only、graph only、KV only）不足以支撑 Agent memory。
- 说出 Mem0 的三个并行 store，以及每个 store 优化什么。
- 描述 Mem0 的 fusion scoring：relevance、importance、recency；并说明它为什么是 weighted sum，而不是 hierarchy。
- 用 stdlib 实现一个 toy three-store memory：`add()` 同时写三类 store，`search()` 融合结果。

## The Problem / 问题

单一 store 会在三类查询中至少错两类：

- **Semantic similarity**：“上周我们讨论过 agent drift 的什么？”Vector 胜出；KV 和 graph 容易漏。
- **Fact lookup**：“用户的电话号码是什么？”KV 胜出；vector 浪费，graph 过重。
- **Relationship reasoning**：“哪些客户共享同一个 billing entity？”Graph 胜出；vector 和 KV 答不了。

生产 Agent 在一个 session 中会发出全部三类查询。单一 store 对其中两类总是错。Mem0 的贡献是把三类 store 接在统一的 `add` / `search` 表面后面，并用 scoring function 融合。

## The Concept / 概念

### Three stores in parallel / 三个并行 store

Mem0 (arXiv:2504.19413, April 2025) 在 `add(text, user_id, metadata)` 时：

1. 从 text 中提取候选 facts（LLM-driven step）。
2. 把每个 fact 写入 vector store（embedding），用于 semantic search。
3. 把每个 fact 写入 KV store，key 是 (user_id, fact_type, entity)，用于 O(1) lookup。
4. 把每个 fact 写入 graph store（Mem0g），作为 typed edges，支持 relationship queries。

在 `search(query, user_id)` 时：

1. Vector store 按 embedding cosine 返回 top-k。
2. KV store 用 query-derived (user_id, type, entity) 返回 direct hits。
3. Graph store 返回从 query entities 可达的 subgraph。
4. Scoring layer 融合三者。

### Fusion scoring / 融合评分

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance**：vector cosine、KV exact match、graph path weight。
- **Importance**：写入时标注或学习得到（有些 facts 更重要：names、IDs、policies）。
- **Recency**：自上次写入或读取以来的 exponential decay。

权重按产品调。chat agents 提高 `w_recency`；compliance agents 提高 `w_importance`；retrieval agents 提高 `w_relevance`。

### Mem0g and temporal reasoning / Mem0g 与时间推理

Mem0g 增加 conflict detector。新 fact 与已有 edge 矛盾时，已有 edge 会被标记 invalid，但不会删除。Temporal queries（“用户三月份在哪个城市？”）会遍历 valid-at-time subgraph。

这是 Letta invalidation pattern 的 compliance-grade 版本。

### Benchmark numbers / Benchmark 数字

Mem0 论文报告（2025）：

- **LoCoMo**（long-form conversation memory）：91.6
- **LongMemEval**（long-horizon episodic memory）：93.4
- **BEAM 1M**（1M-token memory benchmark）：64.1

比较 baseline（full-context 128k LLM、flat vector store、flat KV）都低 10+ 点。benchmark 不能单独决定选型，operational shape 更重要，但这些数字说明 fusion design 不是小修小补。

### Scope taxonomy / Scope 分类

Mem0 按 scope 拆 memory：

- **User memory**：跨 session 持久化，keyed on `user_id`。
- **Session memory**：只在一个 thread 中持久化。
- **Agent memory**：每个 Agent instance 的状态。

每次写入都选择一个 scope。检索可以跨 scopes 查询，并给不同 scope 设置不同权重。混用 scope 不思考，就是 “assistant 把 Bob 项目告诉 Alice” 这类事故的来源。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Embedding drift。** 前一百次查询看起来正确的 vector results，会随着 corpus 增长退化。对 top-N-used records 做周期性 re-embedding。
- **KV schema creep。** `(user_id, type, entity)` 看起来简单，直到每个团队都添加自己的 `type`。每季度审计 type set。
- **Graph explosion。** 一个嘈杂 extractor 每条消息加 50 条 edges。限制每次 `add` 的 graph writes；丢弃低置信 edges。

## Build It / 动手构建

`code/main.py` 用 stdlib 实现 three-store pattern：

- `VectorStore`：用 naive token-overlap similarity 代替 embedding。
- `KVStore`：以 `(user_id, fact_type, entity)` 为 key 的 dict。
- `GraphStore`：typed edges（subject、relation、object、valid）。
- `Mem0`：顶层 facade，带 `add()`、`search()`、fusion scoring 和 scope-aware retrieval。
- 一个 multi-user、multi-session conversation 的 worked trace。

运行：

```
python3 code/main.py
```

输出会展示三条独立 recall paths 和融合后的 top-k。修改 `main()` 顶部的 scoring weights，观察 ranking 如何变化。

## Use It / 应用它

- **Mem0 (Apache 2.0)**：production-ready。可用 Postgres + Qdrant + Neo4j self-host，也可用 managed cloud。
- **Letta**：core/recall/archival 三层；自带 vector 和 graph backends。
- **Zep**：带 temporal KG 和 fact extraction 的商业替代。
- **Custom builds**：当你需要精确控制 extractor（compliance）或 fusion weights（recency 主导的 voice agents）时使用。

## Ship It / 交付它

`outputs/skill-hybrid-memory.md` 生成 three-store memory scaffold，并接好 fusion scorer、scope taxonomy 和 temporal invalidation。

## Exercises / 练习

1. 用真实 embedding model（sentence-transformers、Ollama、OpenAI embeddings）替换 toy vector similarity。在 synthetic long conversation 上测 recall@10。1000 次写入后 ranking 会 drift 吗？
2. 增加 temporal query：`search(query, as_of=timestamp)`。只返回该时间点之前有效的 records。哪个 store 需要最多改动？
3. 实现 conflict detector：如果 incoming fact 和 graph edge 矛盾，invalidate old edge 并记录二者。用 “user lives in Berlin” -> “user lives in Lisbon” 测试。
4. 把 fusion scorer 扩展一个 `user_feedback` 维度（retrieved records 上的 thumbs-up）。如何防止 gaming（Agent 只返回它已经喜欢的 records）？
5. 阅读 Mem0 docs（`docs.mem0.ai`）。把 toy 移植到 `mem0` client calls。用同样 20 条 test queries 比较 retrieval quality。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Hybrid memory | “Vector plus graph plus KV” | 三个 store 并行写入，检索时融合 |
| Fact extraction | “Memory ingestion” | LLM 把 text 拆成 (entity, relation, fact) tuples 的步骤 |
| Fusion scoring | “Relevance ranking” | relevance、importance、recency 的 weighted sum |
| Scope | “Memory namespace” | user / session / agent，决定谁能看到什么 |
| Mem0g | “Memory graph” | 带 temporal validity 的 typed edges，用于 relationship queries |
| Temporal invalidation | “Soft delete” | 把矛盾 edges 标记 invalid；永不硬删 |
| Embedding drift | “Retrieval rot” | corpus 增长后 vector quality 退化；定期 re-embed |

## Further Reading / 延伸阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 原始论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — production API、SDKs、managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — virtual-context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — three-tier sibling design
