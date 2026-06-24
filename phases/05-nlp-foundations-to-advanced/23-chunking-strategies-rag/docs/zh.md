# Chunking Strategies for RAG / RAG 的切块策略

> Chunking configuration 对检索质量的影响和 embedding model 选择一样大（Vectara NAACL 2025）。Chunking 错了，再多 reranking 也救不回来。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 比较 fixed、recursive、semantic、sentence、parent-document、late chunking 与 contextual retrieval
- 实现 fixed/recursive/semantic chunking、parent-child retrieval 与 contextual retrieval
- 根据 query type 选择 chunk size，并用 recall@k 评估
- 识别 overlap cargo cult、min/max 缺失、cross-doc chunking 和 factoid-only eval 等常见问题

## The Problem / 问题

你把一份 50 页合同放进 RAG 系统。用户问："What is the termination clause?" Retriever 返回封面页。为什么？因为模型是在 512-token chunks 上训练的，而 termination clause 位于 20 页后，被 page break 切开，并且本地没有把它与 query 绑定起来的关键词。

修复方法不是“买一个更好的 embedding model”，而是 chunking。多大？要不要 overlap？在哪里切？要不要带周围上下文？

2026 年 2 月 benchmarks 给出了出人意料的结果：

- Vectara 2026 study：recursive 512-token chunking 胜过 semantic chunking，accuracy 69% → 54%。
- Natural Questions 上 SPLADE + Mistral-8B：overlap 没有提供可测收益。
- Context cliff：response quality 在约 2,500 tokens context 附近急剧下降。

“显然”的答案（semantic chunking、20% overlap、1000 tokens）经常是错的。这一课会建立六种策略的直觉，并说明什么时候该用哪一个。

## The Concept / 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.** 每 N 个字符或 tokens 切一次。最简单 baseline。会在句子中间切断。压缩好，连贯性差。

**Recursive.** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 切，再按 `\n`、`.`、空格切。Fallback 干净。2026 年默认方案。

**Semantic.** 对每个句子做 embedding。计算相邻句子的 cosine similarity。在 similarity 低于 threshold 的位置切。能保留主题连贯性。更慢；有时产生 40-token tiny fragments，伤害 retrieval。

**Sentence.** 按句子边界切。每句一个 chunk，或 N 句窗口。在约 5k tokens 以内，效果接近 semantic chunking，成本低得多。

**Parent-document.** 存小 child chunks 用于 retrieval，同时存更大的 parent chunk 用于 context。按 child 检索，返回 parent。退化优雅：即便 child 切得不好，也会返回合理 parent。

**Late chunking (2024).** 先在 token level 对整篇文档做 embedding，再把 token embeddings pool 成 chunk embeddings。保留跨 chunk context。适合 long-context embedders（BGE-M3、Jina v3）。计算成本更高。

**Contextual retrieval (Anthropic, 2024).** 给每个 chunk 前面加上 LLM 生成的文档位置摘要（"This chunk is section 3.2 of the termination clauses..."）。Anthropic 自家 benchmark 中 retrieval 提升 35-50%。Index 成本高。

### The rule that beats every default / 胜过默认参数的规则

让 chunk size 匹配 query type：

| Query type / 查询类型 | Chunk size |
|------------|-----------|
| Factoid（"what is the CEO's name?"） | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

这是 NVIDIA 2026 benchmark 的结论。Chunk 应该大到能包含答案和本地上下文，小到让 retriever 的 top-K 聚焦在答案，而不是 context noise。

## Build It / 动手构建

### Step 1: fixed and recursive chunking / 第 1 步：fixed 与 recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Step 2: semantic chunking / 第 2 步：semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调 `threshold`。太高 → fragments。太低 → 一个巨大的 chunk。

### Step 3: parent-document / 第 3 步：parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键点：dedupe parents。多个 children 可能映射到同一个 parent；全部返回会浪费 context。

### Step 4: contextual retrieval (Anthropic pattern) / 第 4 步：contextual retrieval（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

把 contextualized chunks 建入索引。Query time 时，检索会受益于额外的周边信号。

### Step 5: evaluate / 第 5 步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

永远 benchmark。你 corpus 的“最佳”策略未必符合任何 blog post。

## Pitfalls / 常见坑

- **Chunking evaluated only on factoid queries.** Multi-hop queries 会揭示完全不同的赢家。使用按 query type 分层的 eval set。
- **Semantic chunking without a minimum size.** 会产生 40-token fragments，伤害 retrieval。始终强制 `min_tokens`。
- **Overlap as cargo cult.** 2026 studies 发现 overlap 经常没有收益，却让 index cost 翻倍。要测量，不要假设。
- **No min/max enforcement.** 5 tokens 或 5000 tokens 的 chunk 都会破坏 retrieval。要 clamp。
- **Cross-doc chunking.** 绝不要让一个 chunk 跨越两篇文档。始终按文档切，再合并。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Strategy / 策略 |
|-----------|----------|
| 第一次构建，未知 corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| 大量 cross-reference（合同、论文） | Late chunking 或 contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| 短文本（tweets、reviews） | 一篇文档作为一个 chunk |

从 recursive 512 开始。在 50-query eval set 上测 recall@5。再从那里调。

## Ship It / 交付它

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Exercises / 练习

1. **Easy / 简单。** 用 fixed(512, 0)、recursive(512, 0)、recursive(512, 100) 对一份 20 页文档切块。比较 chunk counts 和 boundary quality。
2. **Medium / 中等。** 在 5 篇文档上构建 30-query eval set。测量 recursive、semantic、parent-document 的 recall@5。谁赢了？与 blog posts 是否一致？
3. **Hard / 困难。** 实现 contextual retrieval。测量相对 baseline recursive 的 MRR 提升。报告 index cost（LLM calls）与 accuracy gain 的权衡。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Chunk | 文档片段 | 会被 embedding、index、retrieved 的子文档单元。 |
| Overlap | 安全边距 | 相邻 chunks 共享的 N tokens；2026 benchmarks 中经常没用。 |
| Semantic chunking | 智能切块 | 在相邻句子 embedding similarity 下降处切分。 |
| Parent-document | 两层 retrieval | 检索小 children，返回更大的 parents。 |
| Late chunking | Embedding 后切块 | 在 token level 嵌入整篇 doc，再 pool 成 chunk vectors。 |
| Contextual retrieval | Anthropic 技巧 | 每个 chunk 前加 LLM-generated summary 后再索引。 |
| Context cliff | 2500-token 墙 | RAG 中约 2.5k context tokens 处观察到质量下降（2026 年 1 月）。 |

## Further Reading / 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产默认。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking 和 embedding choice 一样重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — late chunking 论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 用 LLM-generated context prefixes 提升 35-50% retrieval。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按 query type 选择 chunk size。
