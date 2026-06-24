# Information Retrieval and Search / 信息检索与搜索

> BM25 精确但脆弱。Dense 能撒大网但会漏关键词。Hybrid 是 2026 年默认方案。其他都是调参。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从零实现 BM25、dense retrieval、Reciprocal Rank Fusion 和 hybrid rerank pipeline
- 区分 sparse retrieval、dense retrieval、fusion 与 cross-encoder rerank 的作用
- 使用 Recall@k、MRR、nDCG@k 评估检索质量
- 理解 2026 生产 RAG 中 chunking、parent-doc、HyDE、reranker 与版本化的重要性

## The Problem / 问题

用户输入 "what happens if someone lies to get money"，期待找到真正覆盖它的法规："Section 420 IPC." Keyword search 会完全漏掉它（没有共享 vocabulary）。如果 embeddings 不是在法律文本上训练的，semantic search 也会漏掉。真实搜索必须同时处理两者。

IR 是每个 RAG 系统、每个搜索框、每个文档站点 fuzzy lookup 的底层 pipeline。2026 年生产中有效的架构不是单一方法，而是一串互补方法，每一层都在捕捉前一层的失败。

这一课会构建每个部件，并点名各自能捕捉哪些失败。

## The Concept / 概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。按需要选择。

1. **Sparse retrieval (BM25).** 快，对 exact matches 精确，对语义很差。运行在 inverted index 上。百万文档上每次 query 低于 10ms。能正确处理 statute references、product codes、error messages、named entities。
2. **Dense retrieval.** 把 query 和 documents 编成 vectors。做 nearest neighbor search。能捕捉 paraphrases 和 semantic similarity。会漏掉只差一个字符的精确关键词匹配。配合 FAISS 或 vector DB，每次 query 约 50-200ms。
3. **Fusion.** 合并 sparse 和 dense 的 ranked lists。Reciprocal Rank Fusion (RRF) 是简单默认方案，因为它忽略 raw scores（它们处在不同尺度），只使用 rank positions。如果你知道某个信号在领域中占主导，可以用 weighted fusion。
4. **Cross-encoder rerank.** 从 fusion 取 top-30。运行 cross-encoder（query + document 一起编码，每个 pair 打分）。保留 top-5。Cross-encoder 每个 pair 比 bi-encoder 慢很多，但准确得多。只在 top-30 上运行来摊销成本。

三路检索（BM25 + dense + SPLADE 这类 learned-sparse）在 2026 benchmark 上胜过两路，但需要 learned-sparse indexes 的基础设施。对多数团队，两路加 cross-encoder rerank 是最佳折中方案。

## Build It / 动手构建

### Step 1: BM25 from scratch / 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个参数值得知道。`k1=1.5` 控制 term-frequency saturation；越高，重复词权重越大。`b=0.75` 控制 length normalization；0 忽略文档长度，1 完全归一化。默认值来自 Robertson 原论文推荐，通常不需要调。

### Step 2: dense retrieval with a bi-encoder / 第 2 步：用 bi-encoder 做 dense retrieval

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对 embeddings 做 L2-normalize，这样 dot product 就等于 cosine。`all-MiniLM-L6-v2` 是 384 维，快，而且对多数英文 retrieval 足够强。多语言用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高准确率，用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### Step 3: Reciprocal Rank Fusion / 第 3 步：Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 会削平 rank 差异贡献；更低的 `k` 会让 top ranks 占主导。60 是发表默认值，通常不需要调。

### Step 4: hybrid search + rerank / 第 4 步：hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合在一起。BM25 找 lexical matches。Dense 找 semantic matches。RRF 合并两个 rankings，不需要 score calibration。Cross-encoder 用 query-document pairs 一起重新打分 top-30，捕捉 bi-encoder 漏掉的细粒度相关性。保留 top-5。

### Step 5: evaluation / 第 5 步：评估

| Metric / 指标 | Meaning / 含义 |
|--------|---------|
| Recall@k | 对存在正确文档的 queries，正确文档有多常出现在 top-k 中？ |
| MRR (Mean Reciprocal Rank) | 第一个相关文档的 1/rank 平均值。 |
| nDCG@k | 考虑相关性等级，而不只是二元 relevant/not。 |

对 RAG 来说，retriever 的 **Recall@k** 是最重要的数字。如果正确 passage 不在 retrieved set 中，reader 就无法回答。

调试提示：对失败 queries，比较 sparse 和 dense rankings。如果一个找到了正确文档，另一个没有，你遇到的是 vocabulary mismatch（修复：补缺失的那一半）或 semantic ambiguity（修复：更好的 embeddings 或 reranker）。

## Use It / 应用它

2026 stack：

| Scale / 规模 | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF。不需要单独 DB。 |
| 100k-10M docs | Dense 用 FAISS 或 pgvector，BM25 用 Elasticsearch / OpenSearch。并行运行。 |
| 10M+ docs | Qdrant / Weaviate / Vespa / Milvus，带 hybrid support。Top-30 上 cross-encoder rerank。 |
| Best-quality frontier | Three-way（BM25 + dense + SPLADE）+ ColBERT late-interaction reranking |

无论选什么，都要为 evaluation 预留预算。在 benchmark end-to-end RAG accuracy 前，先 benchmark retrieval recall。Reader 修复不了 retriever 漏掉的内容。

### The hard-won lessons from 2026 production RAG / 2026 生产 RAG 的血汗经验

- **80% 的 RAG 失败来自 ingestion 和 chunking，不是模型。** 团队花几周换 LLM、调 prompt，而 retrieval 每三次 query 就返回一次错误 context。先修 chunking。
- **Chunking strategy 比 chunk size 更重要。** Fixed-size splits 会打断表格、代码和嵌套标题。Sentence-aware 是默认；semantic 或 LLM-based chunking 对技术文档和产品手册值得投入。
- **Parent-doc pattern.** 检索小的 "child" chunks 获得 precision。当同一个 parent section 中出现多个 children 时，替换成 parent block 保留上下文。不需要 retraining 就能稳定提升答案质量。
- **k_rerank=3 通常最优。** 之后每多一个 chunk 都会增加 token cost 和生成延迟，却不提升答案质量。如果 k=8 对你仍然比 k=3 好，说明 reranker 表现不足。
- **HyDE / query expansion.** 从 query 生成 hypothetical answer，对它做 embedding 并检索。它能弥合短问题与长文档之间的表述差距。不训练就能免费提升 precision。
- **Context budget under 8K tokens.** 如果总是撞到这个上限，说明 reranker threshold 太松。
- **Version everything.** Prompts、chunking rules、embedding model、reranker 都要版本化。任何漂移都会安静破坏答案质量。CI gate 要对 faithfulness、context precision 和 unanswered-question rate 阻断 regression。
- **Three-way retrieval（BM25 + dense + SPLADE 这类 learned-sparse）在 2026 benchmarks 上胜过 two-way**，尤其适合 proper nouns 与语义混合的 queries。当基础设施支持 SPLADE indexes 时再上线。

根据 2026 行业测量，好的 retrieval design 能减少 70-90% hallucinations。多数 RAG 性能提升来自更好的 retrieval，而不是 model fine-tuning。

## Ship It / 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Exercises / 练习

1. **Easy / 简单。** 在 500-document corpus 上实现上面的 `hybrid_search`。测试 20 个 queries。比较 BM25-only、dense-only 和 hybrid 在 recall at 5 上的差异。
2. **Medium / 中等。** 增加 MRR 计算。对每个有已知正确文档的 test query，找出 correct doc 在 BM25、dense 和 hybrid rankings 中的 rank。分别报告 MRR。
3. **Hard / 困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上 fine-tune dense encoder。用 500 对 query-document pairs 构建训练集。比较 fine-tune 前后的 recall。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。按 term frequency、IDF 和长度给文档打分。 |
| Dense retrieval | 向量搜索 | 把 query + doc 编成 vectors，找 nearest neighbors。 |
| Bi-encoder | Embedding model | 独立编码 query 和 doc。Query time 快。 |
| Cross-encoder | Reranker model | 把 query + doc 一起编码。慢，但准确。 |
| RRF | Rank fusion | 通过求和 `1/(k + rank)` 合并两个 rankings。 |
| Recall@k | 检索指标 | 相关 doc 出现在 top-k 中的 query 比例。 |

## Further Reading / 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威处理。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，canonical bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — learned-sparse retriever，缩小与 dense 的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction retrieval。
