# Embedding Models — The 2026 Deep Dive / Embedding Models：2026 深入剖析

> Word2Vec 给你每个词一个 vector。现代 embedding models 给你每段 passage 一个 vector，跨语言，还能提供 sparse、dense、multi-vector 视角，并按索引预算调整尺寸。选错了，你的 RAG 就会检索错内容。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 dense、sparse、multi-vector embedding 与 late interaction
- 理解 language coverage、context length、dimension budget、Matryoshka truncation 和 hosted/open 权衡
- 使用 Sentence-BERT、BGE-M3 与 MTEB 做 embedding 选择和评估
- 识别 query prefix、asymmetric encoding、context truncation、latency tail 等生产陷阱

## The Problem / 问题

你的 RAG 系统 40% 时间检索错 passage。罪魁祸首很少是 vector database 或 prompt，而是 embedding model。

2026 年选择 embedding，要跨五个轴做取舍：

1. **Dense vs sparse vs multi-vector.** 每段 passage 一个 vector、每个 token 一个 vector，还是稀疏加权词袋。
2. **Language coverage.** 只做英语时，monolingual English models 仍然胜出。Corpus 混合语言时，multilingual models 胜出。
3. **Context length.** 512 tokens vs 8,192 vs 32,768，而且真实有效容量通常只有标称最大值的 60-70%。
4. **Dimension budget.** 3,072 个 float full precision = 每 vector 12 KB。100M vectors 时，存储约 $1,300/month。Matryoshka truncation 可以砍掉 4 倍。
5. **Open vs hosted.** Open-weight 意味着你控制 stack 和数据。Hosted 意味着你用控制权换 always-latest。

这一课会点名这些取舍，让你基于证据选择，而不是跟随上季度流行模型。

## The Concept / 概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings.** 每段 passage 一个 vector（通常 384-3,072 维）。Cosine similarity 按语义邻近度排序 passages。OpenAI `text-embedding-3-large`、BGE-M3 dense mode、Voyage-3。默认选择。

**Sparse embeddings.** SPLADE 风格。Transformer 为每个 vocab token 预测权重，再把大部分置零。结果是 |vocab| 大小的 sparse vector。像 BM25 一样捕捉 lexical matching，但 term weights 是学习出来的。适合 keyword-heavy queries。

**Multi-vector (late interaction).** ColBERTv2、Jina-ColBERT。每个 token 一个 vector。用 MaxSim 打分：对每个 query token，找最相似的 document token，并求和。存储和打分更贵，但在长 query 和领域 corpus 上胜出。

**BGE-M3: all three at once.** 单个模型同时输出 dense、sparse、multi-vector representations。三者可以独立查询；scores 通过 weighted sum 融合。2026 年想从一个 checkpoint 获得灵活性时的默认选择。

**Matryoshka Representation Learning.** 训练时让 vector 的前 N 维本身也是有用 embedding。把 1,536 维 vector 截到 256 维，可能只损失约 1% accuracy，却节省 6 倍存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 都支持。

### The MTEB leaderboard tells a partial story / MTEB leaderboard 只讲了一部分故事

Massive Text Embedding Benchmark 在发布时（2022）包含 8 类任务共 56 个 tasks，MTEB v2 扩展到 100+ tasks。2026 年初，Gemini Embedding 2 在 retrieval 上领先（67.71 MTEB-R）。Cohere embed-v4 在 general 上领先（65.2 MTEB）。BGE-M3 是 open-weight multilingual 领先者（63.0）。Leaderboard 必要但不充分——永远要在你的领域上 benchmark。

### The three-tier pattern / 三层模式

| Use case / 用例 | Pattern / 模式 |
|----------|---------|
| 快速 first-pass | Dense bi-encoder（BGE-M3, text-3-small） |
| 提升 recall | Sparse（SPLADE, BGE-M3 sparse）+ RRF fuse |
| Top-50 precision | Multi-vector（ColBERTv2）或 cross-encoder reranker |

多数生产 stacks 会三者都用。

## Build It / 动手构建

### Step 1: baseline — dense embeddings with Sentence-BERT / 第 1 步：baseline：用 Sentence-BERT 做 dense embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 让 dot product 等于 cosine similarity。始终设置它。

### Step 2: Matryoshka truncation / 第 2 步：Matryoshka truncation

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后要重新 normalize。Nomic v1.5、OpenAI text-3、Voyage-4 经过训练，在前几档维度上基本无损。非 Matryoshka models（原始 Sentence-BERT）截断后会显著退化。

### Step 3: BGE-M3 multi-functionality / 第 3 步：BGE-M3 多功能输出

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次 inference call，三个 indexes。Score fusion：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调这些权重。

### Step 4: MTEB eval on a custom task / 第 4 步：在自定义任务上做 MTEB eval

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在 *有代表性* 的子集上运行候选模型。不要只相信 leaderboard rank——你的领域很重要。

### Step 5: hand-rolled cosine from scratch / 第 5 步：从零手写 cosine

见 `code/main.py`。它使用 averaged Hashing Trick embeddings（stdlib-only）。无法和 transformer embeddings 竞争，但展示了形状：tokenize → vector → normalize → dot product。

## Pitfalls / 常见坑

- **Same model for query and doc.** 一些模型（Voyage、Jina-ColBERT）使用 asymmetric encoding，query 和 document 走不同路径。始终检查 model card。
- **Missing prefix.** `bge-*` models 需要在 query 前加 `"Represent this sentence for searching relevant passages: "`。忘记会掉 3-5 points recall。
- **Over-trimming Matryoshka.** 1,536 → 256 通常安全。1,536 → 64 不安全。要在 eval set 上验证。
- **Context truncation.** 多数模型会静默截断超过 max length 的输入。长文档需要 chunking（见 lesson 23）。
- **Ignoring latency tail.** MTEB scores 会隐藏 p99 latency。一个 600M model 可能比 335M model 高 2 分，但每次 query 贵 3 倍。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 只做英文，快速，API | `text-embedding-3-large` 或 `voyage-3-large` |
| Open-weight，英文 | `BAAI/bge-large-en-v1.5` |
| Open-weight，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| Long context（32k+） | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU-only deployment | Nomic Embed v2（137M params, MoE） |
| Storage-constrained | Matryoshka-truncated + int8 quantization |
| Keyword-heavy queries | 加 SPLADE sparse，与 dense 做 RRF-fuse |

2026 pattern：从 BGE-M3 或 text-3-large 开始，在你的领域用 MTEB 评估。如果某个 domain-specific model 高出 3 分以上，再替换。

## Ship It / 交付它

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Exercises / 练习

1. **Easy / 简单。** 用 `bge-small-en-v1.5` 以 full dim（384）编码 100 个句子，再截断到 Matryoshka 128。用 10 个 queries 测量 MRR drop。
2. **Medium / 中等。** 在你领域的 500 passages 上比较 BGE-M3 dense、sparse 和 colbert。哪种在 recall@10 上胜出？RRF fusion 是否超过最佳单模式？
3. **Hard / 困难。** 在你的 top-2 domain tasks 上运行 MTEB，比较三个候选模型。报告 MTEB score、100-query batch 的 p99 latency 和 $/1M queries。选择 Pareto-optimal 的那个。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Dense embedding | 那个 vector | 每段文本一个固定大小 vector。用 cosine similarity 排序。 |
| Sparse embedding | 学出来的 BM25 | 每个 vocab token 一个权重，大部分为 0，端到端训练。 |
| Multi-vector | ColBERT-style | 每个 token 一个 vector；MaxSim scoring；index 更大，recall 更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是一个可用的小 embedding。 |
| MTEB | Benchmark | Massive Text Embedding Benchmark，发布时 56 tasks，v2 中 100+。 |
| BEIR | Retrieval benchmark | 18 个 zero-shot retrieval tasks；常用于 cross-domain robustness。 |
| Asymmetric encoding | Query ≠ doc path | 模型对 queries 和 documents 使用不同 projections。 |

## Further Reading / 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bi-encoder 论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — leaderboard 论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三种模式的模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — dimension-ladder training objective。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产中的 late interaction。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — live rankings。
