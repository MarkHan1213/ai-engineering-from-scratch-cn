# Question Answering Systems / 问答系统

> 三类系统塑造了现代 QA：extractive 找 span，retrieval-augmented 用文档 grounding，generative 生成答案。每个现代 AI assistant 都是三者的混合。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 extractive QA、open-domain QA、RAG 与 closed-book generative QA
- 使用 pretrained SQuAD model 构建 extractive QA pipeline
- 组合 dense retrieval 与 reader，搭建 retrieval-augmented QA
- 用 answer accuracy、citation accuracy、refusal calibration、retrieval recall 和 RAGAS 评估 QA 系统

## The Problem / 问题

用户输入 "When did the first iPhone launch?"，期待的是 "June 29, 2007." 不是 "Apple's history is long and varied."，也不是孤零零的 "2007"。用户要的是直接、有 grounding、正确的答案。

过去十年里，三种架构主导了 QA。

- **Extractive QA.** 给定 question 和一个已知包含答案的 passage，找出 passage 中 answer span 的 start/end indices。SQuAD 是标准 benchmark。
- **Open-domain QA.** Passage 没有给定。先检索相关 passage，再抽取或生成答案。这是今天每条 RAG pipeline 的基石。
- **Generative / Closed-book QA.** 大语言模型从 parametric memory 中回答。没有 retrieval。推理最快，但事实可靠性最低。

2026 年趋势是混合：检索最好的几个 passages，然后 prompt 生成式模型，让它基于这些 passages 作答。这就是 RAG，lesson 14 会深入 retrieval 半边。本课构建 QA 半边。

## The Concept / 概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive / 抽取式。** 用 transformer（BERT family）一起编码 question 和 passage。训练两个 heads，分别预测 answer 的 start/end token indices。Loss 是有效位置上的 cross-entropy。输出是 passage 中的 span。结构上不会 hallucinate，也结构上无法回答 passage 中没有答案的问题。

**Retrieval-augmented (RAG).** 两阶段。第一阶段 retriever 从 corpus 找 top-`k` passages。第二阶段 reader（extractive 或 generative）使用这些 passages 产生答案。Retriever-reader 拆分让两者可以独立训练和评估。现代 RAG 经常在两者之间加入 reranker。

**Generative / 生成式。** Decoder-only LLM（GPT、Claude、Llama）从 learned weights 作答。没有 retrieval step。对常识问题很好，对稀有或近期事实灾难性不可靠。Hallucination rate 与 pretraining data 中事实频率负相关。

## Build It / 动手构建

### Step 1: extractive QA with a pretrained model / 第 1 步：用 pretrained model 做 extractive QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，其中包含 unanswerable questions。默认情况下，`question-answering` pipeline 即使 null score 更高，也会返回最高分 span；它不会自动返回空答案。要获得显式 "no answer" 行为，需要在 pipeline call 中传 `handle_impossible_answer=True`：只有当 null score 超过所有 span score 时，pipeline 才返回空 answer。无论哪种方式，都要检查 `score` 字段。

### Step 2: a retrieval-augmented pipeline (sketch) / 第 2 步：retrieval-augmented pipeline（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段 pipeline。Dense retriever（Sentence-BERT）通过 semantic similarity 找相关 passages。Extractive reader（RoBERTa-SQuAD）从合并后的 top passages 中抽取 answer span。适合小 corpus。百万文档 corpus 要用 FAISS 或 vector database。

### Step 3: generative with RAG / 第 3 步：带 RAG 的生成式回答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Prompt pattern 很重要。明确要求模型基于 context 作答，并在 context 不包含答案时返回 "I don't know"，相比 naive prompting 可以把 hallucination rates 降低 40-60%。更复杂的模式会增加 citations、confidence scores 和 structured extraction。

### Step 4: evaluation that reflects the real world / 第 4 步：反映真实世界的评估

SQuAD 使用 **Exact Match (EM)** 和 **token-level F1**。EM 是 normalization（小写、去标点、移除冠词）后的严格匹配：预测完全匹配得 1，否则 0。F1 基于 prediction 与 reference 的 token overlap，给部分匹配分。两者都会低估 paraphrase："June 29, 2007" vs "June 29th, 2007" 通常 EM 为 0（ordinal 破坏 normalization），但 F1 仍会因 token overlap 得到不少分。

生产 QA 应关注：

- **Answer accuracy / 答案准确率**（LLM-judged 或 human-judged，因为传统指标捕捉不了语义等价）。
- **Citation accuracy / 引用准确率。** 引用 passage 是否真的支持答案？通过 generated citations 与 retrieved passages 的 string match 可以自动检查一部分。
- **Refusal calibration / 拒答校准。** 当 retrieved passages 中没有答案时，系统是否正确说 "I don't know"？测量 false confidence rate。
- **Retrieval recall / 检索召回。** 评估 reader 之前，先测 retriever 是否把正确 passage 放进 top-`k`。Reader 修复不了缺失 passage。

### RAGAS: the 2026 production eval framework / RAGAS：2026 年生产评估框架

`RAGAS` 是为 RAG systems 设计的，2026 年是上线默认选择。它不需要 gold references，就能评估四个维度：

- **Faithfulness.** 答案中的每个 claim 是否来自 retrieved context？通过 NLI-based entailment 测量。这是主要 hallucination metric。
- **Answer relevance.** 答案是否回应了问题？通过从答案生成 hypothetical questions，再与真实问题比较来测。
- **Context precision.** Retrieved chunks 中有多大比例确实相关？低 precision = prompt 里噪声多。
- **Context recall.** Retrieved set 是否包含所有所需信息？低 recall = reader 无法成功。

Reference-free scoring 允许你在 live production traffic 上评估，而不需要 curated gold answers。对开放问题，再叠加 LLM-as-judge，因为 exact-match metrics 没用。

`pip install ragas`。接入你的 retriever + reader。每个 query 得到四个 scalars。对 regression 做告警。

## Use It / 应用它

2026 stack：

| Use case / 用例 | Recommended / 推荐 |
|---------|-------------|
| 给定 passage，找 answer span | `deepset/roberta-base-squad2` |
| 固定 corpus 上作答，不能接受 closed-book | RAG：dense retriever + LLM reader |
| 文档库实时问答 | RAG with hybrid（BM25 + dense）retriever + reranker（lesson 14） |
| Conversational QA（追问） | LLM with conversation history + 每轮 RAG |
| 高事实性、受监管领域 | 基于权威 corpus 的 extractive；绝不单独用 generative |

到 2026 年，extractive QA 不再时髦，因为 RAG with LLMs 能处理更多情况。它仍然会在需要逐字引用的场景上线：法律研究、监管合规、审计工具。

## Ship It / 交付它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Exercises / 练习

1. **Easy / 简单。** 在 10 个 Wikipedia passages 上搭建上面的 SQuAD extractive pipeline。手写 10 个问题。测量答案正确次数。如果 passages 和 questions 干净，你应该能看到 7-9 个正确。
2. **Medium / 中等。** 增加 refusal classifier。当 top retrieval score 低于某个阈值（例如 0.3 cosine）时，返回 "I don't know"，而不是调用 reader。在 held-out set 上调阈值。
3. **Hard / 困难。** 在你选择的 10,000-document corpus 上构建 RAG pipeline。实现 hybrid retrieval（BM25 + dense），并用 RRF fusion（见 lesson 14）。测量有无 hybrid step 时的 answer accuracy。记录哪类问题受益最多。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive QA | 找答案 span | 在给定 passage 中预测 answer 的 start/end indices。 |
| Open-domain QA | 在 corpus 上 QA | 没有给定 passage；必须先 retrieve 再 answer。 |
| RAG | 先 retrieve 再 generate | Retrieval-augmented generation。Retriever + reader pipeline。 |
| SQuAD | 标准 benchmark | Stanford Question Answering Dataset。EM + F1 metrics。 |
| Hallucination | 编造答案 | Reader output 不被 retrieved context 支持。 |
| Refusal calibration | 知道何时闭嘴 | 系统无法回答时正确说 "I don't know"。 |

## Further Reading / 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — benchmark 论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 中 canonical dense retriever。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 综合 RAG survey。
