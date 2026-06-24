# Text Summarization / 文本摘要

> Extractive systems 告诉你文档说了什么。Abstractive systems 告诉你作者想表达什么。任务不同，坑也不同。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 extractive summarization 与 abstractive summarization 的建模方式和风险
- 从零实现 TextRank，并使用 BART 做 abstractive summary
- 使用 ROUGE、BERTScore、G-Eval 等指标评估摘要
- 识别 hallucination、entity swap、number drift、polarity flip 等 factuality 风险

## The Problem / 问题

一篇 2,000 词新闻文章出现在你的 feed 里。你需要 120 词抓住重点。你可以从文章中挑出最重要的三句话（extractive），也可以用自己的话重写内容（abstractive）。两者都叫 summarization，但它们是完全不同的问题。

Extractive summarization 是 ranking problem。给每个句子打分，返回 top-`k`。因为输出逐字来自原文，所以总是语法正确。风险是遗漏散落在文章不同位置的内容。

Abstractive summarization 是 generation problem。Transformer 在输入条件下生成新文本。输出流畅、压缩率高，但可能 hallucinate 源文档中没有的事实。风险是自信地编造。

这一课会构建两者，并讲清各自拥有的失败模式。

## The Concept / 概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive / 抽取式。** 把文章视为图：节点是句子，边是相似度。对图运行 PageRank（或类似算法），按句子与其他句子的连接程度打分。最高分句子就是摘要。标准实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**Abstractive / 生成式。** 在 document-summary pairs 上 fine-tune transformer encoder-decoder（BART、T5、Pegasus）。推理时，模型读取文档，并通过 cross-attention 逐 token 生成摘要。Pegasus 特别使用 gap-sentence pretraining objective，因此不需要太多 fine-tuning 就很适合 summarization。

评估使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）。ROUGE-1 和 ROUGE-2 衡量 unigram 与 bigram overlap。ROUGE-L 衡量 longest common subsequence。越高越好，但 40 ROUGE-L 已经算“好”，50 是“非常强”。每篇论文都会报告三者。使用 `rouge-score` package。

## Build It / 动手构建

### Step 1: TextRank (extractive) / 第 1 步：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

两件事值得点名。Similarity function 使用 log-normalized word overlap，这是原始 TextRank 变体。TF-IDF vectors 的 cosine 也能用。Damping factor 0.85 和 iteration count 是 PageRank 默认值。

### Step 2: abstractive with BART / 第 2 步：用 BART 做 abstractive summary

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail corpus 上 fine-tuned。它开箱生成新闻风格摘要。其他领域（科学论文、对话、法律）应使用对应 Pegasus checkpoint，或在目标数据上 fine-tune。

### Step 3: ROUGE evaluation / 第 3 步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终开启 stemming。否则 "running" 和 "run" 会被当成不同词，ROUGE 会低估。

### Beyond ROUGE (2026 summarization eval) / ROUGE 之外（2026 摘要评估）

ROUGE 统治 summarization metric 二十年，但到 2026 年仅靠它不够。NLG 论文的大规模 meta-analysis 显示：

- **BERTScore**（contextual embedding similarity）在 2023 年前后崛起，现在多数 summarization papers 会与 ROUGE 一起报告。
- **BARTScore** 把 evaluation 当作 generation：给定 source，让 pretrained BART 给 summary 打似然分数。
- **MoverScore**（contextual embeddings 上的 Earth Mover's Distance）在 2025 summarization benchmarks 中居于首位，因为它比 ROUGE 更能捕捉语义重叠。
- **FactCC** 和 **QA-based faithfulness** 在 2021-2023 常见，现在经常被 **G-Eval** 替代（一个 GPT-4 prompt chain，用 chain-of-thought reasoning 给 coherence、consistency、fluency、relevance 打分）。
- Rubric 设计好时，**G-Eval** 和类似 LLM-judge 方法与人类判断一致率约 80%。

生产建议：报告 ROUGE-L 做 legacy comparison，报告 BERTScore 做 semantic overlap，报告 G-Eval 做 coherence 和 factuality。用 50-100 个人工标注摘要校准。

### Step 4: the factuality problem / 第 4 步：事实性问题

Abstractive summaries 容易 hallucination。Extractive summaries 的 hallucination 风险低得多，因为输出逐字来自 source；但如果 source sentences 脱离上下文、过时或引用顺序错误，它们仍然可能误导。这是生产系统在合规相关内容中仍然偏好 extractive methods 的最大原因。

需要点名的 hallucination 类型：

- **Entity swap / 实体替换。** Source 写 "John Smith." Summary 写 "John Brown."
- **Number drift / 数字漂移。** Source 写 "25,000." Summary 写 "25 million."
- **Polarity flip / 极性翻转。** Source 写 "rejected the offer." Summary 写 "accepted the offer."
- **Fact invention / 事实发明。** Source 没提 CEO。Summary 写 CEO 批准了。

有效的评估方法：

- **FactCC.** 在 source sentence 与 summary sentence 的 entailment 上训练的二分类器。预测 factual/not-factual。
- **QA-based factuality.** 让 QA model 回答 source 中有答案的问题。如果 summary 支持不同答案，就 flag。
- **Entity-level F1.** 比较 source 和 summary 中的 named entities。只出现在 summary 中的 entities 可疑。

任何面向用户且 factuality 重要的场景（新闻、医疗、法律、金融），extractive 是更安全的默认选择。Abstractive 必须在 loop 中加入 factuality check。

## Use It / 应用它

2026 stack：

| Use case / 用例 | Recommended / 推荐 |
|---------|-------------|
| 新闻，3-5 句英文摘要 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或 tuned T5 |
| 多文档、长文本 | 任何 32k+ context 的 LLM，配 prompt |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，结构上低 hallucination 风险 | TextRank 或 `sumy` 的 LSA / LexRank |

如果计算不是限制，长上下文 LLM 在 2026 年通常胜过专用模型。代价是成本和可复现性；专用模型输出更稳定。

## Ship It / 交付它

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Exercises / 练习

1. **Easy / 简单。** 在 5 篇新闻文章上运行 TextRank。把 top-3 sentences 与 reference summary 对比，测量 ROUGE-L。你应该会在 CNN/DailyMail 风格文章上看到 30-45 ROUGE-L。
2. **Medium / 中等。** 实现 entity-level factuality：从 source 和 summary 中抽取 named entities（spaCy），计算 summary 中 source entities 的 recall，以及 summary entities 相对 source 的 precision。高 precision、低 recall 表示安全但简短；低 precision 表示 hallucinated entities。
3. **Hard / 困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、factuality（用 entity F1）和每条摘要成本。记录两者各自胜出的地方。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive | 挑句子 | 从 source 中逐字返回句子。不会 hallucinate。 |
| Abstractive | 重写 | 在 source 条件下生成新文本。可能 hallucinate。 |
| ROUGE | 摘要指标 | System output 与 reference 之间的 n-gram / LCS overlap。 |
| TextRank | Graph-based extractive | 在 sentence similarity graph 上运行 PageRank。 |
| Factuality | 是否正确 | Summary claims 是否被 source 支持。 |
| Hallucination | 编造内容 | Summary 中 source 不支持的内容。 |

## Further Reading / 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — extractive canonical paper。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和 gap-sentence objective。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — factuality landscape 论文。
