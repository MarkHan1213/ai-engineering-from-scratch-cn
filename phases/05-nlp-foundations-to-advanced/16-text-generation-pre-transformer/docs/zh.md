# Text Generation Before Transformers — N-gram Language Models / Transformer 之前的文本生成：N-gram 语言模型

> 如果一个词让模型意外，说明模型不好。Perplexity 把“意外”变成数字。Smoothing 让它保持有限。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 从计数角度解释 n-gram language model、zero-count problem 与 perplexity
- 实现 trigram counts、Laplace smoothing、bigram Kneser-Ney smoothing、sampling 与 perplexity
- 理解 interpolation、backoff、absolute discounting 与 continuation probability 的作用
- 判断 n-gram LM 在 on-device、spell checker、speech/MT rescoring 和 baseline 中的价值

## The Problem / 问题

在 transformers、RNNs、word embeddings 之前，language model 通过统计某个词跟在前 `n-1` 个词后出现的频率来预测下一个词。统计 "the cat" → "sat" 出现 47 次，"the cat" → "jumped" 出现 12 次，"the cat" → "refrigerator" 出现 0 次。归一化后得到概率分布。

这就是 n-gram language model。从 1980 到 2015 年，它支撑了每个语音识别器、拼写检查器和 phrase-based machine translation 系统。当你需要廉价的端侧语言模型时，它今天仍然会运行。

真正有趣的问题是如何处理未见过的 n-grams。原始计数模型会给没见过的任何序列分配零概率，这很灾难，因为句子很长，几乎每个长句都会包含至少一个未见序列。五十年的 smoothing 研究修复了这个问题。Kneser-Ney smoothing 是结果，现代深度学习也继承了它的经验传统。

## The Concept / 概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram probability / N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（trigram 通常取 3，4-gram 取 4）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**The zero-count problem / 零计数问题。** 训练中没见过的任何 n-gram 概率都是 0。2007 年一项 Brown corpus 研究发现，即使是 4-gram model，held-out 4-grams 也有 30% 在训练中没出现过。不做 smoothing，就没法在真实文本上评估。

**Smoothing approaches / Smoothing 方法（按复杂度）：**

1. **Laplace (add-one).** 每个 count 加 1。简单，但对 rare events 很差。
2. **Good-Turing.** 根据 frequency-of-frequencies，把概率质量从高频事件重新分配给未见事件。
3. **Interpolation.** 用可调权重组合 n-gram、(n-1)-gram 等估计。
4. **Backoff.** 如果 n-gram count 为 0，就退回 (n-1)-gram。Katz backoff 对此做归一化。
5. **Absolute discounting.** 从所有 counts 中减去固定 discount `D`，再把质量分给未见事件。
6. **Kneser-Ney.** Absolute discounting 加上一个巧妙的 lower-order model：使用 *continuation probability*（一个词出现在多少种 contexts 中），而不是 raw frequency。

Kneser-Ney insight 很深。"San Francisco" 是常见 bigram。Unigram "Francisco" 大多只出现在 "San" 后面。Naive absolute discounting 会因为 count 高而给 "Francisco" 高 unigram probability。Kneser-Ney 注意到 "Francisco" 只出现在一种 context 中，因此降低它的 continuation probability。结果：一个以 "Francisco" 结尾的新 bigram 会得到恰当的低概率。

**Evaluation: perplexity / 评估：perplexity。** 在 held-out test set 上，每个词平均 negative log-likelihood 的指数。越低越好。Perplexity 为 100 意味着模型困惑程度相当于在 100 个词中均匀选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## Build It / 动手构建

### Step 1: trigram counts / 第 1 步：trigram counts

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是 tokenized sentences 列表。输出是 n-gram counts 和 context counts。`<s>` 与 `</s>` 是句子边界。

### Step 2: Laplace smoothing / 第 2 步：Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个 count 加 1。它能 smooth，但会给未见事件分配过多概率质量，也会伤害 rare-known events。

### Step 3: Kneser-Ney (bigram, interpolated) / 第 3 步：Kneser-Ney（bigram，interpolated）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个活动部件。`continuation_prob` 捕捉“这个词出现在多少种不同 contexts 中？”（Kneser-Ney 创新点）。`lambda_prev` 是 discount 释放出来的概率质量，用来加权 backoff。最终概率是 discounted main term 加 weighted continuation term。

### Step 4: generating text with sampling / 第 4 步：采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。不同 seed 会给出不同输出。想要类似 beam-search 的输出，可以每一步选 argmax（greedy），再加入一个小的 randomness knob（temperature）。

### Step 5: perplexity / 第 5 步：perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown corpus 上，调好的 4-gram KN model perplexity 大约为 140。同一 test set 上 transformer LM 可以到 15-30。差距约 10 倍。这就是领域继续前进的原因。

## Use It / 应用它

- **Classical NLP teaching.** 这是理解 smoothing、MLE 和 perplexity 最清楚的入口。
- **KenLM.** 生产 n-gram library。在重视低延迟的 speech 和 MT systems 中用作 rescorer。
- **On-device autocomplete.** 键盘里的 trigram models。仍然如此。
- **Baselines.** 在声称 neural LM 很好之前，永远先计算 n-gram LM perplexity。如果 transformer 没有大幅超过 KN，就说明哪里错了。

## Ship It / 交付它

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Exercises / 练习

1. **Easy / 简单。** 在 1,000 句 Shakespeare corpus 上训练 trigram LM。生成 20 个句子。它们会局部合理但全局不连贯。这是经典 demo。
2. **Medium / 中等。** 在 Shakespeare held-out split 上实现你的 KN model perplexity，并与 Laplace 对比。你应该看到 KN 的 perplexity 低 30-50%。
3. **Hard / 困难。** 构建 trigram spell corrector：给定 misspelled word 及其上下文，生成修正候选，并按 LM 的 context probability 排序。在 Birkbeck spelling corpus（公开）上评估。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | 连续 `n` 个 tokens 的序列。 |
| Smoothing | 避免零概率 | 重新分配概率质量，让未见事件得到非零概率。 |
| Perplexity | LM 质量指标 | Held-out data 上的 `exp(-average log-prob)`。越低越好。 |
| Backoff | 退回更短上下文 | 如果 trigram count 为零，就用 bigram。Katz backoff 将其形式化。 |
| Kneser-Ney | 最好的 n-gram smoothing | Absolute discounting + lower-order model 的 continuation probability。 |
| Continuation probability | KN 专用 | 按 `w` 出现的 contexts 数量给 `P(w)` 加权，而不是按 raw count。 |

## Further Reading / 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LMs 和 smoothing 的经典处理。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 n-gram smoother 的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产 n-gram LM，2026 年仍用于 latency-sensitive applications。
