# GloVe, FastText, and Subword Embeddings / GloVe、FastText 与 Subword Embeddings

> Word2Vec 为每个词训练一个 embedding。GloVe 分解共现矩阵。FastText 嵌入词的组成片段。BPE 则把这条路接到了 transformer。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 03 (Word2Vec from Scratch)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 GloVe、FastText 与 BPE 分别解决了 Word2Vec 的哪些问题
- 从零构建 GloVe 共现矩阵训练循环、FastText character n-grams 和 BPE merge 流程
- 判断 GloVe、FastText、BPE、WordPiece、SentencePiece 的适用场景
- 识别 tokenizer-model mismatch 与 OOV 相关的生产风险

## The Problem / 问题

Word2Vec 留下了两个开放问题。

第一，另一路研究传统会直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方式本质上更好吗？还是差异只是两种方法处理计数的方式造成的？**GloVe** 回答了这个问题：只要损失函数设计得好，matrix factorization 可以追平甚至超过 Word2Vec，而且训练成本更低。

第二，两种方法都没有处理未见词的方案。`Zoomer-approved`、`dogecoin`、上周才出现的专有名词、稀有词根的各种屈折形式，都处理不了。**FastText** 用 character n-grams 解决了这个问题：一个词是其组成部分（包括 morphemes）的和，所以即使是 out-of-vocabulary words 也能得到合理 vector。

第三，transformer 出现后，问题又变了。Word-level vocabularies 通常到百万级就到头了，而真实语言远比这开放。**Byte-pair encoding (BPE)** 及其亲戚通过学习高频 subword units 的 vocabulary 覆盖一切文本。每个现代 LLM 的现代 tokenizer 都是 subword tokenizer。

这一课会走完这三者，并解释什么场景该选哪一个。

## The Concept / 概念

**GloVe (Global Vectors).** 构建 word-word co-occurrence matrix `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现了多少次。训练 vectors，使 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对 loss 做加权，避免高频 pair 支配训练。完成。

**FastText.** 一个词是 character n-grams 与词本身的和。`where` 会变成 `<wh, whe, her, ere, re>, <where>`。词向量就是这些组件向量的和。训练方式与 Word2Vec 类似。收益：未见词（`whereupon`）可以由已知 n-grams 组合出来。

**BPE (Byte-Pair Encoding).** 从单个 byte（或 character）的 vocabulary 开始。统计 corpus 中所有相邻 pair。把最频繁的 pair 合并成新 token。重复 `k` 次。结果是一个包含 `k + 256` 个 token 的 vocabulary：高频序列（`ing`、`tion`、`the`）是单个 token，稀有词会被拆成熟悉片段。任何句子都能被 tokenize。

## Build It / 动手构建

### GloVe: factorize the co-occurrence matrix / GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

这里有两个活动部件值得点名。加权函数 `f(x) = (x/x_max)^alpha` 会降低非常高频 pair（如 `(the, and)`）的权重，避免它们支配 loss。最终 embedding 是 `W`（center）和 `W_tilde`（context）两个 table 的和。把两者相加是论文中的技巧，通常比只用其中一个效果更好。

### FastText: subword-aware embeddings / FastText：感知 subword 的 embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词用一组 n-grams 表示（通常是 3 到 6 个字符）。词 embedding 是它的 n-gram embeddings 之和。做 skip-gram 训练时，把 Word2Vec 中的单个 vector 替换成这个表示即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对未见词来说，只要它的一部分 n-grams 已知，你仍然能得到 vector。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以两者会靠近。

### BPE: learned subword vocabulary / BPE：学习出来的 subword vocabulary

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮会合并最常见的相邻 pair。迭代足够多次后，高频子串（`low`、`est`、`tion`）会成为单个 token，稀有词也会被干净地拆开。

真实的 GPT / BERT / T5 tokenizers 会学习 30k-100k 个 merges。结果是：任何文本都能 tokenized 成由已知 ID 组成、长度有界的序列，永远没有 OOV。

## Use It / 应用它

实践中，你很少自己训练这些模型，而是加载预训练 checkpoint。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer 时代的 BPE-style subword tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记 word boundary（GPT-2 约定）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### When to pick which / 如何选择

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 需要预训练通用 word vectors，且不需要 OOV 容忍 | GloVe 300d |
| 需要预训练通用 word vectors，并且必须处理拼写错误 / 新词 / 形态丰富语言 | FastText |
| 任何进入 transformer 的场景（训练或推理） | 模型发布时自带的 tokenizer。不要替换。 |
| 从零训练自己的 language model | 先在你的 corpus 上训练 BPE 或 SentencePiece tokenizer |
| 使用线性模型做生产文本分类 | 仍然先用 TF-IDF。见 Lesson 02。 |

## Ship It / 交付它

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Exercises / 练习

1. **Easy / 简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard overlap。你应该会看到它们共享大量片段（`pla`、`lay`、`play`），这就是 FastText 能在形态变体之间迁移的原因。
2. **Medium / 中等。** 扩展 `learn_bpe`，追踪 vocabulary growth。画出 tokens-per-corpus-character 随 merge 数量变化的曲线。你应该会看到一开始快速压缩，随后渐近到约 2-3 chars per token。
3. **Hard / 困难。** 在 Shakespeare 全集上训练一个 1k-merge BPE。比较常见词与稀有专有名词的 tokenization。测量训练前后的 average tokens per word。写下让你意外的发现。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Co-occurrence matrix | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 周围窗口中出现的频次。 |
| Subword | 词的一部分 | Character n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | Byte-pair encoding | 反复合并最常见相邻 pair，直到 vocabulary 达到目标大小。 |
| OOV | Out of vocabulary | 模型从未见过的词。Word2Vec/GloVe 会失败。FastText 和 BPE 可以处理。 |
| Byte-level BPE | 对原始 byte 做 BPE | GPT-2 的方案。Vocabulary 从 256 个 bytes 开始，所以没有任何东西会 OOV。 |

## Further Reading / 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，至今仍是 loss 推导的最佳解释。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 把 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中到底有什么不同。
