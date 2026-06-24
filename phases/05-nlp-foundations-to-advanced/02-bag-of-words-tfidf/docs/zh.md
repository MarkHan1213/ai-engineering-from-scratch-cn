# Bag of Words, TF-IDF, and Text Representation / 词袋、TF-IDF 与文本表示

> 先计数，再思考。到 2026 年，在边界清晰的任务上，TF-IDF 仍然能打赢 embedding。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从零构建 vocabulary、Bag of Words 矩阵和 TF-IDF 表示
- 解释 `TF`、`DF`、`IDF`、smoothing 与 L2 normalization 的作用
- 使用 scikit-learn 配置 `CountVectorizer` 和 `TfidfVectorizer`
- 判断 TF-IDF、embedding 与混合方案分别适合哪些文本分类场景

## The Problem / 问题

模型需要数字，而你手里是字符串。

每条 NLP pipeline 都必须回答同一个问题：如何把长度可变的 token 流变成分类器能消费的固定长度向量。这个领域最早落地的答案，是最笨但有效的方法：数词。做成一个向量。

这个向量支撑过比任何 embedding 模型都更多的生产 NLP：垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP benchmark 的第一个十年。到 2026 年，工程师在窄域分类任务上仍然会优先尝试它。它快、可解释，而且在“词是否出现”就是关键的任务上，效果常常和 400M 参数 embedding 模型难分高下。

这一课会从零构建 bag of words，再构建 TF-IDF。然后展示 scikit-learn 如何用三行代码完成同样的事。最后说明哪种失败模式会迫使你转向 embedding。

## The Concept / 概念

**Bag of Words (BoW) / 词袋** 会丢掉顺序。对每个文档，统计每个词表词出现了多少次。向量长度等于 vocabulary size。位置 `i` 是第 `i` 个词的计数。

**TF-IDF** 会重新加权 BoW。一个词如果出现在每个文档里，就没什么信息量，所以降低它的权重。一个词如果在整个 corpus 中很少见，但在某个文档中频繁出现，就是信号，所以提高它的权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档内词频，`df` 是 document frequency（包含该词的文档数），`N` 是总文档数。`log` 会让高频词的权重保持在可控范围。

关键性质：两者都会产生稀疏向量，并且每个维度可解释。你可以查看训练好分类器的权重，读出哪些词会把文档推向哪个类别。一个 768 维 BERT embedding 做不到这一点。

```figure
bow-tfidf
```

## Build It / 动手构建

### Step 1: build the vocabulary / 第 1 步：构建 vocabulary

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：tokenized documents 列表（任何 word-level tokenizer 都可以；本课 `code/main.py` 使用了简化的小写版本）。输出：`{word: index}` dict。稳定插入顺序意味着 index 0 是第一篇文档中首次出现的第一个词。不同工具约定不同；scikit-learn 会按字母排序。

### Step 2: bag of words / 第 2 步：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是 vocabulary index。元素 `[i][j]` 表示“第 `j` 个词在第 `i` 篇文档中出现了多少次”。Doc 1 里的 `cat` 是 2，因为它确实出现了两次。Doc 0 里的 `ran` 是 0，因为它没有出现。

### Step 3: term frequency and document frequency / 第 3 步：term frequency 和 document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

这里有两个值得点名的 smoothing 技巧。`(n+1)/(d+1)` 避免 `log(x/0)`。最后的 `+1` 保证一个出现在所有文档里的词仍然有 IDF 1（而不是 0），这与 scikit-learn 默认行为一致。其他实现会使用原始的 `log(N/df)`。两者都能工作；平滑版本更友好。

### Step 4: TF-IDF / 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个 vocabulary words（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在三篇文档里，所以 IDF 低。`dog` 只出现一次，所以 IDF 高。向量是稀疏的（大多数值很小），而区分性强的词会凸显出来。

### Step 5: L2-normalize rows / 第 5 步：对行做 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不做归一化时，长文档会得到更大的向量，并主导 similarity score。L2 normalization 会把每个文档放到单位超球面上。行之间的 cosine similarity 现在就是 dot product。

## Use It / 应用它

scikit-learn 提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用里完成 tokenization、vocabulary 和 BoW。`TfidfVectorizer` 再加上 IDF weighting 和 L2 normalization。两者都返回 sparse matrices。对于 100k 文档，dense 版本放不进内存；除非分类器强制要求 dense，否则保持 sparse。

几个会彻底改变结果的 knob：

| Arg / 参数 | Effect / 影响 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含 bigram。通常会提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 篇文档中的词。能在噪声数据上裁剪 vocabulary。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。相当于不用硬编码列表也能近似 stopword removal。 |
| `stop_words="english"` | scikit-learn 内置 stopword 列表。依任务而定，sentiment analysis 不应该丢掉否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)`，而不是原始 `tf`。当某个词在一篇文档里重复很多次时有帮助。 |

### When TF-IDF still wins (as of 2026) / 到 2026 年 TF-IDF 仍然占优的场景

- 垃圾邮件检测、主题标注、日志异常标记。重要的是词是否出现，而不是语义细微差别。
- 低数据量场景（几百个带标签样本）。TF-IDF 加 logistic regression 不需要预训练成本。
- 任何重视延迟的系统。TF-IDF 加线性模型可以在微秒级响应。用 transformer 给文档做 embedding 需要 10-100ms。
- 必须解释预测的系统。查看分类器系数即可。Top positive words 就是理由。

### When TF-IDF fails / TF-IDF 失效的场景

语义盲区。看这两篇文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是负面评价，一个是正面评价。它们的 TF-IDF 重叠只有 `{the, movie, was}`。词袋分类器必须记住 `not` 出现在 `good` 附近会翻转标签。数据足够多时它可以学到，但永远不如理解 syntax 的模型自然。

另一个失败点是推理时的 out-of-vocabulary words。一个用 IMDb 评论训练的 BoW 模型，如果训练集中从未出现过 `Zoomer-approved`，它就完全不知道该怎么处理这个 token。Subword embeddings（lesson 04）能处理，TF-IDF 不能。

### Hybrid: TF-IDF weighted embeddings / 混合方案：TF-IDF 加权 embedding

2026 年中等数据量分类任务的实用默认方案：用 TF-IDF 权重作为 word embeddings 上的 attention。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你同时得到 embedding 的语义能力，以及 TF-IDF 对稀有词的强调。分类器训练在 pooled vector 上。对于 50k 以下带标签样本的 sentiment、topic 和 intent classification，这种方案经常胜过单独使用任一方法。

## Ship It / 交付它

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Exercises / 练习

1. **Easy / 简单。** 在 L2-normalized TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，vocabulary 完全不相交的文档得分为 0.0。
2. **Medium / 中等。** 给 `bag_of_words` 增加 `n-gram` 支持。参数 `n` 产生 `n`-gram 计数。测试 `n=2` 作用于 `["the", "cat", "sat"]` 时，能产生 `["the cat", "cat sat"]` 的 bigram 计数。
3. **Hard / 困难。** 使用 GloVe 100d vectors 构建上面的 TF-IDF-weighted-embedding 混合方案（下载一次并缓存）。在 20 Newsgroups 数据集上，对比 plain TF-IDF、plain mean-pooled embeddings 和混合方案的分类准确率。报告哪种方法在哪些类别上胜出。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一篇文档中 vocabulary words 的计数。丢弃顺序。 |
| TF | 词频 | 一个词在文档中的计数，可选择按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数量。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`。降低到处都出现的词的权重。 |
| Sparse vector | 大部分是 0 | Vocabulary 通常有 10k-100k 个词；任意一篇文档只包含其中很少一部分。 |
| Cosine similarity | 向量夹角 | L2-normalized vectors 的 dot product。1 表示相同，0 表示正交。 |

## Further Reading / 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 标准 API 参考，也解释了每个 knob。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为默认方案十年的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年视角下，老方法何时胜出以及为什么。
