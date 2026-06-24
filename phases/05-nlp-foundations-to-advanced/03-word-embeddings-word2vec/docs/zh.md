# Word Embeddings — Word2Vec from Scratch / Word Embeddings：从零实现 Word2Vec

> 看一个词身边出现什么词，就能认识这个词。把这个想法训练成一个浅层网络，几何结构就会自然浮现。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 distributional hypothesis 如何导出 word embeddings
- 从零生成 skip-gram training pairs，并实现 negative sampling 训练目标
- 使用相似度与 analogy probe 检查 embedding 空间是否合理
- 说明 Word2Vec 在多义词和 OOV 场景下的局限

## The Problem / 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。它不知道它们意思几乎相同。一个只在 `dog` 上训练过的分类器，无法自然泛化到关于 `puppy` 的评论。你可以用同义词列表勉强补丁，但它会在稀有词、领域术语和所有你没预料到的语言上失效。

你想要一种表示，让 `dog` 和 `puppy` 在空间里靠得很近。让 `king - man + woman` 落到 `queen` 附近。让一个在 `dog` 上训练过的模型，免费把一部分信号迁移到 `puppy`。

Word2Vec 给了我们这种空间。两层神经网络，万亿 token 级训练，2013 年发表。架构简单到近乎尴尬，结果却重塑了 NLP 十年。

## The Concept / 概念

**Distributional hypothesis / 分布假说**（Firth, 1957）：“You shall know a word by the company it keeps.” 如果两个词出现在相似上下文里，它们很可能意思相近。

Word2Vec 有两种形式，都在利用这个想法。

- **Skip-gram.** 给定中心词，预测周围词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW (continuous bag of words).** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对稀有词更好，因此成为默认选择。

这个网络只有一个隐藏层，没有非线性。输入是 vocabulary 上的 one-hot vector。输出是 vocabulary 上的 softmax。训练完成后，你丢掉输出层。隐藏层权重就是 embeddings。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

关键技巧：对 100k 词做 softmax 代价太高。Word2Vec 使用 **negative sampling**，把问题变成二分类任务：预测“这个 context word 是否出现在这个 center word 附近，是或否”。每个训练 pair 只采样少量 negative（没有共现的词），而不是对整个 vocabulary 计算 softmax。

```figure
word-vector-arithmetic
```

## Build It / 动手构建

### Step 1: training pairs from a corpus / 第 1 步：从 corpus 生成训练 pair

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内每个 `(center, context)` pair 都是一个 positive training example。

### Step 2: embedding tables / 第 2 步：embedding tables

两个矩阵。`W` 是 center-word embedding table（你最终保留的那个）。`W'` 是 context-word table（通常丢弃，有时会和 `W` 求平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。Vocab size 10k、dim 100 已经很接近真实规模；教学时 50 个词、16 维就足够看见几何结构。

### Step 3: negative sampling objective / 第 3 步：negative sampling 目标函数

对每个 positive pair `(center, context)`，从 vocabulary 中随机采样 `k` 个词作为 negatives。训练目标是让 positive 的 dot product `W[center] · W'[context]` 变高，让 negatives 的 dot product 变低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式其实是：positive pair 上的 logistic loss（希望 sigmoid 接近 1）加上 negative pairs 上的 logistic loss（希望 sigmoid 接近 0）。梯度会流向两个 table。完整推导在原论文里；如果想真正记住，拿纸笔推一遍。

### Step 4: train on a toy corpus / 第 4 步：在 toy corpus 上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在足够大的 corpus 上训练足够多轮后，共享上下文的词会拥有相似的 center embeddings。在 toy corpus 上，你能隐约看到这个效果。在数十亿 token 上，效果会非常明显。

### Step 5: the analogy trick / 第 5 步：analogy 技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300d Google News vectors 上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型理解什么是王室，而是因为向量 `(king - man)` 捕捉了某种类似“royal”的方向，把它加到 `woman` 上，就会落到“royal-female”区域附近。

## Use It / 应用它

从零写 Word2Vec 是为了教学。生产 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真实工作中，你几乎从不自己训练 Word2Vec，而是下载预训练 vectors。

- **GloVe** — Stanford 的 co-occurrence matrix factorization 方法。提供 50d、100d、200d、300d checkpoint。通用覆盖好。Lesson 04 会专门讲 GloVe。
- **fastText** — Facebook 对 Word2Vec 的扩展，把 character n-grams 也嵌入进去。可以通过组合 subwords 处理 out-of-vocabulary words。Lesson 04。
- **Pretrained Word2Vec on Google News** — 300d、300 万词 vocabulary，2013 年发布。现在仍然每天有人下载。

### When Word2Vec still wins in 2026 / 2026 年 Word2Vec 仍然占优的场景

- 轻量级领域检索。在笔记本上用医学摘要训练一小时，就能得到通用模型捕捉不到的专用 vectors。
- Analogy-style feature engineering。`gender_vector = mean(man - woman pairs)`。从其他词中减去它，可以得到更性别中性的轴。公平性研究仍然会用。
- 可解释性。100d 小到可以用 PCA 或 t-SNE 画出来，并且真的看到 cluster 形成。
- 任何必须在端侧无 GPU 推理的场景。Word2Vec lookup 只是取一行矩阵。

### Where Word2Vec fails / Word2Vec 的失败点

多义词墙。`bank` 只有一个 vector。`river bank` 和 `financial bank` 共用它。`table`（电子表格 vs 家具）也共用它。下游分类器无法仅凭这个 vector 区分词义。

Contextual embeddings（ELMo、BERT，以及之后所有 transformer）通过基于上下文为同一个词的每次出现生成不同 vector，解决了这个问题。这就是从 Word2Vec 到 BERT 的跨越：从 static 到 contextual。Phase 7 会讲 transformer 部分。

另一个失败点是 out-of-vocabulary。训练数据里没见过 `Zoomer-approved`，Word2Vec 就完全没有 fallback。fastText 用 subword composition 修复这个问题（lesson 04）。

## Ship It / 交付它

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercises / 练习

1. **Easy / 简单。** 在一个很小的 corpus（20 个关于 cats 和 dogs 的句子）上运行训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的 top 3 中包含 `dog`。如果没有，就增加 epochs 或 vocabulary。
2. **Medium / 中等。** 增加 frequent words subsampling。频率高于 `10^-5` 的词，按频率比例从 training pairs 中丢弃。测量它对 rare-word similarity 的影响。
3. **Hard / 困难。** 在 20 Newsgroups corpus 上训练模型。计算两个 bias axes：`he - she` 和 `doctor - nurse`。把 occupation words 投影到这两个轴上。报告哪些职业的 bias gap 最大。这就是公平性研究常用的 probe。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Word embedding | 把词表示成向量 | 从上下文中学到的 dense、低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对稀有词更好。 |
| Negative sampling | 训练捷径 | 用针对 `k` 个随机词的二分类，替代完整 vocabulary softmax。 |
| Static embedding | 每个词一个 vector | 无论上下文如何，同一个词都是同一个 vector。会在多义词上失败。 |
| Contextual embedding | 上下文敏感 vector | 基于周围词，为每次出现生成不同 vector。Transformer 产生的就是这个。 |
| OOV | Out of vocabulary | 训练中没见过的词。Word2Vec 无法为这类词生成 vector。 |

## Further Reading / 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — negative-sampling 论文。短，而且可读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果原论文数学太密，这是最清楚的梯度推导。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 真正在生产训练中有用的设置。
