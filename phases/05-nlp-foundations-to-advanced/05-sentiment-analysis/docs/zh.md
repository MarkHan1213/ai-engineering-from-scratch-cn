# Sentiment Analysis / 情感分析

> 经典 NLP 任务。关于经典文本分类的大多数关键知识，都会在这里出现。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从零实现 sentiment classification 的 Naive Bayes 与 logistic regression baseline
- 解释 negation、sarcasm、domain vocabulary 和 class imbalance 如何影响结果
- 使用 precision、recall、F1、confusion matrix 和错误样本评估模型
- 判断何时经典 TF-IDF baseline 足够，何时需要 transformer

## The Problem / 问题

"The food was not great." 是正面还是负面？

Sentiment 看起来简单。评论者说喜欢或不喜欢某个东西，把句子打上标签就行。它之所以成为经典 NLP 任务，是因为每个看似简单的例子背后都有难点。否定会翻转含义。讽刺会反转含义。"Not bad at all" 虽然有两个负面编码的词，整体却是正面的。Emoji 携带的信号可能比周围文本更多。领域词汇也重要（音乐评论里的 `tight` 和时尚评论里的 `tight` 含义不同）。

Sentiment 是经典 NLP 的工作实验室。如果你理解每个 naive baseline 为什么有特定失败模式，你就理解了为什么后来需要更丰富的模型。这一课会从零构建 Naive Bayes baseline，加上 logistic regression，并指出让生产 sentiment 变成合规级问题的陷阱。

## The Concept / 概念

经典 sentiment 是两步 recipe。

1. **Represent / 表示。** 把文本转成 feature vector。BoW、TF-IDF 或 n-grams。
2. **Classify / 分类。** 在带标签样本上拟合线性模型（Naive Bayes、logistic regression、SVM）。

Naive Bayes 是最笨但有效的模型。假设在给定 label 的情况下，每个 feature 相互独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时把概率相乘。“Naive” 的独立性假设明显错误，但结果却强得惊人。原因是：在稀疏文本特征和中等数据量下，分类器更关心每个词倾向于哪一边，而不是倾向强度有多精细。

Logistic regression 修复了独立性假设。它为每个 feature 学一个权重，包括负权重。`not good` 作为 bigram feature 可以得到负权重。Naive Bayes 对从没标注过的 bigram 做不到这一点。

```figure
sentiment-logits
```

## Build It / 动手构建

### Step 1: a real mini-dataset / 第 1 步：一个真实感 mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

刻意保持很小。真实工作会用数万条样本（IMDb、SST-2、Yelp polarity）。数学完全一样。

### Step 2: multinomial Naive Bayes from scratch / 第 2 步：从零实现 multinomial Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing（alpha=1.0）就是 Laplace smoothing。没有它，一个在某类中没见过的词概率为 0，log 会爆掉。实践中常用 `alpha=0.01`。`alpha=1.0` 是教学默认值。

### Step 3: logistic regression from scratch / 第 3 步：从零实现 logistic regression

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization 在这里很重要。文本特征是稀疏的；没有 L2，模型会记住训练样本。从 `0.01` 开始再调参。

### Step 4: handling negation (the failure mode) / 第 4 步：处理否定（失败模式）

考虑 "not good" 和 "not bad"。BoW classifier 看到的是 `{not, good}` 和 `{not, bad}`，然后从训练集中哪种组合出现更多来学习。Bigram classifier 会看到 `not_good` 和 `not_bad`，并把它们当成不同 features 学习。通常这就够了。

如果你没有 bigrams，一个更粗糙但有效的修复是 **negation scoping / 否定范围标记**。把否定词后面直到下一个标点前的 token 都加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同 features。分类器可以给它们相反权重。三行预处理，就能在 sentiment benchmark 上带来可测的准确率提升。

### Step 5: evaluation metrics that matter / 第 5 步：真正重要的评估指标

如果类别不均衡，只看 accuracy 会误导。真实 sentiment corpora 通常是 70-80% positive 或 70-80% negative；一个永远预测多数类的分类器能拿到 80% accuracy，但毫无价值。下面这些都要报告：

- **Per-class precision and recall / 每类 precision 与 recall。** 每个类别一组。做 macro-average 后得到一个尊重类别平衡的单数值。
- **Macro-F1（不均衡数据的主指标）。** 各类别 F1 的平均值，等权重。类别不均衡时用它替代 accuracy。
- **Weighted-F1（备选）。** 与 macro 类似，但按类别频率加权。当不均衡本身有业务意义时，与 macro-F1 一起报告。
- **Confusion matrix / 混淆矩阵。** 原始计数。信任任何标量指标前都要看它；它会揭示模型混淆了哪些类别。
- **Per-class error samples / 每类错误样本。** 每类抽 5 个错误预测，读它们。没有什么能替代直接读错误样本。

对于极端不均衡数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC**，不要报告 accuracy。AUPRC 对少数类更敏感，而少数类通常才是你真正关心的（spam、fraud、稀有 sentiment）。

**Common bug to avoid / 要避免的常见 bug。** 在不均衡数据上报告 micro-F1 而不是 macro-F1，会得到一个看起来很高的数字，因为它被多数类支配。Macro-F1 会迫使你看到少数类表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Use It / 应用它

scikit-learn 用六行代码把它正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 加入 bigrams，让 `not_good` 成为 feature。`sublinear_tf=True` 会削弱重复词。这三个 flag 是 SST-2 上 75% 准确率 baseline 和 85% 准确率 baseline 的区别。

### When to reach for a transformer / 什么时候该上 transformer

- 讽刺检测。经典模型在这里会失败，没有例外。
- 情感在长评论中途转向。
- Aspect-based sentiment。"Camera was great but battery was terrible." 你需要把情感归因到具体 aspect。只有 transformer 或 structured output models 能处理。
- 非英语、低资源语言。Multilingual BERT 可以免费给你一个 zero-shot baseline。

如果你需要以上任何一点，直接跳到 Phase 7（transformers deep dive）。否则，TF-IDF + bigrams + negation handling 上的 Naive Bayes 或 logistic regression，就是你的 2026 生产 baseline。

### The reproducibility trap (again) / 可复现性陷阱（再次出现）

重新训练 sentiment models 很常见。重新评估它们则不常见。论文里的 accuracy 数字使用的是特定 split、特定 preprocessing、特定 tokenizer。如果你没有使用完全相同的 pipeline，却把新模型和论文 baseline 比较，会得到误导性 delta。永远在你自己的 pipeline 上重新生成 baseline，不要直接拿论文数字当基准。

## Ship It / 交付它

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Exercises / 练习

1. **Easy / 简单。** 把 `apply_negation` 作为 scikit-learn pipeline 的预处理步骤加入，并在一个小型 sentiment dataset 上测量 F1 delta。
2. **Medium / 中等。** 实现 class-weighted logistic regression（在 scikit-learn 中传 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不均衡数据上测量效果。
3. **Hard / 困难。** 在 sentiment model 的 residuals 上训练第二个分类器，构建 sarcasm detector。记录你的实验设置。当 accuracy 低于 chance 时要提醒读者（2-class sarcasm 的 chance-level 约 50%，多数第一次尝试都会落在这里）。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Polarity | 正面或负面 | 二分类标签；有时扩展到 neutral 或细粒度（5-star）。 |
| Aspect-based sentiment | 每个 aspect 的 polarity | 把情感归因到文本中提到的具体实体或属性。 |
| Negation scoping | 翻转附近 token | 在 "not" 之后直到标点前的 token 上加 `NOT_` 前缀。 |
| Laplace smoothing | 给计数加 1 | 防止 Naive Bayes 中出现零概率 feature。 |
| L2 regularization | 压缩权重 | 在 loss 上加入 `lambda * sum(w^2)`。对稀疏文本特征很关键。 |

## Further Reading / 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — foundational survey。很长，但前四节覆盖了经典方法的全部核心。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 展示 bigrams + Naive Bayes 在短文本上很难被击败的论文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和所有需要调的 knob 的参考。
