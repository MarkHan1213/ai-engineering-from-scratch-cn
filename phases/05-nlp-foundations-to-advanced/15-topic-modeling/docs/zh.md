# Topic Modeling — LDA and BERTopic / 主题模型：LDA 与 BERTopic

> LDA：documents 是 topics 的混合，topics 是 words 的分布。BERTopic：documents 在 embedding space 中聚类，clusters 就是 topics。目标相同，分解方式不同。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 LDA 的 generative story 与 BERTopic 的 embedding-clustering pipeline
- 使用 scikit-learn 训练 LDA，并使用 BERTopic 训练神经主题模型
- 用 c_v coherence、topic diversity 和人工抽样评估 topic quality
- 根据文档长度、语义需求、算力和 multi-topic distribution 需求选择 LDA、NMF 或 BERTopic

## The Problem / 问题

你有 10,000 条客服工单、50,000 篇新闻文章，或 200,000 条 tweets。你需要知道这批文本在讲什么，但不能逐条阅读。你没有已标注类别，甚至不知道类别有多少。

Topic modeling 可以在无监督下回答这个问题。给它一个 corpus，它会返回一组小而连贯的 topics，以及每篇文档在这些 topics 上的分布。

两类算法占主导。LDA（2003）把每篇文档视为 latent topics 的混合，把每个 topic 视为 words 上的分布。Inference 是 Bayesian。当你需要 mixed-membership topic assignments 和可解释的 word-level probability distributions 时，它今天仍然会上生产。

BERTopic（2020）用 BERT 编码 documents，用 UMAP 降维，用 HDBSCAN 聚类，再通过 class-based TF-IDF 提取 topic words。它在短文本、社交媒体，以及语义相似性比词重叠更重要的场景中胜出。一个 document 只得到一个 topic，这是它对长文内容的限制。

这一课会建立两者直觉，并说明给定 corpus 时该选哪一个。

## The Concept / 概念

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA generative story / LDA 生成故事。** 每个 topic 是 words 上的分布。每篇 document 是 topics 的混合。要生成 document 中的一个 word，先从 document 的 topic mixture 中采样一个 topic，再从该 topic 的 word distribution 中采样一个 word。Inference 反过来：给定观测到的 words，推断每篇 document 的 topic distribution，以及每个 topic 的 word distribution。Collapsed Gibbs sampling 或 variational Bayes 负责数学。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每一行和为 1（document 的 topic mixture）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每一行和为 1（topic 的 word distribution）。

**BERTopic pipeline.**

1. 用 sentence transformer（例如 `all-MiniLM-L6-v2`）编码每篇 document。得到 384-dim vectors。
2. 用 UMAP 降维到约 5 维。BERT embeddings 维度太高，不适合直接聚类。
3. 用 HDBSCAN 聚类。Density-based，会产生可变大小 clusters 和一个 "outlier" label。
4. 对每个 cluster，在该 cluster 的 documents 上计算 class-based TF-IDF，提取 top words。

输出是每篇 document 一个 topic（外加 -1 outlier label）。也可以通过 HDBSCAN 的 probability vector 得到 soft membership。

## Build It / 动手构建

### Step 1: LDA via scikit-learn / 第 1 步：用 scikit-learn 跑 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：移除 stopwords；用 min_df 和 max_df 过滤稀有词与到处出现的词；使用 CountVectorizer（不是 TfidfVectorizer），因为 LDA 期望原始计数。

### Step 2: BERTopic (production) / 第 2 步：BERTopic（生产）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤会丢掉 BERTopic 的 outlier bucket（HDBSCAN 无法聚类的 documents）。`min_topic_size` 控制 HDBSCAN 的最小 cluster size；BERTopic 库默认是 10。本例为了课程规模明确设为 15。超过 10,000 文档的 corpus，增加到 50 或 100。

### Step 3: evaluation / 第 3 步：评估

两种方法都会输出 topic words。问题是这些词是否连贯。

- **Topic coherence (c_v).** 在 sliding-window contexts 上计算 top-word pairs 的 NPMI（normalized pointwise mutual information），把分数聚合成 topic vectors，再用 cosine similarity 比较。越高越好。使用 `gensim.models.CoherenceModel`，参数 `coherence="c_v"`。
- **Topic diversity.** 所有 topics top words 中 unique words 的比例。越高越好（topics 不重叠）。
- **Qualitative inspection.** 阅读每个 topic 的 top words。它们是否命名了一个真实东西？人类判断仍然是最后防线。

## When to pick which / 如何选择

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 短文本（tweets、reviews、headlines） | BERTopic |
| 有 topic mixtures 的长文档 | LDA |
| 没有 GPU / 算力有限 | LDA 或 NMF |
| 需要 document-level multi-topic distributions | LDA |
| 需要 LLM integration 做 topic labeling | BERTopic（直接支持） |
| 资源受限 edge deployment | LDA |
| 追求最大语义连贯性 | BERTopic |

最大的实践因素是文档长度。BERT embeddings 会截断；LDA 计数可以处理任意长度。对超过 embedding model context 的文档，要么 chunk + aggregate，要么使用 LDA。

## Use It / 应用它

2026 stack：

- **BERTopic.** 短文本和任何语义重要场景的默认选择。
- **`gensim.models.LdaModel`.** 经典生产 LDA，成熟、久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`.** 实验用的简易 LDA。
- **NMF.** Non-negative matrix factorization。LDA 的快速替代，在短文本上质量相近。
- **Top2Vec.** 与 BERTopic 设计类似。社区较小，但部分 benchmark 表现不错。
- **FASTopic.** 更新，更适合超大 corpus，速度快于 BERTopic。
- **LLM-based labeling.** 先用任意聚类，再 prompt 模型为每个 cluster 命名。

## Ship It / 交付它

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Exercises / 练习

1. **Easy / 简单。** 在 20 Newsgroups dataset 上用 5 个 topics 拟合 LDA。打印每个 topic 的 top 10 words。手工给每个 topic 命名。算法找到了真实类别吗？
2. **Medium / 中等。** 在同一个 20 Newsgroups subset 上拟合 BERTopic。比较它与 LDA 的 topic 数量、top words 和定性 coherence。哪一个更干净地呈现真实类别？
3. **Hard / 困难。** 在你的 corpus 上计算 LDA 与 BERTopic 的 c_v coherence。分别运行 5、10、20、50 topics。绘制 coherence vs topic count。报告哪种方法对 topic count 更稳定。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Topic | Corpus 在讲的东西 | Words 上的 probability distribution（LDA）或相似 documents 的 cluster（BERTopic）。 |
| Mixed membership | 文档属于多个 topics | LDA 为每篇文档分配所有 topics 上的分布。 |
| UMAP | 降维 | 保留局部结构的 manifold learning；BERTopic 中使用。 |
| HDBSCAN | 密度聚类 | 找可变大小 clusters；为 outliers 产生 "noise" label (-1)。 |
| c_v coherence | Topic quality metric | Sliding windows 中 top topic words 的平均 pointwise mutual information。 |

## Further Reading / 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 及相关指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考。示例很好。
