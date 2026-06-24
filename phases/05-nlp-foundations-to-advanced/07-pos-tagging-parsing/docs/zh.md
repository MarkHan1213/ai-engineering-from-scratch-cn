# POS Tagging and Syntactic Parsing / 词性标注与句法解析

> 语法曾经不流行。后来每条 LLM pipeline 都需要验证结构化抽取，于是它又回来了。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 POS tagging、constituency parsing 与 dependency parsing 的区别
- 实现 most-frequent-tag baseline 和 bigram HMM POS tagger
- 理解 Viterbi decoding 如何选择最高概率 tag sequence
- 使用 spaCy 读取 POS、dependency relation 和 head 信息，并判断生产场景何时直接调用库

## The Problem / 问题

Lesson 01 提到，lemmatization 需要 part-of-speech tag。如果不知道 `running` 是动词，lemmatizer 就无法把它还原为 `run`。如果不知道 `better` 是形容词，就无法把它还原为 `good`。

这句话背后藏着一个完整子领域。Part-of-speech tagging 会分配语法类别。Syntactic parsing 会恢复句子的树结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年打磨这两件事。后来深度学习把它们压成了预训练 transformer 上的 token-classification task，研究社区就转向别处了。

应用社区没有转向。每条结构化抽取 pipeline 仍然在底层使用 POS 和 dependency trees。LLM 生成的 JSON 会按语法约束验证。问答系统会用 dependency parses 拆解查询。机器翻译质量评估会检查 parse tree 对齐。

值得掌握。这一课会介绍 tagsets、baselines，以及你应该停止手写、改用 spaCy 的位置。

## The Concept / 概念

**POS tagging** 会给每个 token 标一个语法类别。**Penn Treebank (PTB)** tagset 是英语默认方案。它有 36 个 tags，对普通读者来说区分很细：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 动词过去式、`VBZ` 第三人称单数现在时，等等。**Universal Dependencies (UD)** tagset 更粗（17 个 tags），并且语言无关；它成为 cross-lingual work 的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing** 会产生一棵树。主要有两种风格：

- **Constituency parsing / 成分句法解析。** 名词短语、动词短语、介词短语彼此嵌套。输出是一棵由非终结类别（NP、VP、PP）和词作为叶子组成的树。
- **Dependency parsing / 依存句法解析。** 每个词都有一个依赖的 head word，并且边上有语法关系标签。输出是一棵树，每条边都是 `(head, dependent, relation)` triple。

Dependency parsing 在 2010s 胜出，因为它能更干净地跨语言泛化，尤其适合自由词序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It / 动手构建

### Step 1: most-frequent-tag baseline / 第 1 步：most-frequent-tag baseline

最笨但有效的 POS tagger：对每个词，预测它在训练集中最常出现的 tag。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown corpus 上，这个 baseline 能达到约 85% accuracy。不算好，但任何严肃模型都不应该低于这个下限。

### Step 2: bigram HMM tagger / 第 2 步：bigram HMM tagger

建模序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：transition probabilities（给定前一个 tag 的当前 tag 概率）和 emission probabilities（给定 tag 的 word 概率）。用计数和 Laplace smoothing 估计两者。用 Viterbi（tag lattice 上的动态规划）decode。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 上的 bigram HMM 大约能达到 93% accuracy。从 85% 到 93% 的提升主要来自 transition probabilities：模型学到了 `DET NOUN` 常见，而 `NOUN DET` 少见。

### Step 3: why modern taggers beat this / 第 3 步：现代 tagger 为什么更强

Transition + emission probabilities 是局部的。它无法捕捉 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie." 中是动词。带任意特征（suffix、word shape、前后词、词本身）的 CRF 能到约 97%。BiLSTM-CRF 或 transformer 能到 98%+。

这个任务的上限由标注者分歧决定。在 Penn Treebank 上，人类标注者一致率约 97%。超过 98% 的模型很可能是在 test set 上过拟合。

### Step 4: dependency parsing sketch / 第 4 步：dependency parsing 草图

从零完整实现 dependency parsing 超出本课范围；经典教材处理见 Jurafsky and Martin。需要知道两类经典方法：

- **Transition-based** parsers（arc-eager、arc-standard）像 shift-reduce parser：读取 tokens，把它们 shift 到栈上，再执行创建 arcs 的 reduce actions。Greedy decoding 快。经典实现是 MaltParser。现代神经版本是 Chen and Manning 的 transition-based parser。
- **Graph-based** parsers（Eisner's algorithm、Dozat-Manning biaffine）会给每条可能的 head-dependent edge 打分，然后选择 maximum spanning tree。更慢，但更准确。

对多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就会浮现。

## Use It / 应用它

每个生产 NLP 库都会把 POS 和 dependency parsers 作为标准 pipeline 的一部分。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快、准，并与 tokenization + NER + lemmatization 集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（dependency relation）。
- **Stanford NLP (stanza)**。Stanford 对 CoreNLP 的继任者。60+ 语言上的 state-of-the-art。
- **trankit**。基于 transformer，UD accuracy 好。
- **NLTK**。`pos_tag`。可用、较慢、偏旧。适合教学。

### Where this still matters in 2026 / 2026 年它仍然重要的地方

- **Lemmatization.** Lesson 01 需要 POS 才能正确 lemmatize。永远如此。
- **Structured extraction from LLM outputs.** 验证生成句子是否满足语法约束（例如 subject-verb agreement、必需 modifiers）。
- **Aspect-based sentiment.** Dependency parses 能告诉你哪个 adjective 修饰哪个 noun。
- **Query understanding.** "movies directed by Wes Anderson starring Bill Murray" 可以通过 parse 拆成结构化约束。
- **Cross-lingual transfer.** UD tags 和 dependency relations 语言无关，能支持对新语言的 zero-shot structured analysis。
- **Low-compute pipelines.** 如果不能部署 transformer，POS + dependency parse + gazetteer 会比你想象中走得更远。

## Ship It / 交付它

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Exercises / 练习

1. **Easy / 简单。** 在一个小型 tagged corpus（例如 NLTK 的 Brown subset）上使用 most-frequent-tag baseline，测量 held-out sentences 的 accuracy。验证约 85% 的结果。
2. **Medium / 中等。** 训练上面的 bigram HMM，并报告 per-tag precision/recall。HMM 最容易混淆哪些 tags？
3. **Hard / 困难。** 使用 spaCy dependency parse，从 1000 句样本中抽取 subject-verb-object triples。在 50 个手工标注 triples 上评估。记录抽取在哪里失败（常见于 passives、coordinations 和 elided subjects）。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| POS tag | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准 tagset | 英语特定。细分动词时态和名词数。 |
| Universal Dependencies | 多语言 tagset | 比 PTB 更粗；语言中立；cross-lingual work 的默认方案。 |
| Dependency parse | 句子树 | 每个词有一个 head，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 给定 emissions 和 transitions，找到最高概率 tag sequence。 |

## Further Reading / 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 和 parsing 的经典教材处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言 parser 都会使用的 cross-lingual tagset 和 treebank collection。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个 attribute 的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 把 neural parsers 带入主流的论文。
