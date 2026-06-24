# Named Entity Recognition / 命名实体识别

> 把名字抽出来。听起来简单，直到你遇到边界歧义、嵌套实体和领域术语。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 用 BIO 标注把实体抽取转化为序列标注问题
- 从零实现 BIO span 转换、手工特征和 gazetteer baseline
- 理解 rule-based、HMM、CRF、BiLSTM-CRF、transformer NER 的演进逻辑
- 判断经典 NER、transformer fine-tune 与 LLM-based NER 的适用边界

## The Problem / 问题

"Apple sued Google over its iPhone search deal in the US." 这里至少有五个实体：Apple (ORG)、Google (ORG)、iPhone (PRODUCT)、search deal（也许算一个）、US (GPE)。好的 NER 系统会抽出所有实体并给出正确类型。差的系统会漏掉 iPhone，把水果 Apple 和公司 Apple 混淆，还把 "US" 标成 PERSON。

NER 是每条结构化抽取 pipeline 背后的主力。简历解析、合规日志扫描、病历匿名化、搜索查询理解、chatbot response 的 grounding、法律合同抽取。你不一定直接看见它，但几乎总在依赖它。

这一课会从经典路线（rule-based、HMM、CRF）走到现代路线（BiLSTM-CRF，再到 transformer）。每一步都在解决前一步的具体限制。这个演进模式本身就是课程重点。

## The Concept / 概念

**BIO tagging**（或 BILOU）把实体抽取变成序列标注问题。给每个 token 打上 `B-TYPE`（entity 开始）、`I-TYPE`（entity 内部）或 `O`（不属于任何 entity）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token entity 会串起来：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意跨度。

架构演进如下：

- **Rule-based / 基于规则。** Regex + gazetteer 查表。对已知实体 precision 高，对新实体覆盖为零。
- **HMM.** Hidden Markov Model。给定 tag 的 token emission probability，以及 tag-to-tag transition probability。用 Viterbi decode。在标注数据上训练。
- **CRF.** Conditional Random Field。类似 HMM，但它是判别式模型，所以可以混入任意特征（word shape、capitalization、相邻词）。到 2026 年，在低资源部署中仍然是经典生产主力。
- **BiLSTM-CRF.** 用神经特征替代手工特征。LSTM 双向读取句子，上面接 CRF layer 强制 tag sequence 一致。
- **Transformer-based.** 用 token-classification head fine-tune BERT。准确率最好，计算成本最高。

```figure
ner-bio-tagging
```

## Build It / 动手构建

### Step 1: BIO tagging helpers / 第 1 步：BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2: hand-crafted features / 第 2 步：手工特征

对经典（非神经）NER 来说，特征就是核心。常用特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对 proper nouns 是高信号特征。

### Step 3: a simple rule-based + dictionary baseline / 第 3 步：一个简单的规则 + 词典 baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产 gazetteers 通常有从 Wikipedia 和 DBpedia 抓取的数百万条目。覆盖不错，但消歧很差（公司 `Apple` vs 水果 apple）。这就是统计模型胜出的原因。

### Step 4: the CRF step (sketch, not full impl) / 第 4 步：CRF（草图，不完整手写）

如果没有概率论基础，50 行从零实现完整 CRF 并不启发。这里直接使用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 与 L2 regularization。`all_possible_transitions=True` 让模型学习非法序列（例如 `O` 后直接接 `I-ORG`）概率很低，这就是 CRF 在你不手写约束的情况下强制 BIO consistency 的方式。

### Step 5: what a BiLSTM-CRF adds / 第 5 步：BiLSTM-CRF 增加了什么

特征变成学习出来的。输入是 token embeddings（GloVe 或 fastText）。LSTM 从左到右、从右到左读取句子。拼接后的 hidden states 进入 CRF output layer。CRF 仍然强制 tag sequence consistency；LSTM 用学习特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF layer 直接使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工特征 CRF，收益是可测的，但如果你没有几万句标注数据，收益通常比预期小。

## Use It / 应用它

spaCy 开箱提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标成了 `ORG`，不是 `PRODUCT`，因为 spaCy small model 的 product entity 覆盖较弱。Large model（`en_core_web_lg`）会更好。Transformer model（`en_core_web_trf`）还会更好。

基于 BERT 的 NER 可以用 Hugging Face：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X tokens 合并成 span。不加它，你得到的是 token-level labels，需要自己合并。

### LLM-based NER (the 2026 option) / LLM-based NER（2026 年选项）

Zero-shot 和 few-shot LLM NER 在许多领域已经能和 fine-tuned models 竞争；当标注数据稀缺时，它明显更强。

- **Zero-shot prompting.** 给 LLM 一组 entity types 和一个 example schema，要求输出 JSON。开箱可用；新领域准确率中等。
- **ZeroTuneBio-style prompting.** 把任务拆成 candidate extraction → meaning explanation → judgment → re-check。多阶段 prompt（不是 one-shot）能显著提升 biomedical NER 准确率。同样模式也适用于法律、金融和科学领域。
- **Dynamic prompting with RAG.** 每次推理时，从一个小型标注 seed set 中检索最相似的 labeled examples，动态构造 few-shot prompt。2026 benchmarks 中，这会让 GPT-4 biomedical NER F1 比 static prompting 高 11-12%。
- **Per-entity-type decomposition.** 对长文档来说，一次调用抽取所有 entity types 会随着长度增长损失 recall。每个 entity type 单独跑一遍抽取。推理成本更高，但准确率显著更高。这是临床笔记和法律合同的标准模式。

2026 年生产建议：在收集训练数据前，先做一个 LLM zero-shot baseline。很多时候 F1 已经足够好，不需要 fine-tune。

### Where classical NER still wins / 经典 NER 仍然胜出的场景

即使有 LLM，经典 NER 仍会在这些场景胜出：

- 延迟预算低于 50ms。
- 你有数千个标注样本，并且需要 98%+ F1。
- 领域 ontology 稳定，预训练 CRF 或 BiLSTM 可以很好迁移。
- 监管约束要求 on-prem、非生成式模型。

### Where it falls apart / 它会在哪里崩掉

- **Domain shift / 领域迁移。** 用 CoNLL 训练的 NER 放到法律合同上，可能比 gazetteer 还差。要在你的领域上 fine-tune。
- **Nested entities / 嵌套实体。** "Bank of America Tower" 既是 ORG，又是 FACILITY。标准 BIO 无法表示重叠 span。你需要 nested NER（multi-pass 或 span-based models）。
- **Long entities / 长实体。** "United States Federal Deposit Insurance Corporation." Token-level models 有时会切裂它。使用 `aggregation_strategy` 或后处理。
- **Sparse types / 稀疏类型。** Medical NER labels 里有 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不了解。Scispacy 和 BioBERT 是起点。

## Ship It / 交付它

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Exercises / 练习

1. **Easy / 简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证 round-trip consistency。
2. **Medium / 中等。** 在 CoNLL-2003 English NER dataset 上训练上面的 sklearn-crfsuite CRF。使用 `seqeval` 报告 per-entity F1。典型结果约为 84 F1。
3. **Hard / 困难。** 在一个领域特定 NER dataset（医疗、法律或金融）上 fine-tune `distilbert-base-cased`。与 spaCy small model 对比。记录 data leakage checks，并写下让你意外的结果。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| NER | 抽名字 | 给 token spans 标类型（PERSON, ORG, GPE, DATE, ...）。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 在外部。 |
| BILOU | 更细的 BIO | 增加 `L-X`（last）和 `U-X`（unit），让边界更清晰。 |
| CRF | 结构化分类器 | 建模 label 之间的转移，而不只是 emission。强制输出有效序列。 |
| Nested NER | 重叠实体 | 一个 span 与它的子 span 是不同实体。BIO 无法表达。 |
| Entity-level F1 | 正确的 NER 指标 | 预测 span 必须和真实 span 完全匹配。Token-level F1 会高估准确率。 |

## Further Reading / 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。Canonical。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入后来成为标准的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个 attribute 的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。始终使用它。
