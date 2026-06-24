# Multilingual NLP / 多语言 NLP

> 一个模型，100+ 种语言，其中大多数没有训练数据。Cross-lingual transfer 是 2020 年代最实用的奇迹。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 shared vocabulary、shared representation、zero-shot transfer 与 few-shot fine-tuning
- 使用 XLM-R 做 zero-shot cross-lingual classification
- 使用 multilingual sentence embeddings 构建跨语言相似度与检索
- 识别 source-language choice、tokenization fertility 与低资源语言覆盖带来的风险

## The Problem / 问题

英语有数十亿标注样本。乌尔都语有几千。Maithili 几乎没有。任何服务全球用户的实用 NLP 系统，都必须在没有 task-specific training data 的长尾语言上工作。

Multilingual models 通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型把高资源语言中学到的能力迁移到低资源语言上。在英语 sentiment analysis 上 fine-tune 后，它可以开箱对乌尔都语给出相当好的 sentiment predictions。这就是 zero-shot cross-lingual transfer，它重塑了 NLP 面向世界交付的方式。

这一课会点名 tradeoffs、canonical models，以及多语言新团队最容易踩的一个决策：选择用于迁移的 source language。

## The Concept / 概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Shared vocabulary / 共享 vocabulary。** Multilingual models 使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece tokenizer。Vocabulary 是共享的：同一个 subword unit 可以表示相关语言中的同一个 morpheme。英语和意大利语中的 `anti-` 会得到同一个 token。

**Shared representation / 共享表示。** 在多语言 masked language modeling 上预训练的 transformer，会学到让不同语言中语义相似句子产生相似 hidden states。mBERT、XLM-R、NLLB 都表现出这一点。英语 "cat" 的 embedding 会靠近法语 "chat" 和西班牙语 "gato"，完整句子 embedding 也类似。

**Zero-shot transfer / 零样本迁移。** 在一种语言（通常英语）的标注数据上 fine-tune 模型。推理时在模型支持的任意其他语言上运行。不需要目标语言标签。对类型相近的语言效果强，对远距离语言较弱。

**Few-shot fine-tuning / 小样本微调。** 增加 100-500 个目标语言标注样本。分类任务准确率会跳到英语 baseline 的 95-98%。这是 multilingual NLP 中性价比最高的杠杆。

## The models / 模型

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | 在 Wikipedia 上训练。第一个实用 multilingual LM。低资源较弱。 |
| XLM-R | 2019 | 100 languages | 在 CommonCrawl 上训练（远大于 Wikipedia）。设定 cross-lingual baseline。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 languages | 带 1M-token vocabulary 的 XLM-R（对比 250k）。低资源更好。 |
| mT5 | 2020 | 101 languages | 用于 multilingual generation 的 T5 架构。 |
| NLLB-200 | 2022 | 200 languages | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 languages + 13 programming | 多语言训练的开源 176B LLM。 |
| Aya-23 | 2024 | 23 languages | Cohere 的 multilingual LLM。阿拉伯语、印地语、斯瓦希里语强。 |

按用例选择。分类任务用 XLM-R-base 作为理性默认。生成任务按 translation vs open generation 选择 mT5 或 NLLB。LLM 风格任务可以用 Aya-23 或 Claude，并显式做 multilingual prompting。

## The source-language decision (2026 research) / Source-language 选择（2026 研究）

多数团队默认用英语作为 fine-tuning source。近期研究（2026）显示这经常是错的。

Language similarity 比 raw corpus size 更能预测 transfer quality。对斯拉夫语目标，德语或俄语经常胜过英语。对印度语族目标，印地语经常胜过英语。**qWALS** similarity metric（2026，基于 World Atlas of Language Structures features）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是另一个更早的方法，它用语言相似性、corpus size 和亲缘关系组合，为候选 source languages 排名。

实用规则：如果 target language 有类型学上接近的高资源亲属语言，先尝试在那个语言上 fine-tune，再与英语 fine-tune 比较。

## Build It / 动手构建

### Step 1: zero-shot cross-lingual classification / 第 1 步：zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一个 API。XLM-R 在 NLI 数据上训练后，通过 entailment trick 可以很好迁移到分类。

### Step 2: multilingual embedding space / 第 2 步：multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译句会在 embedding space 中靠近。另一个不同英文句子会更远。这就是 cross-lingual retrieval、clustering 和 similarity 能工作的原因。

### Step 3: few-shot fine-tuning strategy / 第 3 步：few-shot fine-tuning 策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对 100-500 个目标语言样本来说，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认。更高学习率会让 multilingual alignment 崩塌，最后得到一个 English-only model。

## Evaluation that actually works / 真正有效的评估

- **Per-language accuracy on held-out sets.** 不要聚合。聚合会隐藏长尾。
- **Benchmark against monolingual baseline.** 对数据足够的语言，从零训练的 monolingual model 有时胜过 multilingual model。要测试。
- **Entity-level tests.** 目标语言中的 named entities。Multilingual models 对远离拉丁脚本的 scripts 常常 tokenization 较弱。
- **Cross-lingual consistency.** 两种语言中的同义输入应该产生相同预测。测量差距。

## Use It / 应用它

2026 stack：

| Task / 任务 | Recommended / 推荐 |
|-----|-------------|
| 100 种语言分类 | XLM-R-base（约 270M）fine-tuned |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 200 种语言翻译 | `facebook/nllb-200-distilled-600M`（见 lesson 11） |
| Generative multilingual | Claude, GPT-4, Aya-23, mT5-XXL |
| 低资源语言 NLP | XLM-V，或在相关高资源语言上做 domain-specific fine-tune |

只要性能重要，就为目标语言 fine-tuning 留预算。Zero-shot 是起点，不是终点。

### The tokenization tax (what goes wrong for low-resource languages) / Tokenization 税：低资源语言会出什么问题

Multilingual models 在所有语言上共享一个 tokenizer。这个 vocabulary 是在英语、法语、西班牙语、中文、德语占主导的 corpus 上训练的。对主导集合之外的任何语言，三种税会悄悄叠加：

- **Fertility tax.** 低资源语言文本每个词会被切成比英语多得多的 tokens。一个印地语句子可能需要等价英语句子的 3-5 倍 tokens。这 3-5 倍会吞掉 context window、训练效率和延迟。
- **Variant recovery tax.** 每个 typo、diacritic variant、Unicode normalization mismatch 或大小写变体，在 embedding space 中都成了冷启动的无关序列。模型无法学到 native speaker 觉得显然的 orthographic correspondences。
- **Capacity spillover tax.** 前两种税会消耗 context positions、layer depth 和 embedding dimensions。留给真实 reasoning 的容量，系统性地少于高资源语言从同一模型得到的容量。

实践症状：你的模型在 Hindi 上训练正常，loss curve 看起来对，eval perplexity 也合理，但生产输出微妙地错。Morphology 在句子中途崩掉，稀有屈折形式仍然无法恢复。**你不能靠数据规模弥补坏 tokenizer。**

缓解：选择对目标语言覆盖好的 tokenizer（XLM-V 的 1M-token vocabulary 就是直接修复）；训练前在 held-out target text 上验证 tokenization fertility；对真正长尾脚本使用 byte-level fallback（SentencePiece `byte_fallback=True`、GPT-2-style byte-level BPE），确保永远没有 OOV。

## Ship It / 交付它

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Exercises / 练习

1. **Easy / 简单。** 在英语、法语、印地语、阿拉伯语中，每种语言 10 个句子，运行 zero-shot classification pipeline。分别报告 accuracy。你应该看到法语强、印地语还可以、阿拉伯语波动较大。
2. **Medium / 中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言 corpus 上构建 cross-lingual retriever。用英语 query，检索任意语言文档。测量 recall@5。
3. **Hard / 困难。** 对 Hindi classification task 比较 English-source 与 Hindi-source fine-tuning。两种设置都使用 500 个目标语言样本做 few-shot fine-tuning。报告哪个 source 产生更好的 Hindi accuracy，以及差距多大。这就是 LANGRANK thesis 的微型版本。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Multilingual model | 一个模型，多种语言 | 语言之间共享 vocabulary 和 parameters。 |
| Cross-lingual transfer | 训练一种语言，运行另一种语言 | 在 source 上 fine-tune，在 target 上评估，不需要 target-language labels。 |
| Zero-shot | 没有目标语言标签 | 不在目标语言上 fine-tune 的迁移。 |
| Few-shot | 少量目标语言标签 | 用 100-500 个目标语言样本做 fine-tuning。 |
| mBERT | 第一个 multilingual LM | 在 Wikipedia 上预训练的 104-language BERT。 |
| XLM-R | 标准 cross-lingual baseline | 在 CommonCrawl 上预训练的 100-language RoBERTa。 |
| NLLB | Meta 的 200-language MT | No Language Left Behind。包含 55 种低资源语言。 |

## Further Reading / 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启 cross-lingual transfer 研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的 multilingual LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK source-language 论文。
