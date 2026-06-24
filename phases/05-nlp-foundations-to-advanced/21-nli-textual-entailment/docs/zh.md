# Natural Language Inference — Textual Entailment / 自然语言推理：文本蕴含

> "t entails h" 表示人类读到 t 后会认为 h 为真。NLI 预测 entailment / contradiction / neutral。表面无聊，生产中却是承重结构。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 NLI 的 entailment、contradiction、neutral 三类关系
- 使用 pretrained NLI model 做文本蕴含判断和 zero-shot classification
- 把 NLI 用作 RAG faithfulness 与摘要 hallucination 检查
- 识别 hypothesis-only shortcut、lexical overlap heuristic、template sensitivity 和 domain mismatch 等风险

## The Problem / 问题

你构建了 summarizer，它生成了 summary。你怎么知道 summary 没有 hallucination？

你构建了 chatbot，它回答 "yes."。你怎么知道这个答案被 retrieved passage 支持？

你需要按主题分类 10,000 篇新闻文章，但没有训练标签。能不能复用现有模型？

这三个问题都会归约到 Natural Language Inference。NLI 问的是：给定 premise `t` 和 hypothesis `h`，`h` 是被 `t` 蕴含、与 `t` 矛盾，还是 neutral（无关）？

- **Hallucination check:** `t` = source document，`h` = summary claim。不是 entailment = hallucination。
- **Grounded QA:** `t` = retrieved passage，`h` = generated answer。不是 entailment = fabrication。
- **Zero-shot classification:** `t` = document，`h` = verbalized label（"This is about sports"）。Entailment = predicted label。

一个任务，三种生产用途。这就是每个 RAG evaluation framework 底层都会带 NLI model 的原因。

## The Concept / 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**The three labels / 三个标签。**

- **Entailment.** `t` → `h`。"The cat is on the mat" entails "There is a cat."
- **Contradiction.** `t` → ¬`h`。"The cat is on the mat" contradicts "There is no cat."
- **Neutral.** 无法推出任一方向。"The cat is on the mat" 对 "The cat is hungry." 是 neutral。

**Not logical entailment / 不是形式逻辑蕴含。** NLI 是 *natural* language inference，也就是典型人类读者会推断什么，不是严格逻辑。"John walked his dog" 在 NLI 中 entails "John has a dog"，但严格一阶逻辑只有在你公理化 possession 后才允许这么推。

**Datasets / 数据集。**

- **SNLI**（2015）。570k 人工标注 pairs，premises 是图片 captions。领域窄。
- **MultiNLI**（2017）。433k pairs，覆盖 10 个 genres。2026 年标准训练 corpus。
- **ANLI**（2019）。Adversarial NLI。人类专门写来打破现有模型的例子。更难。
- **DocNLI, ConTRoL**（2020–21）。Document-length premises。测试 multi-hop 和 long-range inference。

**The architecture / 架构。** Transformer encoder（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` representation 输入 3-way softmax。在 MNLI 上训练，在 held-out benchmarks 上评估，in-distribution pairs 可以得到 90%+ accuracy。

**Zero-shot via NLI / 通过 NLI 做 zero-shot。** 给定 document 和 candidate labels，把每个 label 转成 hypothesis（"This text is about sports"）。计算每个 hypothesis 的 entailment probability，取最大。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## Build It / 动手构建

### Step 1: run a pretrained NLI model / 第 1 步：运行 pretrained NLI model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产 NLI 中，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源默认。DeBERTa-v3 在 leaderboard 上领先。

### Step 2: zero-shot classification / 第 2 步：zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认 template 是 "This example is about {label}."。可以用 `hypothesis_template` 自定义。不需要训练数据，不需要 fine-tuning，开箱可用。

### Step 3: faithfulness check for RAG / 第 3 步：RAG faithfulness 检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS faithfulness 的核心。把生成答案拆成 atomic claims。对每个 claim 与 retrieved context 做检查。报告被 entail 的比例。

### Step 4: hand-rolled NLI classifier (conceptual) / 第 4 步：手写 NLI classifier（概念版）

查看 `code/main.py` 中的 stdlib-only toy：用 lexical overlap + negation detection 比较 premise 和 hypothesis。它无法与 transformer models 竞争，但展示了任务形状：两个文本输入，3-way label 输出，loss = `{entail, contradict, neutral}` 上的 cross-entropy。

## Pitfalls / 常见坑

- **Hypothesis-only shortcuts.** 模型只看 hypothesis 就能在 SNLI 上以约 60% 准确率预测 label，因为 "not"、"nobody"、"never" 与 contradiction 相关。检测 label leakage 时，这是强 baseline。
- **Lexical overlap heuristic.** Subsequence heuristic（“每个子序列都被蕴含”）能通过 SNLI，但会在 HANS/ANLI 上失败。要使用 adversarial benchmarks。
- **Document-length degradation.** Single-sentence NLI models 在 document-length premises 上会掉 20+ F1。长上下文用 DocNLI-trained models。
- **Zero-shot template sensitivity.** "This example is about {label}"、"{label}"、"The topic is {label}" 可以让 accuracy 摆动 10+ points。要调 template。
- **Domain mismatch.** MNLI 训练在通用英语上。法律、医疗、科学文本需要 domain-specific NLI models（例如 SciNLI、MedNLI）。

## Use It / 应用它

2026 stack：

| Use case / 用例 | Model |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification（轻量） | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multilingual | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG hallucination detection | RAGAS / DeepEval 内部的 NLI layer |

2026 meta-pattern：NLI 是文本理解的 duct tape。只要你需要判断 “A 是否支持 B？”或 “A 是否与 B 矛盾？”，先用 NLI，再考虑多一次 LLM call。

## Ship It / 交付它

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Exercises / 练习

1. **Easy / 简单。** 在 20 个手写的 (premise, hypothesis, label) triples 上运行 `facebook/bart-large-mnli`，覆盖三类标签。测量 accuracy。加入 adversarial "subsequence heuristic" traps（"I did not eat the cake" vs "I ate the cake"），看模型是否被打破。
2. **Medium / 中等。** 在 100 条 AG News headlines 上比较 zero-shot templates：`"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`。报告 accuracy swing。
3. **Hard / 困难。** 构建 RAG faithfulness checker：atomic-claim decomposition + 每个 claim 做 NLI。在 50 个带 gold context 的 RAG-generated answers 上评估。测量相对人工标签的 false-positive 和 false-negative rates。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | Premise-hypothesis 关系的 3-way classification。 |
| RTE | Recognizing Textual Entailment | NLI 的旧名称；同一任务。 |
| Entailment | "t implies h" | 给定 t，典型读者会认为 h 为真。 |
| Contradiction | "t rules out h" | 给定 t，典型读者会认为 h 为假。 |
| Neutral | "undecided" | t 到 h 无法推出任何方向。 |
| Zero-shot classification | 把 NLI 当分类器 | 把 labels verbalize 成 hypotheses，选 max entailment。 |
| Faithfulness | 答案是否被支持？ | 对 (retrieved context, generated answer) 做 NLI。 |

## Further Reading / 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI benchmark。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 NLI 主力模型。
