# Coreference Resolution / 共指消解

> "She called him. He did not answer. The doctor was at lunch." 三个指代，两个角色，没有人被点名。Coreference resolution 要弄清谁是谁。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 mention、antecedent、cluster、anaphora、cataphora 与 bridging 的含义
- 理解 rule-based、mention-pair、mention-ranking、span-based end-to-end 与 generative coref 架构
- 使用 pretrained coreference model 或 LLM prompt 做共指 baseline
- 评估 CoNLL F1，并识别 singleton、长上下文、gender heuristic 与 LLM drift 风险

## The Problem / 问题

从一篇 300 词文章中抽取 Apple Inc. 的所有 mentions。文章写 "Apple" 时很简单。文章写 "the company"、"they"、"Cupertino's technology giant" 或 "Jobs's firm" 时就难了。如果不把这些 mentions 解析为同一个 entity，你的 NER pipeline 会漏掉 60-80% mentions。

Coreference resolution 会把指向同一个现实实体的所有表达连接到一个 cluster。它是 surface-level NLP（NER、parsing）与 downstream semantics（IE、QA、summarization、KG）之间的胶水。

2026 年它为什么重要：

- Summarization："The CEO announced..." vs "Tim Cook announced..."——summary 应该写出 CEO 名字。
- Question answering："Who did she call?" 需要解析 "she"。
- Information extraction：如果 knowledge graph 里 "PER1 founded Apple" 和 "Jobs founded Apple" 是两条独立记录，那就是错的。
- Multi-document IE：合并多篇文章中同一事件 mentions，就是 cross-document coreference。

## The Concept / 概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**The task / 任务。** 输入：一篇文档。输出：mentions（spans）的聚类，每个 cluster 指向一个 entity。

**Mention types / Mention 类型。**

- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Appositive.** "Tim Cook, Apple's CEO,"

**Architectures / 架构。**

1. **Rule-based (Hobbs, 1978).** 基于 syntactic tree 和语法规则做 pronoun resolution。是好 baseline。在 pronouns 上意外地难被击败。
2. **Mention-pair classifier.** 对每对 mentions (m_i, m_j)，预测它们是否 corefer。用 transitive closure 聚类。2016 年前标准方案。
3. **Mention-ranking.** 对每个 mention，给 candidate antecedents（包括 "no antecedent"）排序，选择 top。
4. **Span-based end-to-end (Lee et al., 2017).** Transformer encoder。枚举长度上限内所有 candidate spans。预测 mention scores。对每个 span 预测 antecedent probability。Greedy clustering。现代默认。
5. **Generative (2024+).** Prompt LLM："List every pronoun in this text and its antecedent." 简单情况效果好，长文档和稀有 referents 上会挣扎。

**The evaluation metrics / 评估指标。** 标准指标有五个（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕捉 clustering quality。通常报告前三者平均值作为 CoNLL F1。2026 年 CoNLL-2012 state-of-the-art 约 83 F1。

**Known hard cases / 已知难点。**

- 指向数页前实体的 definite descriptions。
- Bridging anaphora（"the wheels" → 前面提到的一辆 car）。
- 中文、日语等语言中的 zero anaphora。
- Cataphora（pronoun 出现在 referent 前）："When **she** walked in, Mary smiled."

## Build It / 动手构建

### Step 1: pretrained neural coreference (AllenNLP / spaCy-experimental) / 第 1 步：pretrained neural coreference

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长文档上，你会得到类似结果：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### Step 2: rule-based pronoun resolver (teaching) / 第 2 步：rule-based pronoun resolver（教学）

查看 `code/main.py` 中的 stdlib-only 实现：

1. 抽取 mentions：named entities（capitalized spans）、pronouns（dict lookup）、definite descriptions（"the X"）。
2. 对每个 pronoun，查看前面 K 个 mentions，并按以下因素打分：
   - gender/number agreement（启发式）
   - recency（越近越优先）
   - syntactic role（subjects 优先）
3. 链接最高分 antecedent。

它无法和 neural models 竞争，但展示了搜索空间，以及 end-to-end model 必须做出的决策。

### Step 3: using LLMs for coreference / 第 3 步：用 LLM 做 coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两个失败模式。第一，LLM 会 over-merge（把指向不同人的 "him" 和 "her" 合并）。第二，LLM 会在长文档中静默漏掉 mentions。始终用 span-offset checks 验证。

### Step 4: evaluation / 第 4 步：评估

标准 conll-2012 脚本计算 MUC、B³、CEAF-φ4，并报告平均值。内部 eval 可以从标注 test set 上的 span-level precision/recall 开始，然后增加 mention-linking F1。

## Pitfalls / 常见坑

- **Singleton explosion.** 有些系统把每个 mention 都报成单独 cluster。B³ 比较宽容，MUC 会惩罚。始终检查三个指标。
- **Pronouns in long context.** 2,000 tokens 以上文档性能会掉约 15 F1。要小心 chunk。
- **Gender assumptions.** 硬编码 gender rules 会在非二元 referents、组织、动物上失败。使用 learned models 或 neutral scoring。
- **LLM drift on long docs.** 单次 API call 无法可靠聚类 50+ 段落中的 mentions。使用 sliding-window + merge。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 英文、单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| Cross-document event coref | 专用 end-to-end models（2025–26 SOTA） |
| 快速 LLM baseline | GPT-4o / Claude with structured-output coref prompt |
| 生产 dialog systems | Rule-based fallback + neural primary + critical slots 人审 |

2026 年上线集成模式：先跑 NER，再跑 coref，把 coref clusters 合并进 NER entities。下游任务看到的是每个 cluster 一个 entity，而不是每个 mention 一个 entity。

## Ship It / 交付它

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Exercises / 练习

1. **Easy / 简单。** 在 5 个手写段落上运行 `code/main.py` 里的 rule-based resolver。与 ground truth 比较 mention-link accuracy。
2. **Medium / 中等。** 在一篇新闻文章上使用 pretrained neural coref model。把 clusters 与你的手工标注比较。它在哪里失败了？
3. **Hard / 困难。** 构建 coref-enhanced NER pipeline：先 NER，再通过 coref clusters 合并。在 100 篇文章上测量相对 NER-only 的 entity-coverage improvement。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mention | 一个指代 | 指向 entity 的 text span（name、pronoun、noun phrase）。 |
| Antecedent | "it" 指什么 | 后续 mention 所 corefer 的较早 mention。 |
| Cluster | Entity 的 mentions | 都指向同一个 real-world entity 的 mention 集合。 |
| Anaphora | 向后指代 | 后面的 mention 指向前文（"he" → "John"）。 |
| Cataphora | 向前指代 | 前面的 mention 指向后文（"When he arrived, John..."）。 |
| Bridging | 隐式指代 | "I bought a car. The wheels were bad."（那辆车的 wheels。） |
| CoNLL F1 | Leaderboard 数字 | MUC、B³、CEAF-φ4 F1 scores 的平均值。 |

## Further Reading / 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改善 coref 的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — benchmark。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — rule-based classic。
