# Entity Linking & Disambiguation / 实体链接与消歧

> NER 找到了 "Paris"。Entity linking 决定它是 Paris, France、Paris Hilton、Paris, Texas，还是特洛伊王子 Paris。没有 linking，你的 knowledge graph 就会一直含混。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 entity linking 的 candidate generation 与 disambiguation 两个子任务
- 构建 alias index，并实现 context overlap 与 embedding-based disambiguation
- 理解 prior+context、BLINK/REL、GENRE 与 LLM-prompted EL 的取舍
- 评估 mention recall、top-1 accuracy 和 NIL detection，并识别 popularity bias 与 KB staleness

## The Problem / 问题

句子写道："Jordan beat the press." 你的 NER 把 "Jordan" 标成 PERSON。很好。但 *哪一个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（Berkeley ML professor——在 ML 论文里这真的会混淆）？
- Jordan（国家）？
- Jordan（希伯来名）？

Entity linking（EL）把每个 mention 解析到 knowledge base 中唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域 KB。两个子任务：

1. **Candidate generation.** 给定 "Jordan"，哪些 KB entries 可能？
2. **Disambiguation.** 给定上下文，哪个 candidate 是正确的？

两步都可以学习，也都有 benchmark。组合 pipeline 十年来一直稳定，变化的是 disambiguator 质量。

## The Concept / 概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation / 候选生成。** 给定 mention surface form（"Jordan"），在 alias index 中查候选。Wikipedia alias dictionaries 覆盖多数 named entities：`"JFK"` → John F. Kennedy、Jacqueline Kennedy、JFK airport、JFK (movie)。典型 index 每个 mention 返回 10-30 个 candidates。

**Disambiguation / 消歧：三种方法。**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、不需要训练。
2. **Embedding-based (ESS / REL / Blink).** 编码 mention + context。编码每个 candidate description。取最大 cosine。2020-2024 默认方案。
3. **Generative (GENRE, 2021; LLM-based, 2023+).** 逐 token 解码 entity canonical name。约束到 valid entity names 的 trie，保证输出一定是 valid KB id。

**End-to-end vs pipeline / 端到端 vs pipeline。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）会一次性运行 NER + candidate generation + disambiguation。Pipeline systems 在生产中仍占主导，因为你可以替换组件。

### The two measurements / 两个指标

- **Mention recall (candidate gen).** Gold mentions 中，正确 KB entry 出现在 candidate list 里的比例。这是整个 pipeline 的下限。
- **Disambiguation accuracy / F1.** 给定正确 candidates，top-1 有多常正确。

始终报告两者。一个 candidate recall 只有 80%、disambiguation 有 99% 的系统，整体 pipeline 也只有 80%。

## Build It / 动手构建

### Step 1: build an alias index from Wikipedia redirects / 第 1 步：从 Wikipedia redirects 构建 alias index

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data 大约有 18M 对 (alias, entity)。从 Wikidata dumps 下载，并存成 inverted index。

### Step 2: context-based disambiguation / 第 2 步：基于上下文消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap 是 toy。实际用 embeddings 上的 cosine similarity 替换（见 `code/main.py` step-2 的 transformer 版本）。

### Step 3: embedding-based (BLINK-style) / 第 3 步：embedding-based（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Index time 时，把每个 KB entity embedding 一次。Query time 时，把 mention + context embedding 一次，然后与 candidate pool 做 dot-product，取最大。

### Step 4: generative entity linking (concept) / 第 4 步：generative entity linking（概念）

GENRE 会逐字符解码 entity 的 Wikipedia title。Constrained decoding（见 lesson 20）保证只能输出 valid titles。它与 KB-backed trie 紧密集成。现代后继是 REL-GEN，以及带 structured output 的 LLM-prompted EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合 whitelist（Outlines `choice`），这是 2026 年最简单可交付的 EL pipeline。

### Step 5: evaluate on AIDA-CoNLL / 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL benchmark：1,393 篇 Reuters 文章，34k mentions，Wikipedia entities。报告 in-KB accuracy（`P@1`）和 out-of-KB NIL-detection rate。

## Pitfalls / 常见坑

- **NIL handling.** 有些 mentions 不在 KB 中（新兴实体、冷门人物）。系统必须预测 NIL，而不是猜一个错误 entity。单独测量。
- **Mention boundary errors.** 上游 NER 漏掉部分 spans（"Bank of America" 只标成 "Bank"）。EL recall 会下降。
- **Popularity bias.** 训练系统会过度预测高频 entities。ML 论文中的 "Michael I. Jordan" 经常被链接到篮球 Jordan。
- **Cross-lingual EL.** 把中文文本中的 mentions 映射到 English Wikipedia entities。需要 multilingual encoder 或 translation step。
- **KB staleness.** 新公司、事件、人物不在去年的 Wikipedia dump 中。生产 pipeline 需要 refresh loop。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| Cross-lingual，KB = Wikipedia | mGENRE |
| LLM-friendly，每天少量 mentions | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| 领域 KB（医疗、法律） | Custom BERT with KB-aware retrieval + 在领域 AIDA-style set 上 fine-tune |
| 极低延迟 | Exact-match prior only（Milne-Witten baseline） |
| Research SOTA | GENRE / ExtEnD / generative LLM-EL |

2026 年生产模式：NER → coref → 对每个 mention 做 EL → 把 clusters 折叠成每个 cluster 一个 canonical entity。输出是文档中每个 entity 一个 KB id，而不是每个 mention 一个。

## Ship It / 交付它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Exercises / 练习

1. **Easy / 简单。** 在 10 个 ambiguous mentions（Paris、Jordan、Apple）上实现 `code/main.py` 中的 prior+context disambiguator。手工标注正确 entity，测量 accuracy。
2. **Medium / 中等。** 用 sentence transformer 编码 50 个 ambiguous mentions，并编码每个 candidate 的 description。比较 embedding-based disambiguation 与 Jaccard context overlap。
3. **Hard / 困难。** 构建一个 1k-entity domain KB（例如你公司的员工 + 产品）。实现 NER + EL end-to-end。在 100 个 held-out sentences 上测量 precision 和 recall。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 链接到 Wikipedia | 把 mention 映射到唯一 KB entry。 |
| Candidate generation | 可能是谁？ | 为 mention 返回 plausible KB entries 的 shortlist。 |
| Disambiguation | 选正确的那个 | 使用上下文给 candidates 打分，并选择赢家。 |
| Alias index | 查找表 | 从 surface form 映射到 candidate entities。 |
| NIL | 不在 KB 中 | 明确预测没有 KB entry 匹配。 |
| KB | Knowledge base | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | Benchmark | 带 gold entity links 的 1,393 篇 Reuters 文章。 |

## Further Reading / 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — foundational prior+context approach。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — embedding-based workhorse。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — constrained decoding 的 generative EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — benchmark 论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产 stack。
