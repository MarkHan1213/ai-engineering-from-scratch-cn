# Relation Extraction & Knowledge Graph Construction / 关系抽取与知识图谱构建

> NER 找到了实体。Entity linking 把它们锚定。Relation extraction 找出它们之间的边。Knowledge graph 是 nodes、edges 和 provenance 的总和。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 把自由文本中的事实表示为 `(subject, relation, object)` triples
- 实现 pattern-based extraction、LLM anchoring prompt、relation canonicalization 与小型 graph query
- 理解 rule/pattern、supervised classifier、generative LLM 与 AEVS pipeline 的取舍
- 识别 provenance、entity/relation canonicalization、temporal qualifiers 与 hallucinated triples 的生产风险

## The Problem / 问题

分析师读到："Tim Cook became CEO of Apple in 2011." 这里有四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation Extraction（RE）把自由文本转成结构化 triples `(subject, relation, object)`。跨 corpus 聚合后，就有了 knowledge graph。再聚合、查询，就得到了 RAG、analytics 或 compliance audits 的 reasoning substrate。

2026 年的问题是：LLMs 抽取关系很积极，过于积极。它们会 hallucinate 源文本不支持的 triples。没有 provenance，你无法区分真实 triples 和看起来合理的虚构内容。2026 年的答案是 AEVS-style anchor-and-verify pipelines。

## The Concept / 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Triple form / Triple 形式。** `(subject_entity, relation_type, object_entity)`。Relations 可以来自 closed ontology（Wikidata properties、FIBO、UMLS），也可以是 open set（OpenIE 风格，什么都可以）。

**Three extraction approaches / 三种抽取方法。**

1. **Rule / pattern-based.** Hearst patterns："X such as Y" → `(Y, isA, X)`。再加手写 regex。脆弱、精确、可解释。
2. **Supervised classifier.** 给定句子中的两个 entity mentions，从固定集合预测 relation。在 TACRED、ACE、KBP 上训练。2015–2022 标准方案。
3. **Generative LLM.** Prompt 模型输出 triples。开箱有效。需要 provenance，否则会 hallucinate 看起来合理的垃圾。

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** 当前 hallucination-mitigation framework：

- **Anchor.** 用精确位置识别每个 entity span 和 relation-phrase span。
- **Extract.** 生成与 anchor spans 绑定的 triples。
- **Verify.** 把每个 triple element 匹配回 source text；拒绝任何 unsupported 内容。
- **Supplement.** Coverage pass 确保 anchored span 没被漏掉。

Hallucinations 会显著下降。需要更多计算，但可审计。

**The open-vs-closed tradeoff / 开放与封闭的取舍。**

- **Closed ontology.** 固定 property list（例如 Wikidata 11,000+ properties）。可预测、可查询、难以发明。
- **Open IE.** 任意 verbal phrase 都能成为 relation。高 recall，低 precision，查询混乱。

生产 KG 通常混合使用：用 open IE 做发现，然后把 relations canonicalize 到 closed ontology，再合并进主图。

## Build It / 动手构建

### Step 1: pattern-based extraction / 第 1 步：pattern-based extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整 toy extractor 见 `code/main.py`。Hearst patterns 今天仍然会出现在领域专用 pipelines 中，因为它们可调试。

### Step 2: supervised relation classification / 第 2 步：supervised relation classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是 seq2seq relation extractor：文本输入，triples 输出，并且已经使用 Wikidata property ids。它在 distant-supervision data 上 fine-tuned，是标准 open-weights baseline。

### Step 3: LLM-prompted extraction with anchoring / 第 3 步：带 anchoring 的 LLM 抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回 span 是否对应 source。只要 `text[start:end] != triple_entity`，就拒绝。这是 AEVS "verify" step 的最小形式。

### Step 4: canonicalize onto a closed ontology / 第 4 步：canonicalize 到 closed ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Canonicalization 经常占 60-80% 工程工作。要为它预留预算。

### Step 5: build a small graph and query / 第 5 步：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子。要扩展，可以使用 RDF triple stores（Blazegraph、Virtuoso）、property graphs（Neo4j）或 vector-augmented graph stores。

## Pitfalls / 常见坑

- **Coreference before RE.** "He founded Apple"——RE 需要知道 "he" 是谁。先跑 coref（lesson 24）。
- **Entity canonicalization.** "Apple Inc" 和 "Apple" 必须解析到同一个 node。先做 entity linking（lesson 25）。
- **Hallucinated triples.** LLMs 会输出文本不支持的 triples。强制 span verification。
- **Relation canonicalization drift.** Open IE relations 不一致（"was born in," "came from," "is a native of"）。不折叠到 canonical ids，图就无法查询。
- **Temporal errors.** "Tim Cook is CEO of Apple" 现在为真，2005 年为假。许多 relations 有时间边界。使用 qualifiers（Wikidata 中 `P580` start time、`P582` end time）。
- **Domain mismatch.** REBEL 在 Wikipedia 上训练。法律、医疗、科学文本通常需要 domain-fine-tuned RE models。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred with Wikidata canonicalization |
| 领域专用（biomed, legal） | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted，可审计输出 | AEVS pipeline：anchor → extract → verify → supplement |
| 高吞吐新闻 IE | Pattern-based + supervised hybrid |
| 从零构建 KG | Open IE + manual canonicalization pass |
| Temporal KG | 带 qualifiers 抽取（start/end time, point in time） |

集成模式：NER → coref → entity linking → relation extraction → ontology mapping → graph load。每个阶段都可能成为质量闸门。

## Ship It / 交付它

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Exercises / 练习

1. **Easy / 简单。** 在 5 条新闻句子上运行 `code/main.py` 中的 pattern extractor。手工检查 precision。
2. **Medium / 中等。** 在同样句子上使用 REBEL（或小 LLM）。比较 triples。哪个 extractor precision 更高？哪个 recall 更高？
3. **Hard / 困难。** 构建 AEVS pipeline：用 LLM 抽取 + 按 source 验证 spans。在 50 个 Wikipedia 风格句子上测量 verify step 前后的 hallucination rate。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | `(s, r, o)` tuple，KG 的原子单位。 |
| Open IE | 什么都抽 | Open-vocabulary relation phrases；高 recall，低 precision。 |
| Closed ontology | 固定 schema | 有边界的 relation types 集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 全部归一化 | 把 surface names / relations 映射到 canonical ids。 |
| AEVS | Grounded extraction | Anchor-Extraction-Verification-Supplement pipeline（2026）。 |
| Provenance | Source-of-truth link | 每个 triple 都带 doc id + 来源 char-span。 |
| Distant supervision | 廉价标签 | 把文本与已有 KG 对齐，生成训练数据。 |

## Further Reading / 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — distant-supervision 论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries。
