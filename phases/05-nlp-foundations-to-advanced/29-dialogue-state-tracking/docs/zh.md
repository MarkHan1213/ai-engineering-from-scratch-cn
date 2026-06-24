# Dialogue State Tracking / 对话状态追踪

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian." 三轮对话，三次状态更新。DST 让 slot-value dict 保持同步，预订才不会出错。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 用 schema、domains、slots 和 values 表示 task-oriented dialogue state
- 实现 rule-based slot extractor、state update loop、LLM + Pydantic DST 与 JGA evaluation
- 处理 correction、negation、implicit inheritance、closed/open slots 等状态更新问题
- 判断 rule-based、LDST、LLM-with-Pydantic 与混合 confirmation flow 的适用场景

## The Problem / 问题

在 task-oriented dialogue system 中，用户目标被编码成一组 slot-value pairs：`{cuisine: italian, area: north, price: moderate}`。每个 user turn 都可能新增、修改或移除 slot。系统必须读取完整对话，并正确输出当前 state。

一个 slot 错了，系统就会订错餐厅、排错航班，或扣错卡。DST 是“用户说了什么”和“后端执行什么”之间的铰链。

为什么 2026 年有 LLM 之后它仍然重要：

- 合规敏感领域（银行、医疗、航班预订）需要确定性 slot values，而不是自由生成。
- Tool-use agents 在调用 APIs 前仍然需要 slot resolution。
- 多轮修正比看起来更难："actually no, make it Thursday."

现代 pipeline：经典 DST 概念 + LLM extractors + structured-output guardrails。

## The Concept / 概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**Task structure / 任务结构。** Schema 定义 domains（restaurant、hotel、taxi）及其 slots（cuisine、area、price、people）。每个 slot 可以为空，可以从 closed set 中填值（price: {cheap, moderate, expensive}），也可以是 free-form value（name: "The Copper Kettle"）。

**Two DST formulations / 两种 DST 表述。**

- **Classification.** 对每个 (slot, candidate_value) pair 预测 yes/no。适合 closed-vocab slots。2020 年前标准。
- **Generation.** 给定 dialogue，生成 free text slot values。适合 open-vocab slots。现代默认。

**Metric / 指标。** Joint Goal Accuracy（JGA）——每个 turn 中 *所有* slots 都正确的比例。All-or-nothing。2026 年 MultiWOZ 2.4 leaderboard 约 83% 封顶。

**Architectures / 架构。**

1. **Rule-based (slot regex + keyword).** 窄域强 baseline。可调试。
2. **TripPy / BERT-DST.** 基于 BERT encoding 的 copy-based generation。LLM 前标准。
3. **LDST (LLaMA + LoRA).** 带 domain-slot prompting 的 instruction-tuned LLM。在 MultiWOZ 2.4 上达到 ChatGPT-level quality。
4. **Ontology-free (2024–26).** 跳过 schema，直接生成 slot names 和 values。处理开放领域。
5. **Prompt + structured output (2024–26).** LLM + Pydantic schema + constrained decoding。5 行代码，生产可用。

### The classic failure modes / 经典失败模式

- **Co-reference across turns.** "Let's stay with the first option." 需要解析是哪一个 option。
- **Over-write vs append.** 用户说 "add Italian." 你是替换 cuisine，还是追加？
- **Implicit confirmations.** "OK cool"——这是否接受了提供的 booking？
- **Correction.** "Actually make it 7 pm." 必须更新时间，而不清空其他 slots。
- **Coreference to previous system utterance.** "Yes, that one." 哪个 "that"？

## Build It / 动手构建

### Step 1: rule-based slot extractor / 第 1 步：rule-based slot extractor

见 `code/main.py`。Regex + synonym dictionaries 可以覆盖窄域 canonical utterances 的 70%：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

离开 canonical vocabulary 后会很脆弱。适合确定性 slot confirmations。

### Step 2: state update loop / 第 2 步：state update loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要 reset 用户没有触碰的 slot。
- 显式否定（"never mind the cuisine"）必须清空。
- 用户修正（"actually..."）必须 overwrite，而不是 append。

### Step 3: LLM-driven DST with structured output / 第 3 步：用 structured output 驱动 LLM-DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证得到 valid state object。没有 regex、没有 schema mismatch、没有 hallucinated slots。

### Step 4: JGA evaluation / 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少比例的 turns 上把所有 slots 都做对？MultiWOZ 2.4 上，2026 年顶级系统约 80-83%。你的 in-domain system 在窄 vocabulary 上应该超过它，否则 LLM baseline 已经打赢你了。

### Step 5: handling correction / 第 5 步：处理 correction

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到 correction 后，overwrite last-updated slot，而不是 append。如果没有 LLM 帮助，这很难做对。现代模式：始终让 LLM 从 history regenerate whole state，而不是增量更新；这样天然处理 corrections。

## Pitfalls / 常见坑

- **Full-history regeneration cost.** 让 LLM 每轮重新生成 state，总 token 成本是 O(n²)。要限制 history 或总结旧 turns。
- **Schema drift.** 事后新增 slots 会破坏旧训练数据。Version your schema。
- **Case sensitivity.** "Italian" vs "italian" vs "ITALIAN"——到处 normalize。
- **Implicit inheritance.** 如果用户之前指定 "for 4 people"，新的不同时间请求不应该清空 people。始终传完整 history。
- **Free-form vs closed-set.** Names、times、addresses 需要 free-form slots；cuisines 和 areas 是 closed。Schema 中要混用两者。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Approach / 方法 |
|-----------|----------|
| 窄域（一两个 intents） | Rule-based + regex |
| 广域，有标注数据 | LDST（LLaMA + LoRA on MultiWOZ-style data） |
| 广域，无标签，生产可用 | LLM + Instructor + Pydantic schema |
| 语音 / voice | ASR + normalizer + LLM-DST |
| 多领域 booking flow | Schema-guided LLM with per-domain Pydantic models |
| 合规敏感 | Rule-based primary, LLM fallback with confirmation flow |

## Ship It / 交付它

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Exercises / 练习

1. **Easy / 简单。** 为 3 个 slots（cuisine、area、price）构建 `code/main.py` 中的 rule-based state tracker。在 10 个手写 dialogues 上测试。测量 JGA。
2. **Medium / 中等。** 用 Instructor + Pydantic + 小 LLM 在同一 dataset 上做 DST。比较 JGA。检查最难的 turns。
3. **Hard / 困难。** 同时实现两者并路由：rule-based primary，当 rule-based 输出 <2 个 slots 且 confidence 低时使用 LLM fallback。测量 combined JGA 和每轮 inference cost。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | 跨对话 turns 维护 slot-value dict。 |
| Slot | 用户意图单元 | 后端需要的命名参数（cuisine, date）。 |
| Domain | 任务领域 | Restaurant、hotel、taxi——一组 slots。 |
| JGA | Joint Goal Accuracy | 每个 slot 都正确的 turns 比例。All-or-nothing。 |
| MultiWOZ | Benchmark | Multi-domain WOZ dataset；标准 DST evaluation。 |
| Ontology-free DST | 无 schema | 直接生成 slot names 和 values，不使用固定列表。 |
| Correction | "Actually..." | 覆盖之前已填 slot 的 turn。 |

## Further Reading / 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — canonical benchmark。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — copy-based DST workhorse。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results。
