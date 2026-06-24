# Structured Outputs & Constrained Decoding / 结构化输出与受约束解码

> 让 LLM 返回 JSON。多数时候你会得到 JSON。在生产里，“多数”就是问题。Constrained decoding 在采样前修改 logits，把“多数”变成“永远”。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 prompting、native structured output APIs 与 constrained decoding
- 理解 logit processor 如何按 JSON Schema、regex 或 CFG 屏蔽非法 tokens
- 使用 Outlines、Instructor 和 vendor structured output API 生成结构化结果
- 设计 schema field order、nullable fields、enum/regex 与 validation plan，避免语义正确性问题

## The Problem / 问题

一个分类器 prompt LLM："Return one of {positive, negative, neutral}." 模型返回 "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的 parser 崩溃。分类器 F1 变成 0.0。

自由生成不是契约，只是建议。生产系统需要契约。

2026 年有三层方案。

1. **Prompting.** 好好请求。"Return only the JSON object." 对 frontier models 大约 80% 有效，小模型更差。
2. **Native structured output APIs.** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。对支持的 schemas 可靠。Vendor-locked。
3. **Constrained decoding.** 在每个 generation step 修改 logits，让模型 *无法* 发出 invalid tokens。结构上 100% valid。适用于任意 local model。

这一课会建立三者直觉，并说明什么时候该用哪一个。

## The Concept / 概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**How constrained decoding works / Constrained decoding 如何工作。** 每个生成步骤，LLM 会在完整 vocabulary（约 100k tokens）上产生一个 logit vector。一个 *logit processor* 位于模型和 sampler 之间。它根据 target grammar 中当前位置计算哪些 tokens 合法——JSON Schema、regex、context-free grammar——并把所有非法 tokens 的 logits 设成 negative infinity。剩余 logits 上的 softmax 只会把概率质量分配给合法续写。

2026 年实现：

- **Outlines.** 把 JSON Schema 或 regex 编译成 finite-state machine。每个 token 都有 O(1) valid-next-token lookup。基于 FSM，所以递归 schema 需要 flattening。
- **XGrammar / llguidance.** Context-free grammar engines。处理递归 JSON Schema。几乎零 decoding overhead。OpenAI 在 2025 structured output implementation 中 credit 了 llguidance。
- **vLLM guided decoding.** 通过 Outlines、XGrammar 或 lm-format-enforcer backends 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor.** Pydantic-based wrapper over any LLM。Validation failure 后重试。跨 provider，但不修改 logits，而是依赖 retries + structured-output-aware prompts。

### The counterintuitive result / 反直觉结果

Constrained decoding 经常比 unconstrained generation 更快。两个原因：第一，它缩小了 next-token search space。第二，聪明实现会对 forced tokens 直接跳过 token generation（例如 `{"name": "` 这类脚手架，每个 byte 都已确定）。

### The pitfall that costs you / 真正会让你付出代价的坑

Field order 很重要。把 `answer` 放在 `reasoning` 前面，模型会在思考前先提交答案。JSON 是合法的，但答案是错的。Validation 抓不到它。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema field order 是逻辑，不是格式。

## Build It / 动手构建

### Step 1: regex-constrained generation from scratch / 第 1 步：从零实现 regex-constrained generation

查看 `code/main.py` 获取独立 FSM 实现。核心思想 30 行：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 会跟踪 grammar 已经满足到哪一步。`valid_tokens(state, tokenizer)` 计算哪些 vocabulary tokens 可以推进 FSM，并且不会离开 accepting path。

### Step 2: Outlines for JSON Schema / 第 2 步：用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零 validation errors，永远如此。FSM 让 invalid output 无法到达。

### Step 3: Instructor for provider-agnostic Pydantic / 第 3 步：用 Instructor 做 provider-agnostic Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不触碰 logits。它把 schema 格式化进 prompt，解析输出，并在 validation failure 后重试（默认 3 次）。适用于任意 provider。重试会增加延迟和成本。跨 provider portability 是它的卖点。

### Step 4: native vendor APIs / 第 4 步：原生 vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Server-side constrained decoding。对支持的 schemas，可靠性与 Outlines 持平。不需要管理 local model，但会锁定 vendor。

## Pitfalls / 常见坑

- **Recursive schemas.** Outlines 会把递归 flatten 到固定深度。Tree-structured outputs（nested comments、AST）需要 XGrammar 或 llguidance（CFG-based）。
- **Huge enums.** 10,000-option enum 编译很慢或超时。换成 retriever：先预测 top-k candidates，再约束到这些候选。
- **Grammar too strict.** 强制 `date: "YYYY-MM-DD"` regex，模型就无法在缺失日期时输出 `"unknown"`。模型会补偿性地发明一个日期。允许 `null` 或 sentinel。
- **Premature commitment.** 见上面的 field-order pitfall。始终把 reasoning 放前面。
- **Vendor JSON mode without schema.** 纯 JSON mode 只保证 JSON 合法，不保证对你的用例合法。始终提供完整 schema。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| OpenAI/Anthropic/Google model，简单 schema | Native vendor structured output |
| 任意 provider，Pydantic workflow，可接受 retries | Instructor |
| Local model，需要 100% validity，flat schema | Outlines（FSM） |
| Local model，recursive schema | XGrammar 或 llguidance |
| Self-hosted inference server | vLLM guided decoding |
| Batch processing，可接受 retries | Instructor + cheapest model |

## Ship It / 交付它

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Exercises / 练习

1. **Easy / 简单。** 不使用 constrained decoding，prompt 一个小型 open-weights model（例如 Llama-3.2-3B），要求输出 `Review(sentiment, confidence, evidence_span)`。在 100 条 reviews 上测量能解析为 valid JSON 的比例。
2. **Medium / 中等。** 在同一 corpus 上使用 Outlines JSON mode。比较 compliance rate、latency 和 semantic accuracy。
3. **Hard / 困难。** 为 phone numbers（`\d{3}-\d{3}-\d{4}`）从零实现 regex-constrained decoder。验证 1000 个 samples 中 0 invalid outputs。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Constrained decoding | 强制合法输出 | 在每个 generation step 屏蔽 invalid-token logits。 |
| Logit processor | 约束它的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | Finite-state machine | 编译后的 grammar 表示；O(1) valid-next-token lookup。 |
| CFG | Context-free grammar | 能处理递归的 grammar；比 FSM 慢但表达力更强。 |
| Schema field order | 重要吗？ | 重要。第一个 field 会提交立场；始终把 reasoning 放在 answer 前。 |
| Guided decoding | vLLM 对它的叫法 | 同一个概念，集成进 inference server。 |
| JSON mode | OpenAI 的早期版本 | 保证 JSON syntax；不保证匹配 schema。 |

## Further Reading / 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速 CFG-based constrained decoding。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — inference server 集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas。
- [Instructor library](https://python.useinstructor.com/) — 跨 providers 的 Pydantic + retries。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6 个 constrained decoding frameworks 的 benchmark。
