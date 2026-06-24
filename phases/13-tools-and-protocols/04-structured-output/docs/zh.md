# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding / 结构化输出：JSON Schema、Pydantic、Zod 与受约束解码

> “好好提示模型返回 JSON”即使在 frontier model 上也会有 5% 到 15% 的失败率。Structured outputs 通过 constrained decoding 补上这个缺口：模型被字面意义上阻止输出会违反 schema 的 token。OpenAI strict mode、Anthropic schema-typed tool use、Gemini `responseSchema`、Pydantic AI 的 `output_type`、Zod 的 `.parse`，都是同一个想法的五种表面形态。本课会构建 schema validator 和 strict-mode contract，后续每条生产 extraction pipeline 都会用到它们。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites / 前置知识：** Phase 13 · 02 (function calling deep dive)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 使用正确约束（enum、min/max、required、pattern）为 extraction target 编写 JSON Schema 2020-12。
- 解释 strict mode 和 constrained decoding 为什么比“生成后校验”提供不同保证。
- 区分三种 failure mode：parse error、schema violation、model refusal。
- 交付一个带 typed repair 和 typed refusal handling 的 extraction pipeline。

## The Problem / 问题

一个 agent 读取采购订单邮件时，需要把自由文本变成 `{customer, line_items, total_usd}`。有三种 approach。

**Approach one: prompt for JSON.** “Reply in JSON with fields customer, line_items, total_usd.” 在 frontier models 上大约 85% 到 95% 成功。失败方式有六种：缺 brace、尾随逗号、类型错误、幻觉字段、在 token limit 截断、泄漏 prose，例如 "Here is your JSON:"。

**Approach two: validate after generation.** 自由生成、parse、按 schema validate，失败就 retry。可靠但昂贵：每次 retry 都要付费，而且截断 bug 每出现一次就多花一轮。

**Approach three: constrained decoding.** provider 在 decode time 强制 schema。无效 token 会从 sampling distribution 中被 mask 掉。输出被保证可 parse，也被保证可 validate。失败模式被压缩为一种：refusal（模型判断输入不适合该 schema）。

到 2026 年，每个 frontier provider 都提供了 approach three 的某种形式。

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝，则 response 中带 `refusal`。
- **Anthropic.** 对 `tool_use` inputs 做 schema enforcement；没有 `stop_reason: "refusal"`，但 `end_turn` 且没有 tool call 就是信号。
- **Gemini.** request-level 的 `responseSchema`；2026 年 Gemini 对选定类型提供 token-level grammar constraints。
- **Pydantic AI.** `output_type=InvoiceModel` 产出 typed 到 `InvoiceModel` 的 structured `RunResult`。
- **Zod (TypeScript).** runtime parser，用 Zod schema 校验 provider output；通常配合 OpenAI 的 `beta.chat.completions.parse`。

共同点是：只声明一次 schema，然后端到端强制它。

## The Concept / 概念

### JSON Schema 2020-12 — the lingua franca / JSON Schema 2020-12：通用语言

每家 provider 都接受 JSON Schema 2020-12。最常用的构造：

- `type`: one of `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: field name 到 subschema 的映射。
- `required`: 必须出现的 field name 列表。
- `enum`: 允许值的 closed set。
- `minimum` / `maximum`（numbers），`minLength` / `maxLength` / `pattern`（strings）。
- `items`: 应用于每个 array element 的 subschema。
- `additionalProperties`: `false` 禁止额外字段（默认随 mode 而变）。

OpenAI strict mode 额外增加三条要求：每个 property 必须列在 `required` 中，所有位置都要有 `additionalProperties: false`，且不能有未解析 `$ref`。违反这些要求时，API 会在 request time 返回 400。

### Pydantic, the Python binding / Pydantic：Python 绑定

Pydantic v2 会通过 `model_json_schema()` 从 dataclass-shaped models 生成 JSON Schema。Pydantic AI 把这层包起来，让你可以写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent framework 会在边界把 schema 翻译成 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会作为 typed `Invoice` instance 返回。Validation errors 会抛出带 typed error paths 的 `ValidationError`。

### Zod, the TypeScript binding / Zod：TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，把它翻译成 API 的 JSON Schema payload。

### Refusals / 拒绝

Strict mode 不能强迫模型回答。如果输入不适合 schema（例如“邮件是一首诗，不是 invoice”），模型会发出包含原因的 `refusal` field。你的代码必须把它当作一等 outcome，而不是 failure。refusal 也能作为 safety signal：如果要求模型从 protected-content email 中抽取 credit card number，它会带着 safety reason 返回 refusal。

### Constrained decoding in the open / 开源权重中的受约束解码

Open-weights 实现通常使用三种技术。

1. **Grammar-based decoding**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建 deterministic finite automaton；每一步都 mask 掉会违反 FSM 的 token logits。
2. **Logit masking with a JSON parser**：让 streaming JSON parser 和模型同步运行；每一步计算 valid-next-token set。
3. **Speculative decoding with a verifier**：廉价 draft model 提议 tokens，verifier 强制 schema。

商业 provider 会在幕后选择其中一种。2026 年的 state of the art：短结构化输出比 plain generation 更快，长输出大致同速。

### The three failure modes / 三种失败模式

1. **Parse error.** 输出不是合法 JSON。strict mode 下不可能发生。non-strict provider 仍可能发生。
2. **Schema violation.** 输出能 parse，但违反 schema。strict mode 下不可能发生。其他模式下常见。
3. **Refusal.** 模型拒绝。必须作为 typed outcome 处理。

### Retry strategy / 重试策略

当你不在 strict mode 中（Anthropic tool use、non-strict OpenAI、older Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次 retry 足够。三次 retry 能兜住弱模型偶发错误。超过三次说明 schema 可能不好：模型无法为某些输入满足它，prompt 或 schema 需要修。

### Small-model support / 小模型支持

Constrained decoding 对小模型同样有效。带 grammar enforcement 的 3B-parameter open model，在结构化任务上会胜过 raw prompting 的 70B-parameter model。这是 structured outputs 对生产很重要的核心原因：它把可靠性和模型规模解耦了。

## Build It / 动手构建

本课会先定义一个 strict-mode-compatible 的 Invoice schema，再实现一个 stdlib validator，覆盖类型、required fields、enum、范围、pattern、array items 和 `additionalProperties`。随后把 fake LLM output 送入 validator，分别走 parse error、schema violation 和 refusal 分支。

## Use It / 应用它

`code/main.py` 提供了一个只用 stdlib 写成的最小 JSON Schema 2020-12 validator（types、required、enum、min/max、pattern、items、additionalProperties）。它包住一个 `Invoice` schema，并把 fake LLM output 跑过 validator，展示 parse error、schema violation 和 refusal paths。生产中可以把 fake output 换成任意 provider 的真实 response。

重点看：

- validator 返回 typed `[ValidationError]` list，包含 path 和 message。这正是你想暴露给 retry prompt 的形状。
- refusal branch 不会 retry。它记录日志并返回 typed refusal。Phase 14 · 09 会把 refusals 用作 safety signal。
- `additionalProperties: false` check 会在 adversarial test input 上触发，展示 strict mode 为什么能堵住 hallucinated fields。

## Ship It / 交付它

本课产出 `outputs/skill-structured-output-designer.md`。给定一个 free-text extraction target（invoices、support tickets、resumes 等），这个 skill 会生成 strict-mode-compatible 的 JSON Schema 2020-12，以及与其镜像的 Pydantic model，并预置 typed refusal 和 retry handling stub。

## Exercises / 练习

1. 运行 `code/main.py`。添加第四个 test case，让 `total_usd` 是负数。确认 validator 会用 `minimum` constraint path 拒绝它。

2. 扩展 validator，支持带 discriminator 的 `oneOf`。常见场景：`line_item` 要么是 product，要么是 service，由 `kind` 标记。Strict mode 在这里有微妙规则；查阅 OpenAI structured outputs guide。

3. 用 Pydantic BaseModel 写同一个 Invoice schema，并把 `model_json_schema()` 输出与你手写的 schema 对比。找出一个 Pydantic 默认设置、但手写版本遗漏的字段。

4. 测量 refusal rates。构造十个不应被抽取的输入（一段 song lyric、一个 math proof、一封 blank email），用 strict mode 跑真实 provider。统计 refusals vs hallucinated outputs。这就是 refusal-aware retries 的 ground truth。

5. 从头到尾阅读 OpenAI structured outputs guide。找出一个 strict mode 明确禁止、但 plain JSON Schema 允许的构造。然后设计一个非必要使用该 forbidden construct 的 schema，并把它重构为 strict-compatible。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “schema spec” | 每个现代 provider 都支持的 IETF-draft schema dialect |
| Strict mode | “保证 schema” | OpenAI flag，通过 constrained decoding 强制 schema |
| Constrained decoding | “Logit masking” | decode-time enforcement，会 mask 掉无效 next-token |
| Refusal | “模型拒绝” | 输入无法适配 schema 时的 typed outcome |
| Parse error | “Invalid JSON” | 输出无法 parse 为 JSON；strict 下不可能发生 |
| Schema violation | “形状错误” | 已 parse，但违反 types / required / enum / range |
| `additionalProperties: false` | “不允许额外字段” | 禁止 unknown fields；OpenAI strict 中必需 |
| Pydantic BaseModel | “Typed output” | 生成并校验 JSON Schema 的 Python class |
| Zod schema | “TypeScript output type” | 用于 provider output validation 的 TS runtime schema |
| Grammar enforcement | “Open-weights constrained decode” | 基于 FSM 的 logit masking，例如 outlines / guidance |

## Further Reading / 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、refusals 和 schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月 launch post，解释 decoding guarantee
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 会序列化到各 provider 的 typed output_type bindings
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — canonical spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — enterprise deployment notes 和 strict-mode caveats
