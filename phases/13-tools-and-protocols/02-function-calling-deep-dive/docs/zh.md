# Function Calling Deep Dive — OpenAI, Anthropic, Gemini / Function Calling 深入解析：OpenAI、Anthropic、Gemini

> 三家 frontier provider 在 2024 年收敛到同一个 tool-call loop，随后又在其他所有细节上分叉。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` blocks。Gemini 使用 `functionDeclarations` 和 unique-id correlation。本课会把三者并排 diff，确保你在一个 provider 上写出的代码，迁移时不会在另一个 provider 上断掉。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, schema translators)
**Prerequisites / 前置知识：** Phase 13 · 01 (the tool interface)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出 OpenAI、Anthropic 和 Gemini function-calling payload 的三类 shape difference（declaration、call、result）。
- 把一个 tool declaration 翻译到三家 provider format，并预测 strict-mode constraints 会在哪些地方不同。
- 在每家 provider 中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解各 provider 的 hard limits（tool count、schema depth、argument length）以及违反限制时的 error signature。

## The Problem / 问题

function-calling request 的形状随 provider 而变。2026 年生产栈中有三个具体例子：

**OpenAI Chat Completions / Responses API.** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须由你解析的 JSON string。Strict mode（`strict: true`）通过 constrained decoding 强制 schema compliance。

**Anthropic Messages API.** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 返回。`input` 已经是解析后的对象，不是字符串。你需要用新的 `user` message 回复，其中包含 `{type: "tool_result", tool_use_id, content}` block。

**Google Gemini API.** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`，嵌在 `functionDeclarations` 下。响应出现在 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 中；Gemini 3 及以上版本的 `id` 对 parallel-call correlation 是唯一的。你用 `{functionResponse: {name, id, response}}` 回复。

同一个循环，不同字段名、不同嵌套、不同 string-vs-object 约定、不同 correlation mechanism。团队如果在 OpenAI 上写了一个 weather agent，迁到 Anthropic 往往要花两天处理 plumbing，再迁到 Gemini 又要花一天。

本课会构建一个 translator：把三种格式统一成一个 canonical tool declaration，并在边界做 route。Phase 13 · 17 会把同一个模式泛化为 LLM gateway。

## The Concept / 概念

### The common structure / 共同结构

每家 provider 都需要五件事：

1. **Tool list.** 每个工具的 name、description 和 input schema。
2. **Tool choice.** 强制指定工具、禁止工具，或让模型自行决定。
3. **Call emission.** 结构化输出，说明工具名和参数。
4. **Call id.** 把结果关联到正确调用；并行时尤其重要。
5. **Result injection.** 一个 message 或 block，把结果绑定回对应调用。

### Shape diffs, field by field / 逐字段形状差异

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | `tool_calls[]` on assistant message | `content[]` of type `tool_use` | `parts[]` of type `functionCall` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...` (OpenAI generates) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Result block | role `tool`, `tool_call_id` | `user` with `tool_result`, `tool_use_id` | `functionResponse` with matching `id` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema (always enforced) | `responseSchema` at request level |

### Limits you will actually hit / 真实会踩到的限制

- **OpenAI.** 每个 request 最多 128 个 tools。Schema depth 5。Argument string <= 8192 bytes。Strict mode 要求没有 `$ref`，没有带 overlap 的 `oneOf`/`anyOf`/`allOf`，每个 property 都列在 `required` 中。
- **Anthropic.** 每个 request 最多 64 个 tools。Schema depth 理论上几乎不受限，但 practical limit 大约 10。没有 strict-mode flag；schema 是契约，模型通常会遵守。
- **Gemini.** 每个 request 最多 64 个 functions。Schema types 是 OpenAPI 3.0 subset（和 JSON Schema 2020-12 有轻微差异）。Gemini 3 开始为 parallel calls 提供 unique-id。

### `tool_choice` behavior / `tool_choice` 行为

所有 provider 都支持三种模式，只是命名不同。

- **Auto.** 模型选择工具或文本。默认模式。
- **Required / Any.** 模型必须至少调用一个工具。
- **None.** 模型不能调用工具。

此外每家还有自己的特殊模式：

- **OpenAI.** 按 name 强制指定一个工具。
- **Anthropic.** 按 name 强制指定一个工具；`disable_parallel_tool_use` flag 单独控制 single vs multi。
- **Gemini.** `mode: "VALIDATED"` 会不论模型意图如何，都把每个响应送过 schema validator。

### Parallel calls / 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）会在一个 assistant message 中发出多个调用。你运行全部调用，然后用一个 batched tool-role message 回复，每个 entry 对应一个 `tool_call_id`。Anthropic 历史上是 single-call；`disable_parallel_tool_use: false`（Claude 3.5 之后默认）启用 multi。Gemini 2 支持 parallel calls，但没有 stable ids；Gemini 3 加入 UUID，使乱序响应可以干净关联。

### Streaming / 流式输出

三家都支持 streamed tool calls，但 wire format 不同：

- **OpenAI.** `tool_calls[i].function.arguments` 的 delta chunks 逐步到达。你持续累积，直到 `finish_reason: "tool_calls"`。
- **Anthropic.** Block-start / block-delta / block-stop events。`input_json_delta` chunks 携带 partial arguments。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3 新增）会带着 `functionCallId` 发出 chunk，因此多个 parallel calls 可以交错。

Phase 13 · 03 会深入 parallel + streaming reassembly。本课聚焦 declaration 和 single-call shapes。

### Errors and repair / 错误与修复

invalid-argument error 的形状也各不相同。

- **OpenAI (non-strict).** 模型返回 `arguments: "{bad json}"`，你的 JSON parse 失败，再注入 error message 并重新调用。
- **OpenAI (strict).** Validation 在 decoding 阶段发生；invalid JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic.** `input` 可能包含 unexpected fields；schema 更像 advisory。需要 server-side validate。
- **Gemini.** OpenAPI 3.0 quirk：object field 上的 `enum` 可能被静默忽略；自己校验。

### The translator pattern / Translator 模式

你代码里的 canonical tool declaration 可以长这样（形状由你定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它翻译成三家 provider shape。`code/main.py` 中的 harness 正是这样做的，然后把 fake tool call 往返转换到每家 provider 的 response shape。无需网络：本课教的是形状，不是 HTTP。

生产团队会把这个 translator 包进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。Phase 13 · 17 会交付一个 gateway，在任意三家 provider 前面暴露 OpenAI-shaped API。

## Build It / 动手构建

本课的构建重点是 canonical shape。你会先定义一个 provider-neutral 的 `Tool`，再写三个 translator，把同一份 schema 投影到 OpenAI、Anthropic 和 Gemini。随后再写 parser，把三家响应都还原成统一的 `{id, name, args}` 调用对象。

## Use It / 应用它

`code/main.py` 定义一个 canonical `Tool` dataclass，以及三个 translator，用来输出 OpenAI、Anthropic 和 Gemini 的 declaration JSON。它还把每种 provider shape 的手写响应解析成同一个 canonical call object，证明语义在表面差异之下是相同的。运行它，并把三份 declaration 并排 diff。

重点看：

- 三个 declaration block 只在 envelope 和 field name 上不同。
- 三个 response block 的差异在于 call 位于哪里：top-level `tool_calls`、`content[]` block，或 `parts[]` entry。
- 一个 `canonical_call()` function 能从三种 response shape 中抽取 `{id, name, args}`。

## Ship It / 交付它

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个绑定某家 provider 的 function-calling integration，这个 skill 会生成 portability audit：它依赖了哪些 provider limits，哪些字段需要重命名，迁移到其他 provider 时会断在哪里。

## Exercises / 练习

1. 运行 `code/main.py`，确认三家 provider declaration JSON 都序列化自同一个底层 `Tool` object。修改 canonical tool，加入一个 enum parameter，确认只有 Gemini translator 需要处理 OpenAPI quirk。

2. 为每家 provider 添加一个 `ListToolsResponse` parser，从模型在 `list_tools` 或 discovery call 后返回的结果中抽取 tool list。OpenAI 原生没有这个接口；记录这个 asymmetry。

3. 实现 `tool_choice` conversion：把 canonical `ToolChoice(mode="force", tool_name="x")` 映射到三家 provider shape。然后映射 `mode="any"` 和 `mode="none"`。对照本课 diff table。

4. 从三家 provider 中任选一家，完整阅读它的 function-calling guide。找出一个它的 schema spec 支持、另外两家不支持的字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个 test vector：一个参数违反已声明 schema 的 tool call。把它送进每家 provider 的 validator（Lesson 01 的 stdlib validator 可作为 proxy），记录触发哪些错误。说明如果你要在生产里追求 strictness，会选择哪家 provider。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Function calling | “工具使用” | provider-level API，用于发出结构化 tool-call |
| Tool declaration | “工具规格” | Name + description + JSON Schema input payload |
| `tool_choice` | “强制 / 禁止” | Auto / required / none / specific-name modes |
| Strict mode | “Schema enforcement” | OpenAI flag，通过 constrained decoding 强制匹配 schema |
| `tool_use` block | “Anthropic 的调用形状” | 带 id、name、input 的 inline content block |
| `functionCall` part | “Gemini 的调用形状” | 包含 name、args 和 id 的 `parts[]` entry |
| Arguments-as-string | “Stringified JSON” | OpenAI 把 args 返回为 JSON string，而不是 object |
| Parallel tool calls | “一轮 fan-out” | 一个 assistant message 中的多个 tool call |
| Refusal | “模型拒绝” | strict-mode-only refusal block，而不是 call |
| OpenAPI 3.0 subset | “Gemini schema quirk” | Gemini 使用类似 JSON Schema 的 dialect，但有少量差异 |

## Further Reading / 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — canonical reference，包括 strict mode 和 parallel calls
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` block semantics
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — parallel calls、unique ids 和 OpenAPI subset
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的 enterprise surface
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema enforcement details
