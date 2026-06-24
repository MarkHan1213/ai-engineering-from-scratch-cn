# The Tool Interface — Why Agents Need Structured I/O / 工具接口：为什么 Agent 需要结构化 I/O

> 语言模型产生 token，程序执行动作。两者之间的缝隙，就是工具接口：一份契约，让模型可以请求一个动作，让 host 负责真正执行。2026 年的每一套技术栈，无论是 OpenAI、Anthropic、Gemini 的 function calling，MCP 的 `tools/call`，还是 A2A 的 task parts，本质上都在编码同一个四步循环。本课会给这个循环命名，并展示运行它所需的最小机制。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, no LLM)
**Prerequisites / 前置知识：** Phase 11 (LLM completion APIs)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释为什么只能生成文本的 LLM 不能独立对真实世界采取行动。
- 画出四步工具调用循环（describe → decide → execute → observe），并说清每一步由谁负责。
- 把一个工具描述成三部分：name、JSON Schema input，以及确定性的 executor function。
- 区分 pure tool 和 side-effecting tool，并说明这种区分为什么对安全很重要。

## The Problem / 问题

LLM 的输出表面，本质上只是下一个 token 的概率分布。如果你问一个聊天模型“Bengaluru 现在天气怎样”，它可以写出一句看起来可信的话，但它不能自己拨到天气 API。那句话可能刚好正确，也可能是三天前的旧信息。

工具接口的目的就是补上这个缺口。host program，也就是你的 agent runtime、Claude Desktop、ChatGPT、Cursor，或一段自定义脚本，会向模型公布一组可调用工具。模型判断需要行动时，会产出一个结构化 payload，说明要调用哪个工具以及参数是什么。host 解析这个 payload，真正运行工具，再把结果喂回模型。这个循环会一直继续，直到模型判断不再需要调用工具。

这份契约的第一个版本，是 OpenAI 在 2023 年 6 月通过 `"functions"` parameter 推出的。Anthropic 随后在 Claude 2.1 中加入 `tool_use` block；Gemini 几个月后加入 `functionDeclarations`。现在每家 provider 都暴露了同一种形状：请求里传入一组 JSON-Schema-typed tool list，响应里返回一个 JSON-payload tool call。Model Context Protocol（2024 年 11 月）把这份契约泛化成模型无关的工具 registry。A2A（2026 年 4 月，v1.0）又把同一个原语用于 agent-to-agent delegation。

所有这些表面之下，不变的是四步循环。Phase 13 后面的内容，都是对它的展开。

## The Concept / 概念

### Step one: describe / 第一步：描述

host 用三个字段声明每个工具。

- **Name.** 稳定、机器可读的标识符。应该是 `get_weather`，不是 "weather thing"。
- **Description.** 一段自然语言简介。"Use when the user asks about current conditions for a specific city. Do not use for historical data."
- **Input schema.** 一个 JSON Schema object（draft 2020-12），描述工具参数。

模型会收到这份列表。现代 provider 会用各自的模板把这些声明序列化进 system prompt，所以调用方通常只需要处理结构化形式。

### Step two: decide / 第二步：决策

给定用户消息和可用工具，模型会选择三种行为之一。

1. **直接用文本回答。** 不产生工具调用。
2. **调用一个或多个工具。** 产出结构化 call object。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认开启，Anthropic 需要 opt-in），模型可以在一轮里发出多个调用。
3. **拒绝。** strict-mode structured outputs 可以产出 typed `refusal` block，而不是工具调用。

一个 tool call payload 有三个稳定字段：call `id`、tool `name`，以及 JSON `arguments` object。id 用来让 host 把之后返回的结果和具体调用对应起来；并行调用乱序返回时，这一点尤其关键。

### Step three: execute / 第三步：执行

host 收到调用后，会按已声明 schema 校验参数，然后运行 executor。参数无效意味着模型幻觉了字段或使用了错误类型，这是弱模型上非常常见的失败模式。生产 host 面对无效参数通常有三种做法：快速失败并把错误暴露给模型、用受约束 parser 修复 JSON，或把 validation error 放回 prompt 后重试模型。

executor 本身就是普通代码：Python、TypeScript、shell command、database query 都可以。它产出一个 result，通常是字符串，但也可以是任意 JSON value 或结构化 content block（MCP 中的 text、image、resource reference）。结果必须可序列化。

### Step four: observe / 第四步：观察

host 把工具结果追加到对话中（作为带 matching `id` 的 `tool` role message），然后再次调用模型。模型现在能在上下文里看到工具输出，于是可以生成最终答案，或请求更多调用。这个过程持续到模型停止发出调用，或者 host 触达迭代次数的安全上限。

### The trust split / 信任边界拆分

从安全角度看，工具分为两类。

- **Pure.** 只读、确定性、没有副作用。比如 `get_weather`、`search_docs`、`get_current_time`。可以安全地 speculative call。
- **Consequential.** 会修改状态、花钱或触碰用户数据。比如 `send_email`、`delete_file`、`execute_trade`。必须加 gate。

Meta 2026 年面向 agent security 的 "Rule of Two" 说，单轮中最多只能同时包含以下三项中的两项：untrusted input、sensitive data、consequential action。工具接口正是执行这条规则的位置：拒绝调用、要求用户确认，或升级 scope。完整安全章节见 Phase 13 · 15，agent-level permission policies 见 Phase 14 · 09。

### Where the loop lives / 循环位于哪里

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

无论在哪个场景，都是同样的四步。列名会变，结构不会变。

### Why not just prompt the model to emit JSON? / 为什么不直接提示模型输出 JSON？

"Ask the model to reply in JSON" 是 function calling 之前的常见模式。它在 frontier model 上仍会有大约 5% 到 15% 的失败率，在小模型上更高。失败模式包括缺右括号、尾随逗号、幻觉字段、类型错误。之后你还需要 JSON repair pass、retry，或 constrained decoder。

原生 function calling 更好，原因有三点。第一，provider 会围绕精确 call shape 对模型做端到端训练，因此 strict mode 下 valid-JSON rate 可以提升到 98% 到 99%。第二，call payload 位于独立的 protocol slot 中，而不是夹在 free-text 里，所以工具调用不会泄漏到用户可见回复。第三，provider 会用 constrained decoding 强制 schema compliance（OpenAI strict mode、Anthropic `tool_use`、Gemini `responseSchema`）。输出会被保证通过校验。

Phase 13 · 02 会并排讲解三家 provider API。Phase 13 · 04 会深入 structured outputs。

### Circuit breakers / 断路器

当模型停止发出调用，或 host 达到最大 turn count 时，循环终止。生产 host 通常把上限设在 5 到 20 轮之间。超过这个范围，几乎可以确定你进入了模型无法自行退出的循环。Claude Code 默认 20，OpenAI Assistants 默认 10，Cursor 的 agent mode 默认 25。

另一种选择是不设上限，这几乎每六个月都会以“agent 半夜花掉 400 美元 API 调用费”的事故复盘形式出现。不要在没有上限的情况下上线。

Phase 14 · 12 会深入错误恢复和 self-healing；Phase 17 会讲生产 rate limits。

### Where Phase 13 goes from here / Phase 13 后续路线

- Lessons 02 through 05 会打磨 provider-level tool-call surface。
- Lessons 06 through 14 会把循环泛化到 MCP。
- Lessons 15 through 18 会防御 hostile servers、adversarial users 和 unauthenticated remote auth surfaces。
- Lessons 19 through 22 会把模式扩展到 agent-to-agent collaboration、observability、routing 和 packaging。
- Lesson 23 会用每个原语交付一个完整 ecosystem。

后面的每一课，都是对这个四步循环的展开。把它当成不变量记住。

## Build It / 动手构建

本课的构建目标不是调用真实 LLM，而是把四步循环拆到可观察的最小 harness 中：registry 负责描述，fake decider 模拟模型决策，validator 与 executor 负责执行前校验和动作执行，observe step 再把结果带回下一轮。这样你能先理解协议形状，再把 fake decider 替换成真实 provider。

## Use It / 应用它

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的 "decider" function 通过 pattern-matching 用户消息来模拟模型；executor、schema validator 和 observe-step harness 都是真的。运行它，观察完整 request/response choreography 和可打印的中间状态，然后在后续课程里把 fake decider 换成任意真实 provider。

重点看：

- tool registry 为每个工具保存三个字段：name、description、schema，以及 executor reference。
- validator 是一个只用 stdlib 写成的最小 JSON Schema subset（types、required、enum、min/max）。Phase 13 · 04 会提供更完整版本。
- 循环把迭代次数限制在五次。生产 agent 正需要这种 circuit breaker。

## Ship It / 交付它

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一份 draft tool definition（name + description + schema + executor outline），这个 skill 会审计它是否适合放进循环：name 是否机器稳定、description 是否是一份完整的 usage brief、schema 是否正确使用 JSON Schema 2020-12，以及 pure-vs-consequential 分类是否明确。

## Exercises / 练习

1. 在 `code/main.py` 中添加第四个工具 `get_stock_price(ticker)`。它的 description 写成 "Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries." 运行 harness，确认 fake decider 会把提到 ticker 的查询路由到新工具。

2. 故意破坏 schema validator。传入一个 `arguments` object 缺少 required field 的调用，确认 host 会在执行前拒绝它。然后传入一个带额外未知字段的调用。做一个决策：host 应该 reject 还是 ignore？用安全论证说明理由。

3. 将 harness 中每个工具分类为 pure 或 consequential。给需要的 registry entry 增加 `consequential: true` flag，并修改循环：只要选中 consequential tool，就打印一行 "would confirm with user"。这就是每个生产 host 都需要的 confirmation gate 形状。

4. 在纸上画出四步循环，并把上面的 provider-column table 填成你最常用的 client（Claude Desktop、Cursor、ChatGPT 或 custom stack）。再和 Phase 13 · 06 中 MCP-specific variant 对照。

5. 从头到尾阅读 OpenAI 的 function-calling guide。找出一个位于 request 中、但不属于本课四步循环的字段。解释它增加了什么，以及为什么它是方便项而不是必要项。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Tool | “模型可以调用的东西” | name + JSON-Schema-typed input + executor function 组成的三元组 |
| Function calling | “原生工具使用” | provider-level API 支持：让模型输出结构化 tool call，而不是 prose |
| Tool call | “模型请求采取行动” | 模型发出的 JSON payload，包含 `id`、`name`、`arguments` |
| Tool result | “工具返回的内容” | executor 输出，被包装成带 matching id 的 `tool` role message |
| Parallel tool calls | “一次多个调用” | 一个 model turn 中的多个 call object，彼此独立并可按 id 排序 |
| Strict mode | “保证 JSON” | constrained decoding，强制模型输出符合已声明 schema |
| Pure tool | “只读工具” | 没有副作用；可以安全重复运行 |
| Consequential tool | “动作工具” | 会修改外部状态；需要 gate、audit 或用户确认 |
| Four-step loop | “工具调用周期” | describe → decide → execute → observe |
| Host | “Agent runtime” | 持有 tool registry、调用模型并运行 executor 的程序 |

## Further Reading / 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形状的 canonical reference
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` block format
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 与 parallel-call semantics
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 工具接口的 provider-agnostic generalization
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 现代工具 API 共同使用的 schema dialect
