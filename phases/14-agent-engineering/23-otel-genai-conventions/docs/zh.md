# OpenTelemetry GenAI Semantic Conventions / OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 Agent telemetry 的标准 schema。Span names、attributes 和 content-capture rules 正在跨 vendors 收敛，让 agent traces 在 Datadog、Grafana、Jaeger 和 Honeycomb 中表达同一件事。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 13（LangGraph）, 第 14 阶段 · 24（Observability Platforms）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 GenAI span categories：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT 与 INTERNAL spans，以及它们各自何时适用。
- 列出顶层 GenAI attributes：provider name、request model、data-source ID。
- 解释 content-capture contract：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、external-reference recommendation。

## The Problem / 问题

每个 vendor 都发明自己的 span names。Ops 团队最终要给每个 framework 单独做 dashboard。OpenTelemetry 的 GenAI SIG 通过定义一个生态共同面向的标准来解决这个问题。

## The Concept / 概念

### Span categories / Span 分类

1. **Model / client spans.** 覆盖原始 LLM calls。由 provider SDKs（Anthropic、OpenAI、Bedrock）和 framework model adapters 发出。
2. **Agent spans.** `create_agent`（构造 agent 时）和 `invoke_agent`（运行 agent 时）。
3. **Tool spans.** 每次 tool invocation 一个；通过 parent-child relation 连接到 agent span。

### Agent span naming / Agent span 命名

- Span name：如果有名字，使用 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span kind：
  - **CLIENT** — 用于 remote agent services（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于 in-process agent frameworks（LangChain、CrewAI、本地 ReAct）。

### Key attributes / 关键属性

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`。
- `gen_ai.request.model` — model ID。
- `gen_ai.response.model` — 实际解析出的 model（可能因 routing 与 request 不同）。
- `gen_ai.agent.name` — agent identifier。
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个 corpus 或 store。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有技术栈特定的 conventions。

### Content capture / 内容捕获

默认规则是：instrumentations SHOULD NOT 默认捕获 inputs/outputs。捕获必须通过以下字段 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：把 content 存在外部（S3、你的 log store），只在 spans 上记录引用（pointer IDs，而不是 prose）。这就是 Lesson 27 的 content-poisoning 防御接入 observability 的方式。

### Stability / 稳定性

截至 2026 年 3 月，大多数 conventions 仍是 experimental。用下面的配置 opt in 到 stable preview：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 会把 GenAI attributes 原生映射到它的 LLM Observability schema。其他 backends（Grafana、Honeycomb、Jaeger）支持原始 attributes。

### Where this pattern goes wrong / 这种模式容易出错的地方

- **Capturing full prompts in spans.** PII、secrets、customer data 进入 ops 可读取的 traces。应存外部引用。
- **No `gen_ai.provider.name`.** attribution 缺失时，multi-provider dashboards 会失效。
- **Spans without parent links.** tool spans 变成孤儿。始终传播 context。
- **Not setting stability opt-in.** backend upgrade 时，你的 attributes 可能被重命名。

## Build It / 动手构建

`code/main.py` 实现了一个符合 GenAI conventions 的 stdlib span emitter：

- 带 GenAI attribute schema 的 `Span`。
- 带 `start_span` 和 nested contexts 的 `Tracer`。
- 一个 scripted agent run，会发出：`create_agent`、`invoke_agent`（INTERNAL）、per-tool spans，以及 LLM calls 的 `chat` spans。
- 一种 content-capture mode：把 prompts 存到外部，并在 spans 上记录 IDs。

运行：

```
python3 code/main.py
```

输出：一棵包含所有必需 GenAI attributes 的 span tree，以及展示 opt-in content references 的 “external store”。

## Use It / 应用它

- **Datadog LLM Observability**（v1.37+）会原生映射 attributes。
- **Langfuse / Phoenix / Opik**（Lesson 24）— 自动 instrument 生态。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel traces；基于 GenAI attributes 构建 dashboards。
- **Self-hosted** — 运行带 GenAI processor 的 OTel Collector。

## Ship It / 交付它

`outputs/skill-otel-genai.md` 会把 OTel GenAI spans 接入现有 agent，并设置 content-capture defaults 和 external-reference storage。

## Exercises / 练习

1. 用 `invoke_agent`（INTERNAL）+ per-tool spans instrument 你的 Lesson 01 ReAct loop，并发送到一个 Jaeger instance。
2. 增加 “references only” 模式的 content capture：prompts 写入 SQLite，span attributes 只携带 row IDs。
3. 阅读 `gen_ai.data_source.id` 的 spec。把它接到 Lesson 09 Mem0 search。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证 collector 不会重命名你的 attributes。
5. 只用 GenAI attributes 构建一个 dashboard：“哪些 tool errors 与哪些 models 相关”。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | “OpenTelemetry GenAI group” | 定义 schema 的 OTel working group |
| invoke_agent | “Agent span” | 表示一次 agent run 的 span 名称 |
| CLIENT span | “Remote call” | 调用 remote agent service 的 span |
| INTERNAL span | “In-process” | in-process agent run 的 span |
| gen_ai.provider.name | “Provider” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG source” | 某个 retrieval hit 来自哪个 corpus/store |
| Content capture | “Prompt logging” | opt-in 捕获 messages；生产环境外部存储 |
| Stability opt-in | “Preview mode” | 用于固定 experimental conventions 的 env var |

## Further Reading / 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans built in
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
