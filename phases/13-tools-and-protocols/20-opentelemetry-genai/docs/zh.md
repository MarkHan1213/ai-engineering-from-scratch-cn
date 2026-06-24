# OpenTelemetry GenAI — Tracing Tool Calls End-to-End / OpenTelemetry GenAI：端到端追踪工具调用

> 一个 agent 调用五个工具、三个 MCP servers、两个 sub-agents。你需要一条 trace 覆盖全部。OpenTelemetry GenAI semantic conventions（v1.37 及以上的 stable attributes）是 2026 年标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 都原生支持。本课会命名 required attributes，走过 span hierarchy（agent → LLM → tool），并交付一个可插到任意 OTel exporter 的 stdlib span emitter。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, OTel span emitter)
**Prerequisites / 前置知识：** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出 LLM span 和 tool-execution span 所需的 OTel GenAI attributes。
- 构建覆盖 agent loop、LLM call、tool call 和 MCP client dispatch 的 trace hierarchy。
- 决定哪些 content 要 capture（opt-in），哪些要 redact（defaults）。
- 在不重写 tool code 的情况下向 local collector（Jaeger、Langfuse）发出 spans。

## The Problem / 问题

2026 年 2 月的一次 debug：用户反馈“我的 agent 有时 30 秒才响应，有时 3 秒”。没有 traces。日志显示 LLM call，但没有 tool dispatch，没有 MCP server round-trip，也没有 sub-agent。你只能猜。最后发现：某个 MCP server 偶尔卡在 cold-start。

没有 end-to-end tracing，你找不到这个问题。OTel GenAI 解决它。

conventions 在 2025-2026 年由 OpenTelemetry semantic-conventions group 敲定。它们定义 stable attribute names，让 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析同一批 spans。instrument 一次，发往任意 backend。

## The Concept / 概念

### Span hierarchy / Span 层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整棵树位于同一个 trace id 下。Span ids 连接 parent-child relationships。

### Required attributes / 必需属性

按 2025-2026 semconv：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — requested model string（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际 served 的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — provider response id，用于 correlation。

tool spans：

- `gen_ai.tool.name` — tool identifier。
- `gen_ai.tool.call.id` — 具体 call id。
- `gen_ai.tool.description` — tool description（optional）。

agent spans：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span kinds / Span kind

- `SpanKind.CLIENT` 用于跨越 process boundary 的调用（LLM provider、MCP server）。
- `SpanKind.INTERNAL` 用于 agent 自己的 loop steps 和 tool execution。

### Opt-in content capture / Opt-in 内容采集

默认情况下，spans 携带 metrics 和 timing，不携带 prompts 或 completions。大 payload 和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及特定 content-capture env vars 后才包含 content。生产启用前要认真审查。

### Events on spans / Span 上的 events

token-level events 可以作为 span events 添加：

- `gen_ai.content.prompt` — input messages。
- `gen_ai.content.completion` — output messages。
- `gen_ai.content.tool_call` — 记录下来的 tool call。

Events 在 span 内按时间排序，可用于 detailed replay。

### Exporters / Exporters

OTel spans 可以导出到：

- **Jaeger / Tempo.** OSS，on-prem。
- **Langfuse.** 面向 LLM observability；可视化 token usage。
- **Arize Phoenix.** Evals + tracing combined。
- **Datadog.** 商业；原生解析 `gen_ai.*` attributes。
- **Honeycomb.** Column-oriented，query-friendly。

它们都说 OTLP，也就是 wire format。你的代码不关心具体后端。

### Propagation across MCP / 跨 MCP 传播

MCP client 调用 server 时，把 W3C traceparent header 注入 request。Streamable HTTP 支持标准 headers。Stdio 不原生携带 HTTP headers；spec 的 2026 roadmap 正在讨论在 JSON-RPC calls 上加入 `_meta.traceparent` field。

在它落地前：手动把 traceparent 放进每个 request 的 `_meta`。server 记录 trace id。

### Metrics / 指标

GenAI semconv 除 spans 外还定义 metrics：

- `gen_ai.client.token.usage` — histogram。
- `gen_ai.client.operation.duration` — histogram。
- `gen_ai.tool.execution.duration` — histogram。

这些适合不需要 per-call detail 的 dashboards。

### AgentOps layer / AgentOps 层

AgentOps（成立于 2024）专注 GenAI observability。它包装 LangGraph、Pydantic AI、CrewAI 等流行框架，自动发出 OTel spans。如果你的 stack 使用支持的 framework，它很有用；否则使用 manual instrumentation。

## Build It / 动手构建

本课会用 stdlib 生成 OTLP-JSON-like spans：一个 top-level agent span、LLM spans、tool execution spans 和 MCP dispatch span。你会显式设置 trace id、parentSpanId、SpanKind 和 `gen_ai.*` attributes，从而理解 exporter 之外的核心数据模型。

## Use It / 应用它

`code/main.py` 对一个会调用 LLM、dispatch 两个 tools、并做一次 MCP round-trip 的 agent，向 stdout 发出 OTel-shaped spans（OTLP-JSON-like format）。没有真实 exporter；本课聚焦 span shape 和 attribute set。把输出贴到 OTLP-compatible viewer，或直接阅读。

重点看：

- Trace id 在所有 spans 中共享。
- Parent-child links 通过 `parentSpanId` 编码。
- Required `gen_ai.*` attributes 已填充。
- content capture 默认关闭；一个场景会通过 env var 打开它。

## Ship It / 交付它

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个 agent codebase，这个 skill 会生成 instrumentation plan：在哪里加 spans、填哪些 attributes、面向哪些 exporters。

## Exercises / 练习

1. 运行 `code/main.py`。数清 spans，并识别哪些是 CLIENT，哪些是 INTERNAL。

2. 打开 content capture（env var），确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` events。注意 PII 影响。

3. 添加 tool-execution metric `gen_ai.tool.execution.duration`，每次 call 发出一个 histogram sample。

4. 把 traceparent 从 parent agent span 传播到 MCP request 的 `_meta.traceparent` field。验证 MCP server 会看到相同 trace id。

5. 阅读 OTel GenAI semconv spec。找出一个 semconv 中列出但本课代码未发出的 attribute。把它加上。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| OTel | “OpenTelemetry” | traces、metrics、logs 的开放标准 |
| GenAI semconv | “GenAI semantic conventions” | LLM / tool / agent spans 的 stable attribute names |
| `gen_ai.*` | “attribute namespace” | 所有 GenAI attributes 共享这个 prefix |
| Span | “Timed operation” | 带 start、end 和 attributes 的 work unit |
| Trace | “Cross-span ancestry” | 共享 trace id 的 spans 树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | span direction hints |
| OTLP | “OpenTelemetry Line Protocol” | exporters 使用的 wire format |
| Opt-in content | “Prompt / completion capture” | 默认关闭；通过 env var 启用 |
| traceparent | “W3C header” | 跨服务传播 trace context |
| Exporter | “Backend-specific shipper” | 把 spans 发送到 Jaeger / Datadog 等后端的组件 |

## Further Reading / 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI spans、metrics 和 events 的 canonical conventions
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和 tool-execution span attribute list
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent-level `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub-hosted source of truth
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — production integration walkthrough
