# MCP Fundamentals — Primitives, Lifecycle, JSON-RPC Base / MCP 基础：原语、生命周期与 JSON-RPC 基底

> MCP 之前的每个集成几乎都是一次性的。Model Context Protocol 由 Anthropic 在 2024 年 11 月首次发布，现由 Linux Foundation 的 Agentic AI Foundation 负责治理，标准化了 discovery 和 invocation，让任意 client 都能与任意 server 对话。2025-11-25 spec 定义了六个 primitives（三个 server primitive、三个 client primitive）、三阶段 lifecycle，以及 JSON-RPC 2.0 wire format。掌握这些，Phase 13 后续 MCP 章节只是细节展开。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, JSON-RPC parser)
**Prerequisites / 前置知识：** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 说出全部六个 MCP primitives（server 侧的 tools、resources、prompts；client 侧的 roots、sampling、elicitation），并各举一个用例。
- 走通三阶段 lifecycle（initialize、operation、shutdown），说明每个阶段由谁发送哪条 message。
- 解析并发出 JSON-RPC 2.0 request、response 和 notification envelopes。
- 解释 `initialize` 中的 capability negotiation 是什么，以及没有它会坏在哪里。

## The Problem / 问题

MCP 之前，每个 tool-using agent 都有自己的协议。Cursor 有一套形似 MCP 但不兼容的工具系统。Claude Desktop 带着另一套。VS Code 的 Copilot extension 又有第三套。一个团队如果构建了 "Postgres query" 工具，需要为三个 host API 写三遍。同一能力想复用，就只能复制代码。

结果就是一次性集成的寒武纪爆发，同时压低了 ecosystem velocity 的上限。

MCP 用标准 wire format 解决这个问题。一个 MCP server 可以在所有 MCP client 中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ clients。SDK 月下载量 110M。公开 servers 10,000+。2025 年 12 月，Linux Foundation 在新的 Agentic AI Foundation 下接手 stewardship。

本 phase 使用的 spec revision 是 **2025-11-25**。它加入 async Tasks（SEP-1686）、URL-mode elicitation（SEP-1036）、sampling with tools（SEP-1577）、incremental scope consent（SEP-835）和 OAuth 2.1 resource-indicator semantics。Phase 13 · 09 到 16 会覆盖这些扩展。本课只讲 base。

## The Concept / 概念

### Three server primitives / 三个 server 原语

1. **Tools.** 可调用动作。就是 Phase 13 · 01 的四步循环。
2. **Resources.** 暴露的数据。可通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、custom schemes。
3. **Prompts.** 可复用模板。host UI 里的 slash-commands；server 提供模板，client 填参数。

### Three client primitives / 三个 client 原语

4. **Roots.** server 被允许访问的 URI 集合。client 声明它们，server 必须尊重。
5. **Sampling.** server 请求 client 侧模型执行一次生成。让 server-hosted agent loop 不需要 server-side API keys。
6. **Elicitation.** server 在执行过程中向 client 的用户请求结构化输入。可以是 forms，也可以是 URLs（SEP-1036）。

MCP 中的每项能力都精确归属于这六个之一。Phase 13 · 10 到 14 会逐个深入。

### Wire format: JSON-RPC 2.0 / Wire format：JSON-RPC 2.0

每条 message 都是一个 JSON object，包含这些字段：

- Requests: `{jsonrpc: "2.0", id, method, params}`。
- Responses: `{jsonrpc: "2.0", id, result | error}`。
- Notifications: `{jsonrpc: "2.0", method, params}` — 没有 `id`，也不期望 response。

基础规范定义了约 15 个 methods，按 primitive 分组。重要的有：

- `initialize` / `initialized`（handshake）
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage`（server-to-client）
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### Three-phase lifecycle / 三阶段生命周期

**Phase 1: initialize.**

Client 发送 `initialize`，带上它的 `capabilities` 和 `clientInfo`。Server 返回自己的 `capabilities`、`serverInfo` 以及它支持的 spec version。Client 消化响应后发送 `notifications/initialized`。从这里开始，双方都可以按 negotiated capabilities 发送 requests。

**Phase 2: operation.**

双向通信。Client 调用 `tools/list` 做 discovery，再调用 `tools/call` 做 invocation。如果 server 声明了能力，可以发送 `sampling/createMessage`。当 tool set 变化时，server 可以发送 `notifications/tools/list_changed`。当用户修改 root scope 时，client 可以发送 `notifications/roots/list_changed`。

**Phase 3: shutdown.**

任意一方关闭 transport。MCP 没有结构化 shutdown method；transport（stdio 或 Streamable HTTP，见 Phase 13 · 09）承载 end-of-connection signal。

### Capability negotiation / 能力协商

`initialize` handshake 中的 `capabilities` 就是契约。一个 server 示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

server 声明它可以发出 `tools/list_changed` notifications，并支持 `resources/subscribe`。client 通过声明自己的能力来配合：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果 client 没声明 `sampling`，server 就不能调用 `sampling/createMessage`。反过来也一样：如果 server 没声明 `resources.subscribe`，client 就不能尝试订阅。

这就是阻止 ecosystem drift 的机制。不支持 sampling 的 client 仍然是合法 MCP client；不调用 `sampling` 的 server 也仍然是合法 MCP server。它们只是不会一起使用这个 feature。

### Structured content and error shapes / 结构化内容与错误形状

`tools/call` 返回 typed blocks 组成的 `content` array：`text`、`image`、`resource`。Phase 13 · 14 会把 MCP Apps（`ui://` interactive UI）加入这个列表。

错误使用 JSON-RPC error codes。spec-defined additions 包括：`-32002` "Resource not found"、`-32603` "Internal error"，以及放在 `error.data` 中的 MCP-specific error data。

### Client capabilities vs tool call details / Client capabilities 与工具调用细节

一个常见混淆：`capabilities.tools` 表示 client 是否支持 tool-list-changed notifications。client 是否会调用某个具体工具，是由模型在运行时做出的选择，不是 capability flag。capability flag 是 spec-level contract。模型选择与它正交。

### Why JSON-RPC and not REST? / 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是轻量的双向协议。REST 是 client-initiated。MCP 需要 server-initiated messages（sampling、notifications），所以带 symmetric request/response shape 的 JSON-RPC 很合适。JSON-RPC 也能自然组合到 stdio 和 WebSocket/Streamable HTTP 上，不需要重新发明 HTTP 的 request shape。

```figure
mcp-tool-call
```

## Build It / 动手构建

本课会先用 stdlib 写一个 JSON-RPC parser/emitter，再手动走完 `initialize`、`tools/list`、`tools/call` 和 shutdown。目标不是做完整 server，而是把 envelope、id matching、notification 无响应、capability negotiation 这些基础不变量看清楚。

## Use It / 应用它

`code/main.py` 提供一个最小 JSON-RPC 2.0 parser 和 emitter，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` sequence，并打印每条 message。没有真实 transport，只有 message shapes。把它和 Further Reading 中链接的 spec 对比，验证每个 envelope。

重点看：

- `initialize` 双向声明 capabilities；response 有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` array；每个 entry 有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- response `content` 是 `{type, text}` blocks 的 array。

## Ship It / 交付它

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一份 pcap-style 的 MCP client-server interaction transcript，这个 skill 会为每条 message 标注它属于哪个 primitive、哪个 lifecycle phase，以及依赖哪个 capability。

## Exercises / 练习

1. 运行 `code/main.py`。找出发生 capability negotiation 的那一行，并描述如果 server 没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展 parser，支持 `notifications/progress`。message shape 是 `{method: "notifications/progress", params: {progressToken, progress, total}}`。在 long-running `tools/call` 进行中发出它，确认 client handler 会显示 progress bar。

3. 从头到尾阅读 MCP 2025-11-25 spec，整份文档约 80 页。找出大多数 server 并不需要的一个 capability flag。提示：它和 resource subscription 有关。

4. 在纸上草拟一个假想 "cron job" feature 应该属于哪个 primitive。（提示：server 希望 client 在预定时间调用它。今天六个 primitive 都不完全适合。）MCP 的 2026 roadmap 已有相关 draft SEP。

5. 解析一个 GitHub 上 open MCP server 的 session log。统计 request、response、notification messages 数量。计算 lifecycle vs operation traffic 的比例。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MCP | “Model Context Protocol” | 用于 model-to-tool discovery 与 invocation 的开放协议 |
| Server primitive | “server 暴露什么” | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive | “client 允许 server 使用什么” | roots（scope）、sampling（LLM callbacks）、elicitation（用户输入） |
| JSON-RPC 2.0 | “wire format” | 对称的 request/response/notification envelopes |
| `initialize` handshake | “Capability negotiation” | 第一组 message；server 和 client 声明支持的 features |
| `tools/list` | “Discovery” | client 向 server 查询当前 tool set |
| `tools/call` | “Invocation” | client 请求 server 用 arguments 执行工具 |
| `notifications/*_changed` | “Mutation events” | server 告诉 client 某个 primitive list 已改变 |
| Content block | “Typed result” | tool result 中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名的 draft proposal，例如 SEP-1686 for async Tasks |

## Further Reading / 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — canonical spec document
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六 primitive mental model
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月 launch post
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 spec changes
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835、1724 的摘要
