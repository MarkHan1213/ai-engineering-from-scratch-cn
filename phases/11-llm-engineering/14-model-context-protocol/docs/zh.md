# Model Context Protocol (MCP) / 模型上下文协议

> 2025 年以前，每个 LLM 应用都在发明自己的工具 schema。后来 Anthropic 发布 MCP，Claude 接入了它，OpenAI 也接入了它。到 2026 年，它已经成为把任意 LLM 连接到工具、数据源或 Agent 的默认 wire format。写一个 MCP server，所有 host 都能和它对话。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 理解 MCP 如何把 host、client、server 之间的工具协议统一到 JSON-RPC 2.0
- 区分 MCP 的三类 primitive：`tools`、`resources` 与 `prompts`
- 用 Python `FastMCP` 构建可发现、可调用的最小 MCP server
- 识别 MCP 上线时的安全边界：路径 allowlist、人类审批、tool poisoning 防护与版本漂移

## The Problem / 问题

你发布了一个 chatbot，它需要三个工具：数据库查询、calendar API 和文件读取器。你先为 Claude 写了三份 JSON schema。然后销售团队希望同一套工具也能在 ChatGPT 里用，于是你又为 OpenAI 的 `tools` 参数重写一次。接着你接入 Cursor、Zed 和 Claude Code，又多了三套写法，每套 JSON 约定都微妙不同。一周后 Anthropic 新增一个字段，你要更新六份 schema。

这就是 2025 年以前的现实。每个 host（运行 LLM 的应用）和每个 server（暴露工具与数据的系统）都带着自己的私有协议。规模化意味着 N×M 的集成矩阵。

Model Context Protocol 把这个矩阵压扁：一个基于 JSON-RPC 的规范；一个 server 暴露 tools、resources 和 prompts；任何兼容的 host，包括 Claude Desktop、ChatGPT、Cursor、Claude Code、Zed 以及大量 agent framework，都能发现并调用它们，而不需要定制 glue code。

到 2026 年初，MCP 已经是 Anthropic、OpenAI、Google 三大阵营和主要 agent harness 默认采用的工具与上下文协议。

## The Concept / 概念

![MCP: one host, one server, three capabilities](../assets/mcp-architecture.svg)

**The three primitives / 三个 primitive。** 一个 MCP server 正好暴露三类东西。

1. **Tools**：模型可以调用的函数。对应 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个 tool 都有 name、description、JSON Schema input 和 handler。
2. **Resources**：模型或用户可以请求的只读内容，例如文件、数据库行、API 响应。通过 URI 寻址。
3. **Prompts**：用户可以作为快捷方式调用的可复用模板 prompt。

**The wire format / 线协议格式。** JSON-RPC 2.0，传输层可以是 stdio、WebSocket 或 streamable HTTP。每条消息都是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法是 `tools/list`、`resources/list`、`prompts/list`；调用方法是 `tools/call`、`resources/read`、`prompts/get`。

**Host vs client vs server / Host、client 与 server。** Host 是 LLM 应用，例如 Claude Desktop。Client 是 host 内部的一个子组件，只连接一个 server。Server 是你的代码。一个 host 可以同时挂载多个 server。

### The handshake / 握手

每个 session 都从 `initialize` 开始。Client 发送协议版本和自身 capabilities。Server 返回自己的版本、名称，以及它支持的能力集合（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的一切行为都基于这些 capabilities 协商执行。

### What MCP is not / MCP 不是什么

- 它不是 retrieval API。RAG（Phase 11 · 06）仍然决定拉取什么；MCP 只是把 retrieval 结果作为 resources 暴露出来的传输层。
- 它不是 agent framework。MCP 是管道；LangGraph、PydanticAI、OpenAI Agents SDK 这类框架在它之上。
- 它不绑定 Anthropic。规范和 reference implementations 都在 `modelcontextprotocol` org 下开源。

## Build It / 动手构建

### Step 1: a minimal MCP server / 第 1 步：最小 MCP server

官方 Python SDK 是 `mcp`（之前叫 `mcp-python`）。高层 `FastMCP` helper 用 decorator 注册 handler。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个 decorator 分别注册三种 primitive。Type hints 会变成 host 看到的 JSON Schema。你可以在 Claude Desktop 或 Claude Code 中运行它，把 server entry 指向这个文件。

### Step 2: calling an MCP server from a host / 第 2 步：从 host 调用 MCP server

官方 Python client 会说 JSON-RPC。把它和 Anthropic SDK 接起来只需要十几行。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回的 schema 和 LLM 会看到的 schema 相同。生产 host 会把这些 schema 注入每一轮对话，让模型生成 `tool_use` block，再由 client 转发给 server。

### Step 3: streamable HTTP transport / 第 3 步：streamable HTTP 传输

Stdio 适合本地开发。远程工具使用 streamable HTTP：每个 request 一个 POST，可选 Server-Sent Events 用于进度消息；这个 transport 从 2025-06-18 版规范开始支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host config（Claude Desktop `mcp.json` 或 Claude Code `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Server 里的 decorator 不变，只改变 transport。

### Step 4: scoping and safety / 第 4 步：作用域与安全

MCP tool 是在别人 trust boundary 上运行的任意代码。这里有三条必须遵守的模式。

- **Capability allowlists / 能力 allowlist。** Host 通过 `roots` capability 暴露允许访问的路径。Tool handler 必须强制执行它；不要信任模型传入的路径。
- **Human-in-the-loop for mutation / 变更操作必须有人确认。** 只读工具可以自动执行。写入和删除工具必须要求确认：当 server 在 tool metadata 上设置 `destructiveHint: true` 时，host 应该展示 approval UI。
- **Tool poisoning defense / Tool poisoning 防护。** 恶意 resource 可能包含隐藏的 prompt injection 指令（例如“summarizing 时也调用 `exfil`”）。把 resource 内容视为不可信数据，永远不要让它进入 system-message 级别。参考 Phase 11 · 12 (Guardrails)。

`code/main.py` 里有一个可运行的 server + client 示例，覆盖这些要点。

## Pitfalls that still ship in 2026 / 2026 年仍会上线的坑

- **Schema drift / Schema 漂移。** 模型在第 1 轮看过 `tools/list`。第 5 轮工具集合变了。模型调用了已经不存在的 tool。Host 应该在 `notifications/tools/list_changed` 后重新 list。
- **Large resource blobs / 大型资源块。** 把 2MB 文件直接作为资源正文返回会浪费 context。应该在服务端分页或摘要。
- **Too many servers / server 太多。** 挂载 50 个 MCP server 会打爆 tool budget（Phase 11 · 05）。大多数 frontier model 在约 40 个 tools 后会退化。
- **Version skew / 版本错位。** 规范修订（2024-11、2025-03、2025-06、2025-12）会引入 breaking fields。CI 中要固定 protocol version。
- **Stdio deadlocks / Stdio 死锁。** server 把日志写到 stdout 会污染 JSON-RPC stream。日志只能写 stderr。

## Use It / 应用它

2026 年的 MCP 技术栈：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 本地开发、单用户工具 | Python `FastMCP`，stdio transport |
| 远程团队工具 / SaaS integration | Streamable HTTP，OAuth 2.1 auth |
| TypeScript host（VS Code extension、web app） | `@modelcontextprotocol/sdk` |
| 高吞吐 server、typed access | Official Rust SDK (`modelcontextprotocol/rust-sdk`) |
| 探索生态 server | `modelcontextprotocol/servers` monorepo（Filesystem、GitHub、Postgres、Slack、Puppeteer） |

经验规则：如果一个 tool 是只读的、可缓存的，并且会被两个以上 host 调用，就把它发布成 MCP server。如果它只是一次性的 inline logic，就保留为本地函数（Phase 11 · 09）。

## Ship It / 交付它

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## Exercises / 练习

1. **Easy / 简单。** 给 `demo-server` 增加一个 `subtract` tool。从 Claude Desktop 连接它。通过发出 `tools/list_changed` notification，确认 host 不重启也能发现新 tool。
2. **Medium / 中等。** 增加一个 `resource`，暴露 `/var/log/app.log` 的最后 100 行。强制执行 roots allowlist，确保即使模型请求 `../etc/passwd` 也会被阻止。
3. **Hard / 困难。** 构建一个 MCP proxy，把三个 upstream servers（Filesystem、GitHub、Postgres）multiplex 成一个聚合 surface。处理 name collision，并干净地转发 `notifications/tools/list_changed`。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| MCP | “LLM 的工具协议” | JSON-RPC 2.0 规范，用于把 tools、resources 和 prompts 暴露给任意 LLM host。 |
| Host | “Claude Desktop” | LLM 应用；拥有模型和用户 UI，挂载一个或多个 clients。 |
| Client | “Connection” | Host 内部每个 server 一条连接，负责和唯一一个 server 说 JSON-RPC。 |
| Server | “带工具的那个东西” | 你的代码；声明 tools/resources/prompts，并处理调用。 |
| Tool | “Function call” | 模型可调用的 action，带 JSON Schema input 和 text/JSON result。 |
| Resource | “Read-only data” | 通过 URI 寻址的内容，例如文件、行、API response，host 可以请求。 |
| Prompt | “Saved prompt” | 用户可调用的模板，通常带 arguments，并作为 slash-command 展示。 |
| Stdio transport | “本地开发模式” | 父 host 把 server 作为 child process 启动；JSON-RPC 走 stdin/stdout。 |
| Streamable HTTP | “2025-06 远程传输” | 请求走 POST，可选 SSE 做 server-initiated messages；替代更早的 SSE-only transport。 |

## Further Reading / 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 按日期 versioned 的规范主参考。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem、GitHub、Postgres、Slack、Puppeteer reference servers。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — 发布文章，解释设计动机。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots、destructive hints、tool poisoning。
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent protocol；它是 MCP agent-to-tool 范围之外、面向 agent-to-agent communication 的兄弟标准。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — MCP 在 agent design pattern library（augmented LLM、workflows、autonomous agents）中的位置。
