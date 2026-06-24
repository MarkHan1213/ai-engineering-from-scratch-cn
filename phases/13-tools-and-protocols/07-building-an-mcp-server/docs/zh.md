# Building an MCP Server — Python + TypeScript SDKs / 构建 MCP Server：Python 与 TypeScript SDK

> 大多数 MCP 教程只展示 stdio hello-world。真正的 server 会同时暴露 tools、resources 和 prompts，处理 capability negotiation，发出 structured errors，并且在不同 SDK 中保持同样行为。本课会端到端构建一个 notes server：stdlib stdio transport、JSON-RPC dispatch、三个 server primitives，以及一种 pure-function 风格，之后可以无痛迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, stdio MCP server)
**Prerequisites / 前置知识：** Phase 13 · 06 (MCP fundamentals)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` methods。
- 编写 dispatch loop，从 stdin 读取 JSON-RPC messages，并向 stdout 写 responses。
- 按 JSON-RPC 2.0 spec 和 MCP 的 additional codes 发出 structured error responses。
- 在不重写 tool logic 的前提下，把 stdlib implementation 迁移到 FastMCP（Python SDK）或 TypeScript SDK。

## The Problem / 问题

在使用 remote transport（Phase 13 · 09）或 auth layer（Phase 13 · 16）之前，你需要一个干净的 local server。local 意味着 stdio：client 作为 child process 启动 server，messages 通过 stdin/stdout newline-delimited 流动。

2025-11-25 spec 规定 stdio messages 编码为 JSON objects，并用显式 `\n` 分隔。这里没有 SSE；SSE 是旧的 remote mode，正在 2026 年中移除（Atlassian 的 Rovo MCP server 于 2026 年 6 月 30 日 deprecated，Keboola 于 2026 年 4 月 1 日 deprecated）。对 stdio 来说，每行一个 JSON object 就是完整 wire format。

notes server 是一个好例子，因为它会用到全部三个 server primitives。Tools 做 mutation（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课的形状可以泛化到任何 domain。

## The Concept / 概念

### Dispatch loop / 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- stdout 上不要打印任何不是 JSON-RPC envelope 的内容。debug logs 走 stderr。
- 每个 request 都必须匹配一个携带相同 `id` 的 response。
- Notifications 绝不能被 response。

### Implementing `initialize` / 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的能力。client 会依赖 capability set 来 gate features。

### Implementing `tools/list` and `tools/call` / 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个 entry 有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，返回 `{content: [blocks], isError: bool}`。

Content blocks 是 typed 的。最常见的是：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

tool errors 有两种形状。Protocol-level errors（unknown method、bad params）是 JSON-RPC errors。Tool-level errors（valid call 但工具失败）以 `{content: [...], isError: true}` 返回。这样模型可以在上下文中看到失败。

### Implementing resources / 实现 resources

Resources 在设计上是只读的。`resources/list` 返回 manifest；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...`，或 `notes://` 这样的 custom scheme。

当你把数据暴露为 resource 而不是 tool：

- 模型不会“调用”它；client 可以在用户请求时把它注入 context。
- Subscriptions 允许 server 在 resource 变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 会用 `ui://` 把它扩展为 interactive resources。

### Implementing prompts / 实现 prompts

Prompts 是带命名参数的模板。host 会把它们显示成 slash-commands。一个 `review_note` prompt 可以接收 `note_id` argument，产出一个 multi-message prompt template，再由 client 送进模型。

### Stdio transport subtleties / Stdio transport 细节

- Newline-delimited JSON。没有 length-prefixed framing。
- 不要 buffer。每次写后执行 `sys.stdout.flush()`。
- client 控制生命周期。stdin 关闭（EOF）时，干净退出。
- 不要悄悄吞掉 SIGPIPE；记录日志并退出。

### Annotations / 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — pure read，可以安全 retry。
- `destructiveHint: true` — 不可逆副作用；client 应该确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

client 会用这些决定 UX（confirmation dialogs、status indicators）和 routing（Phase 13 · 17）。

### Graduation path / 升级路径

`code/main.py` 中的 stdlib server 大约 180 行。FastMCP（Python）会把同样逻辑压缩为 decorator-style：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价形状。等你准备好后可以直接升级；概念（capabilities、dispatch、content blocks）保持不变。

## Build It / 动手构建

本课会从 stdio dispatch loop 开始，逐步接入三个 server primitives：先握手，再列出和调用 tools，再暴露 notes resources，最后加入 `review_note` prompt。你会把 domain logic 写成 pure functions，让后续迁移 SDK 时只替换协议外壳。

## Use It / 应用它

`code/main.py` 是一个完整的 notes MCP server，使用 stdio，且只依赖 stdlib。它处理 `initialize`，为三个工具（`notes_list`、`notes_search`、`notes_create`）处理 `tools/list` 与 `tools/call`，为每条 note 处理 `resources/list` 和 `resources/read`，并提供一个 `review_note` prompt。你可以通过 pipe JSON-RPC messages 驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

重点看：

- dispatcher 是一个以 method name 为 key 的 `dict[str, Callable]`。
- 每个 tool executor 返回 content blocks list，而不是 bare string。
- executor 抛错时设置 `isError: true`。

## Ship It / 交付它

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个 domain（notes、tickets、files、database），这个 skill 会 scaffold 一个 MCP server，并正确拆分 tools / resources / prompts，同时给出 SDK graduation path。

## Exercises / 练习

1. 运行 `code/main.py`，用手写 JSON-RPC messages 驱动它。执行 `notes_create`，再用 `resources/read` 取回新 note。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` tool。验证 client 会展示 confirmation dialog（这需要真实 host；Claude Desktop 可用）。

3. 实现 `resources/subscribe`，让 server 在 note 被修改时推送 `notifications/resources/updated`。添加 keepalive task。

4. 把 server 移植到 FastMCP。Python 文件应缩到 80 行以内。wire behavior 必须一致；用同一个 JSON-RPC test harness 验证。

5. 阅读 spec 的 `server/tools` section，找出本课 server 没实现的一个 tool definition 字段。（提示：有好几个；选一个加上。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MCP server | “暴露 tools 的东西” | 通过 stdio 或 HTTP 说 MCP JSON-RPC 的进程 |
| stdio transport | “Child process model” | server 由 client 启动，通过 stdin/stdout 通信 |
| Dispatcher | “Method router” | JSON-RPC method name 到 handler function 的映射 |
| Content block | “Tool result chunk” | tool response 的 `content` array 中的 typed element |
| `isError` | “Tool-level failure” | 表示工具失败；区别于 JSON-RPC error |
| Annotations | “Safety hints” | readOnly / destructive / idempotent / openWorld flags |
| FastMCP | “Python SDK” | MCP protocol 上的 decorator-based higher-level framework |
| Resource URI | “Addressable data” | 标识 resource 的 `file://`、`db://` 或 custom scheme |
| Prompt template | “Slash-command brief” | 带 argument slots 的 server-supplied template |
| Capability declaration | “Feature toggle” | 在 `initialize` 中声明的 per-primitive flags |

## Further Reading / 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — parallel TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — MCP servers 的 decorator-style Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的 end-to-end tutorial
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* messages 的完整参考
