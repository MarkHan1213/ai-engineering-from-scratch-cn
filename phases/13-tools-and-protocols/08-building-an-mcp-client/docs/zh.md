# Building an MCP Client — Discovery, Invocation, Session Management / 构建 MCP Client：发现、调用与会话管理

> 大多数 MCP 内容都在写 server tutorial，对 client 只是一笔带过。真正困难的 orchestration 在 client code：process spawning、capability negotiation、跨多个 server 的 tool list merging、sampling callbacks、reconnection，以及 namespace collision resolution。本课会构建一个 multi-server client，把三个不同 MCP servers 提升到一个供模型使用的 flat tool namespace 中。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, multi-server MCP client)
**Prerequisites / 前置知识：** Phase 13 · 07 (building an MCP server)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 将 MCP server 作为 child process 启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护 per-server session state（capabilities、tool list、last-seen notification ids）。
- 把多个 server 的 tool lists 合并成一个 namespace，并处理 collisions。
- 把 tool call 路由到拥有它的 server，并重新组装 response。

## The Problem / 问题

真实 agent host（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP servers。用户可能同时运行 filesystem server、Postgres server 和 GitHub server。client 的工作是：

1. 启动每个 server。
2. 独立完成每个 handshake。
3. 对每个 server 调用 `tools/list`，并 flatten 结果。
4. 当模型发出 `notes_search` 时，在 merged namespace 中查找，并路由到正确 server。
5. 不阻塞地处理来自任意 server 的 notifications（`tools/list_changed`）。
6. transport failure 后重连。

手写这些，是 "toy" 和 "serviceable" 的分界。官方 SDK 会封装它，但 mental model 必须属于你。

## The Concept / 概念

### Child-process spawning / 启动子进程

使用 `subprocess.Popen`，并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设 `bufsize=1`，使用 text mode 做 line-by-line reads。每个 server 是一个 process；client 为每个 server 持有一个 `Popen` handle。

### Per-server session state / 每个 server 的会话状态

每个 server 有一个 `Session` object，包含：

- `process` — Popen handle。
- `capabilities` — server 在 `initialize` 中声明的能力。
- `tools` — 最近一次 `tools/list` 结果。
- `pending` — request id 到等待 response 的 promise/future 的 map。

Requests 天然是 async 的；发给 server A 的 `tools/call` 不应因为 server B 正在调用而阻塞。可以使用 threads + queues，也可以使用 asyncio。

### Merged namespace / 合并命名空间

client 看 aggregate tool list 时，name 可能冲突。两个 server 都可能暴露 `search`。client 有三种选项：

1. **Prefix by server name.** `notes/search`、`files/search`。清楚但不美观。
2. **Silent first-come.** 后加载 server 的 `search` 覆盖前一个。风险高；会隐藏 collisions。
3. **Collision rejection.** 拒绝加载第二个 server，并通知用户。对安全敏感 host 最稳。

Claude Desktop 使用 prefix-by-server。Cursor 使用 collision rejection，并给清晰错误。VS Code MCP 也采用 prefix-by-server。

### Routing / 路由

合并后，dispatch table 映射 `tool_name -> session`。模型按 name 发出调用；client 找到 session，把 `tools/call` message 写入该 server 的 stdin，然后等待 response。

### Sampling callback / Sampling 回调

如果 server 在 `initialize` 声明了 `sampling` capability，它可能发送 `sampling/createMessage`，要求 client 运行它的 LLM。client 必须：

1. 在 sample resolve 前阻塞发往该 server 的进一步 requests，或在实现支持 concurrency 时 pipeline。
2. 调用自己的 LLM provider。
3. 把 response 发回 server。

Lesson 11 会端到端讲 sampling。本课只 stub 它以保持完整性。

### Notification handling / Notification 处理

`notifications/tools/list_changed` 表示需要重新调用 `tools/list`。`notifications/resources/updated` 表示如果该 resource 正在使用，就重新读取。Notifications 不能产生 responses，不要试图 ack。

常见 client bug：在 read loop 上阻塞等待 `tools/call`，导致 notification 卡在 stream 里。使用 background reader thread，把每条 message 推到 queue；main thread dequeue 后 dispatch。

### Reconnection / 重连

transport 可能失败：server crashed、OS kill 了 process、stdio pipe 断开。client 在 stdout 上检测到 EOF，并把 session 标记为 dead。选项：

- 静默重启 server 并重新 handshake。适合 pure read-only servers。
- 把 failure 暴露给用户。适合有用户可见会话状态的 stateful servers。

Phase 13 · 09 会讲 Streamable HTTP reconnection semantics；stdio 更简单。

### Keepalive and session id / Keepalive 与 session id

Streamable HTTP 使用 `Mcp-Session-Id` header。Stdio 没有 session id，process identity 就是 session。Keepalive pings 是可选的；stdio pipes 不会因为空闲而断开。

## Build It / 动手构建

本课会实现一个 multi-server client harness：启动多个 toy servers，分别握手，读取各自 tool list，构建 dispatch table，并在 name collision 时做 prefix。它会让你看清 MCP client 的核心不是“调用工具”，而是维护多条独立 session 并把它们折叠成模型可理解的一个工具面。

## Use It / 应用它

`code/main.py` 会把三个 simulated MCP servers 作为 subprocesses 启动，分别 handshake，合并它们的 tool lists，并把 tool calls 路由到正确 server。这些 "servers" 实际是运行 toy responders 的其他 Python processes（没有真实 LLM）。运行后你会看到：

- 三次 initialization，每次都有自己的 capability set。
- 三个 `tools/list` 结果被合并成一个 7-tool namespace。
- 基于 tool name 的 routing decision。
- 通过 namespace prefixing 防止 collision。

重点看：

- `Session` dataclass 干净保存 per-server state。
- background reader thread 会 drain stdout 上的每一行，不阻塞 main thread。
- dispatch table 是简单的 `dict[str, Session]`。
- collision handling 是显式的：两个 server 声明同名工具时，后者会带 prefix 重命名。

## Ship It / 交付它

本课产出 `outputs/skill-mcp-client-harness.md`。给定一组声明式 MCP servers（name、command、args），这个 skill 会生成 harness：启动它们、合并 tool lists，并交付一个带 collision resolution 的 routing function。

## Exercises / 练习

1. 运行 `code/main.py`，观察 server spawn log。用 SIGTERM 杀掉一个 simulated server process，观察 client 如何检测 EOF 并把该 session 标记为 dead。

2. 实现 namespace prefixing。当两个 servers 暴露 `search` 时，把第二个重命名为 `<server>/search`。更新 dispatch table，验证 tool calls 路由正确。

3. 为 server restart 添加 connection-pool-style backoff：连续失败时指数退避，上限 30 秒，三次失败后向用户发出 notification。

4. 画一个支持 100 个 concurrent MCP servers 的 client。什么数据结构替代 simple dispatch dict？（提示：用于 prefix namespacing 的 trie，再加 tool-count-per-server metric。）

5. 把 client 移植到官方 MCP Python SDK。SDK 包装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩到约 40 行，同时保留 multi-server routing。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MCP client | “agent host” | 启动 servers 并编排 tool calls 的进程 |
| Session | “Per-server state” | capabilities、tool list 和 pending-request bookkeeping |
| Merged namespace | “一个 tool list” | 所有 active servers 的 flat tool name 集合 |
| Namespace collision | “两个 server 同名 tool” | client 必须 prefix、reject 或 first-come duplicate |
| Routing | “谁处理这个调用？” | 从 tool name dispatch 到 owning server |
| Background reader | “Non-blocking stdout” | 把 server stdout drain 到 queue 的 thread 或 task |
| Sampling callback | “LLM-as-a-service” | client 处理 server 发来的 `sampling/createMessage` |
| `notifications/*_changed` | “Primitive mutated” | 通知 client 必须重新 discover 或 re-read |
| Reconnection policy | “server 死了怎么办” | transport 失败后的 restart semantics |
| Stdio session | “Process = session” | 没有 session id；child process 生命周期就是 session |

## Further Reading / 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — canonical client behavior
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world client tutorial
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — reference `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS parallel
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单一 editor host 中 multiplex 多个 MCP servers
