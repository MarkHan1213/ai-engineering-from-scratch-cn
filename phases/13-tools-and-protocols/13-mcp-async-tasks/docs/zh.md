# Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work / Async Tasks（SEP-1686）：长任务先调用、后获取

> 真实 agent 工作常常需要几分钟到几小时：CI runs、deep-research synthesis、batch exports。同步 tool call 会断连接、超时或阻塞 UI。SEP-1686 在 2025-11-25 合并，加入 Tasks primitive：任何 request 都可以被增强为 task，结果可以稍后获取，或通过 state notifications 流式推送。Drift-risk note：Tasks 到 2026 H1 仍属 experimental；SDK surface 仍在围绕 spec 设计。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, async task state machine)
**Prerequisites / 前置知识：** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 判断什么时候应把 tool 从 synchronous 提升为 task-augmented（server-side work >30 秒）。
- 走通 task lifecycle：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化 task state，让 crash 不会丢失 in-flight work。
- 正确 poll `tasks/status` 并 fetch `tasks/result`。

## The Problem / 问题

一个 `generate_report` tool 会运行多分钟 extraction pipeline。在同步模型下有几个选择：

1. 把连接保持三分钟。remote transports 会断；clients 会超时；UIs 会卡住。
2. 立即返回 placeholder；要求 client poll 一个 custom endpoint。破坏 MCP uniformity。
3. Fire-and-forget；没有结果。

都不好。SEP-1686 加入第四种：task augmentation。任何 request（通常是 `tools/call`）都可以标记为 task。server 立即返回 task id。client poll `tasks/status`，完成后 fetch `tasks/result`。server-side state 能在重启后保留。

## The Concept / 概念

### Task augmentation / Task 增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由 server 决定），request 会变成 task。server 立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是 server 承诺保留状态的时间；ttl 之后 task result 会被丢弃。

### Per-tool opt-in / 每个工具的 opt-in

Tool annotations 可以声明 task support：

- `taskSupport: "forbidden"` — 这个工具永远同步运行。适合 fast tools。
- `taskSupport: "optional"` — client 可以请求 task-augmentation。
- `taskSupport: "required"` — client 必须使用 task augmentation。

`generate_report` tool 应该是 `required`。`notes_search` tool 应该是 `forbidden`。

### States / 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

state machine 是 append-only：一旦进入 `completed`、`failed` 或 `cancelled`，task 就是 terminal。

### Methods / 方法

- `tasks/status {taskId}` — 返回当前 state 和 progress hint。
- `tasks/result {taskId}` — 阻塞或在未完成时返回 404。
- `tasks/cancel {taskId}` — idempotent；terminal states 会忽略。
- `tasks/list` — optional；枚举 active 和 recently-completed tasks。

### Streaming state changes / 流式状态变化

server 支持时，client 可以订阅 state notifications：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

使用 stream 而不是 poll 的 clients 可以获得更好 UX。polling 始终是最小 surface。

### Durable state / 持久状态

spec 要求声明 task support 的 servers 持久化状态。crash 不应丢失 ttl 内已完成结果。store 可以是 SQLite、Redis 或 filesystem。Lesson 13 harness 使用 filesystem。

### Cancellation semantics / 取消语义

`tasks/cancel` 是 idempotent。如果 task 正在执行，server 会尝试停止（依赖 executor-cooperative cancellation）。如果已经 terminal，请求就是 no-op。

### Crash recovery / 崩溃恢复

server process 重启时：

1. 加载所有 persisted task states。
2. 把进程死亡时仍处于 `working` 的 tasks 标记为 `failed`，error 为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### Async tasks plus sampling / Async tasks 加 sampling

task 本身可以调用 `sampling/createMessage`。这就是 long-running research tasks 的工作方式：server 的 task thread 按需采样 client 模型，同时 client UI 显示 task 为 `working` 并周期性更新 progress。

### Why this is experimental / 为什么仍是 experimental

SEP-1686 已在 2025-11-25 发布，但 broader roadmap 明确指出三个 open issues：durable subscription primitives、subtasks（parent-child task relationships）和 result-TTL standardization。预计 spec 会在 2026 年继续演进。生产代码应只把 common case 当作稳定，并为未来 SDK subtasks changes 做 guard。

## Build It / 动手构建

本课会实现一个 filesystem-backed task store，以及一个后台线程执行的 `generate_report` tool。核心是状态机、ttl、progress update、cancel event 和 crash recovery，而不是具体 report 生成逻辑。

## Use It / 应用它

`code/main.py` 实现一个 durable task store（filesystem-backed）和一个在 background thread 中运行的 `generate_report` tool。clients 调用工具后立刻拿到 task id，在 worker 更新 progress 时 poll `tasks/status`，完成后 fetch `tasks/result`。Cancellation 可用；crash recovery 通过杀掉 worker thread 并 reload state 来模拟。

重点看：

- Task state JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker thread 更新 `progress` field；poll 能看到它前进。
- client-side cancellation 会设置 event；worker 检查后提前退出。
- "crash" 后 state reload 会把 in-flight task 标记为带 `CRASH_RECOVERY` 的 `failed`。

## Ship It / 交付它

本课产出 `outputs/skill-task-store-designer.md`。给定一个 long-running tool（research、build、export），这个 skill 会设计 task store（state shape、ttl、durability），选择合适的 taskSupport flag，并草拟 progress notifications。

## Exercises / 练习

1. 运行 `code/main.py`。启动一个 `generate_report` task，poll status，然后 fetch result。

2. 在运行中添加一个 `tasks/cancel` call。验证 worker 会遵守它，state 变成 `cancelled`。

3. 模拟 crash recovery：杀掉 worker thread，重启 loader，观察 `CRASH_RECOVERY` failure mode。

4. 把 store 扩展到 SQLite。Durability 收益相同，但 query options 会打开（例如 list all tasks from session X）。

5. 阅读 MCP 2026 roadmap post。找出一个最可能影响下一年 SDK API design 的 Tasks-related open issue。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Task | “Long-running tool call” | 用 `_meta.task` 增强的 request，用于 async execution |
| SEP-1686 | “Tasks spec” | 在 2025-11-25 加入 Tasks 的 Spec Evolution Proposal |
| `_meta.task` | “Task envelope” | per-request metadata，包含 id、state、ttl |
| taskSupport | “Tool flag” | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “Poll method” | 获取当前 state 和可选 progress hint |
| `tasks/result` | “Fetch result” | 返回完成 payload；未完成时 404 |
| `tasks/cancel` | “Stop it” | idempotent cancellation request |
| ttl | “Retention budget” | server 承诺保留 task state 的毫秒数 |
| `notifications/tasks/updated` | “State push” | server-initiated state-change event |
| Durable store | “Crash-safe state” | filesystem / SQLite / Redis persistence layer |

## Further Reading / 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — originating proposal 和完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带 rationale 的 design walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics 和 state machine
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK-level task implementation patterns
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 包含 subtasks 的 open issues 和 2026 priorities
