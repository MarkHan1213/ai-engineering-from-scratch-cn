# Claude Agent SDK: Subagents and Session Store / Claude Agent SDK：Subagents 与 Session Store

> Claude Agent SDK 是 Claude Code harness 的 library 形态。内建工具、用于 context isolation 的 subagents、hooks、W3C trace propagation、session store parity。Claude Managed Agents 是面向长时间异步工作的托管替代方案。

**Type / 类型：** Learn + Build / 学习 + 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 Anthropic Client SDK（raw API）和 Claude Agent SDK（harness shape）的区别。
- 描述 subagents：parallelization 和 context isolation，以及什么时候使用它们。
- 说出 Python SDK 的 session store surface（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个 stdlib harness，带 built-in tools、带 isolated context 的 subagent spawning、lifecycle hooks 和 session store。

## The Problem / 问题

raw LLM API 只能给你一次 round-trip。生产 Agent 需要 tool execution、MCP servers、lifecycle hooks、subagent spawning、session persistence、trace propagation。Claude Agent SDK 把这个形状作为 library 提供出来，也就是 Claude Code 使用的同一套 harness，开放给 custom agents。

## The Concept / 概念

### Client SDK vs Agent SDK / Client SDK 与 Agent SDK

- **Client SDK (`anthropic`)。** Raw Messages API。循环、工具、状态都由你负责。
- **Agent SDK (`claude-agent-sdk`)。** built-in tool execution、MCP connections、hooks、subagent spawning、session store。Claude Code loop 的 library 形态。

### Built-in tools / 内建工具

SDK 开箱提供 10+ 工具：file read/write、shell、grep、glob、web fetch 等。自定义工具通过标准 tool-schema interface 注册。

### Subagents / Subagents

Anthropic 文档中 subagents 有两个用途：

1. **Parallelization。** 并发运行独立工作。“为这 20 个模块分别找到 test file” 就是 20 个并行 subagent tasks。
2. **Context isolation。** Subagents 使用自己的 context window；只有结果返回 orchestrator。orchestrator 的预算被保留。

Python SDK 新近增加了 `list_subagents()`、`get_subagent_messages()`，用于读取 subagent transcripts。

### Session store / Session store

与 TypeScript 协议保持 parity：

- `append(session_id, message)`：添加一轮。
- `load(session_id)`：恢复对话。
- `list_sessions()`：枚举。
- `delete(session_id)`：删除，并级联 subagent sessions。
- `list_subkeys(session_id)`：列出 subagent keys。

`--session-mirror`（CLI flag）会在 streaming 时把 transcript 镜像到外部文件，用于 debugging。

### Hooks / Hooks

可注册的 lifecycle hooks：

- `PreToolUse`、`PostToolUse`：gate 或 audit tool calls。
- `SessionStart`、`SessionEnd`：setup 和 teardown。
- `UserPromptSubmit`：在模型看到用户输入前处理。
- `PreCompact`：context compaction 前运行。
- `Stop`：Agent 退出时 cleanup。
- `Notification`：side-channel alerts。

Hooks 是 pro-workflow（Phase 14 curriculum reference）和类似系统添加 cross-cutting behavior 的方式。

### W3C trace context / W3C trace context

caller 上活跃的 OTel spans 会通过 W3C trace context headers 传播进 CLI subprocess。整个 multi-process trace 会在 backend 中显示成同一条 trace。

### Claude Managed Agents / Claude Managed Agents

托管替代方案（beta header `managed-agents-2026-04-01`）。面向 long-running async work，内建 prompt caching 和 compaction。用托管基础设施换取控制权。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Subagent over-spawn。** 为 100 个小任务生成 100 个 subagents。overhead 会压过收益。改为 batch。
- **Hook creep。** 每个团队都加 hooks，startup time 膨胀。每季度 review hooks。
- **Session bloat。** sessions 积累，size 增长。使用 `list_sessions` + expiry policy。

## Build It / 动手构建

`code/main.py` 用 stdlib 实现 SDK shape：

- `Tool`、`ToolRegistry`，内建 `read_file`、`write_file`、`list_dir`。
- `Subagent`：private context、isolated run、返回 results。
- `SessionStore`：append、load、list、delete、list_subkeys。
- `Hooks`：`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- demo：main agent 并行 spawn 3 个 subagents（每个 isolated），聚合结果，并持久化 session。

运行：

```
python3 code/main.py
```

trace 会展示 subagent context isolation（orchestrator context size 保持有界）、hook execution 和 session persistence。

## Use It / 应用它

- **Claude Agent SDK** 用于想要 Claude Code harness shape 的 Claude-first products。
- **Claude Managed Agents** 用于 hosted long-running async work。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI-first counterpart。
- **LangGraph + custom tools** 用于你更想要 graph-shaped state machine 的场景。

## Ship It / 交付它

`outputs/skill-claude-agent-scaffold.md` scaffold 一个 Claude Agent SDK app，带 subagents、hooks、session store、MCP server attachment 和 W3C trace propagation。

## Exercises / 练习

1. 增加 subagent spawner，把 20 个 tasks 分成每组 5 个 parallel subagents。测 orchestrator context size 与 one-per-task 的差异。
2. 实现 `PreToolUse` hook，对 `write_file` calls 做 rate limit（每 session 每分钟 5 次）。trace 行为。
3. 把 `list_subkeys` 接成 subagent tree 渲染。深层嵌套长什么样？
4. 把 toy 移植到真实 `claude-agent-sdk` Python package。tool registration 有什么变化？
5. 阅读 Claude Managed Agents docs。什么时候你会从 self-hosted 切换到 managed？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | “Claude Code as a library” | harness shape：tools、MCP、hooks、subagents、session store |
| Subagent | “Child agent” | 独立 context、独立 budget；结果向上冒泡 |
| Session store | “Conversation DB” | 持久化、加载、列出、删除 turns，并级联 subagent |
| Hook | “Lifecycle callback” | pre/post tool、session、prompt submit、compact、stop |
| W3C trace context | “Cross-process trace” | parent span 传播进 CLI subprocess |
| Managed Agents | “Hosted harness” | Anthropic-hosted long-running async work |
| `--session-mirror` | “Transcript mirror” | streaming 时把 session turns 写到外部文件 |
| MCP server | “Tool surface” | 挂到 Agent 上的外部 tool / resource source |

## Further Reading / 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的 library 形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — production patterns
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted alternative
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — counterpart
