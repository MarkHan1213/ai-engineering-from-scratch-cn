# A2A — Agent-to-Agent Protocol / A2A：Agent-to-Agent 协议

> MCP 是 agent-to-tool。A2A（Agent2Agent）是 agent-to-agent：一个开放协议，让不同框架构建的 opaque agents 可以协作。它由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，并获得 AWS、Cisco、Microsoft、Salesforce、SAP、ServiceNow 等 150+ 支持者。它吸收了 IBM 的 ACP，并加入 AP2 payments extension。本课会走过 Agent Card、Task lifecycle 和两种 transport bindings。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, Agent Card + Task harness)
**Prerequisites / 前置知识：** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 agent-to-tool（MCP）和 agent-to-agent（A2A）use cases。
- 在 `/.well-known/agent.json` 发布带 skills 和 endpoint metadata 的 Agent Card。
- 走通 Task lifecycle（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（text、file、data）的 Messages 和作为输出的 Artifacts。

## The Problem / 问题

一个 customer-service agent 需要把 report-writing 委派给专门的 writer agent。A2A 之前的选项：

- Custom REST API。能工作，但每一对 agent 都是一套 one-off。
- Shared codebase。要求两个 agents 跑在同一个 framework 上。
- MCP。不合适：MCP 用于调用 tools，不适合两个 agents 在保留各自 opaque internal reasoning 的同时协作。

A2A 填补了这个空白。它把交互建模成一个 agent 向另一个 agent 发送 Task，带 lifecycle、messages 和 artifacts。被调用 agent 的内部状态保持 opaque；caller 只能看到 task state transitions 和最终 outputs。

A2A 是“让不同框架的 agents 彼此通信”的协议。它不替代 MCP；两者互补。

## The Concept / 概念

### Agent Card / Agent Card

每个 A2A-compliant agent 都在 `/.well-known/agent.json` 发布 card：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Discovery 基于 URL：fetch card，获知 A2A endpoint URL，并枚举 skills。

### Signed Agent Cards (AP2) / 签名 Agent Cards（AP2）

AP2 extension（2025 年 9 月）为 Agent Cards 加入 cryptographic signatures。publisher 用 JWT 签名自己的 card；consumers 验证。这样可以防止 impersonation。

### Task lifecycle / Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

clients 通过 `tasks/send` 发起。被调用 agent 穿过各个 state；clients 可以通过 SSE 订阅 state updates，也可以 poll。

### Messages and Parts / Messages 与 Parts

message 携带一个或多个 Parts：

- `text` — plain content。
- `file` — 带 mimeType 的 base64 blob。
- `data` — typed JSON payload（给被调用 agent 的 structured input）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts / Artifacts

outputs 是 Artifacts，而不是 raw strings。Artifact 是一个 named、typed output：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为 chunks streaming。caller 负责累积。

### Two transport bindings / 两种 transport binding

1. **JSON-RPC over HTTP.** `/a2a` endpoint，POST requests，可选 SSE streaming。默认 binding。
2. **gRPC.** 适合 gRPC 原生的 enterprise environments。

两种 binding 承载同样的 logical message shape。

### Opacity preservation / 保持不透明性

关键设计原则：被调用 agent 的内部状态是不透明的。caller 看到 task state 和 artifacts。被调用 agent 的 chain-of-thought、tool calls、sub-agent delegation 都不可见。这不同于 MCP；MCP 中的 tool calls 是透明的。

理由：A2A 允许竞争者在不泄露内部机制的情况下协作。A2A 可以是“调用这个 customer-service agent”，但 caller 不知道该 agent 如何实现服务。

### Timeline / 时间线

- **2025-04-09.** Google announces A2A.
- **2025-06-23.** Donated to Linux Foundation.
- **2025-08.** Absorbs IBM's ACP.
- **2025-09.** AP2 extension (Agent Payments) ships.
- **2026-04.** v1.0 released with 150+ supporting organizations.

### Relationship to MCP / 与 MCP 的关系

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用一个具体工具时，用 MCP。当你想把整个任务委派给另一个 agent 时，用 A2A。很多生产系统会同时使用：一个 agent 用 MCP 做工具层，用 A2A 做协作层。

## Build It / 动手构建

本课会实现一个最小 A2A harness：research agent 发布 Agent Card，writer agent 接收 `tasks/send`，处理中途进入 `input_required`，最终返回 text artifact。重点是 Task lifecycle 和 message shape，而不是网络框架。

## Use It / 应用它

`code/main.py` 实现一个 minimal A2A harness：research agent 发布 card，writer agent 接收一个 `tasks/send`，其中 parts 包含 PDF 和 text instruction；它经历 working → input_required → working → completed，并返回一个 text artifact。全部使用 stdlib；使用 in-memory transport 以聚焦 message shapes。

重点看：

- Agent Card JSON shape。
- Task id assignment 和 state transitions。
- mixed-type parts 的 messages。
- task 中途的 input-required branch。
- completion 时返回 artifact。

## Ship It / 交付它

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应被其他 agents 调用的新 agent，这个 skill 会生成 Agent Card JSON、skills schema 和 endpoint blueprint。

## Exercises / 练习

1. 运行 `code/main.py`。追踪完整 Task lifecycle，包括 called agent 请求 clarification 的 input-required pause。

2. 添加 signed Agent Card。对 card 的 canonical JSON 做 HMAC 签名。写一个 verifier，确认 card 被修改时验证失败。

3. 实现 task streaming：writer agent 通过 SSE 发出三个 incremental artifact chunks，caller 累积它们。

4. 设计一个包装 MCP server 的 A2A agent。把每个 MCP tool 映射为一个 A2A skill。记录 trade-offs：失去了什么 opacity？

5. 阅读 A2A v1.0 announcement，找出截至 2026 年 4 月没有任何 framework 实现的一个 feature。（提示：它和 multi-hop task delegation 有关。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | “Agent-to-Agent protocol” | 用于 opaque agent collaboration 的开放协议 |
| Agent Card | “`.well-known/agent.json`” | 描述 agent skills 和 endpoint 的发布 metadata |
| Skill | “A callable unit” | agent 支持的 named operation（类似 MCP tool） |
| Task | “Unit of delegation” | 带 lifecycle 和 final artifact 的 work item |
| Message | “Task input” | 携带 Parts（text、file、data） |
| Part | “Typed chunk” | message 中的 `text` / `file` / `data` element |
| Artifact | “Task output” | completion 时返回的 named、typed output |
| AP2 | “Agent Payments Protocol” | 用于 trust 和 payments 的 signed Agent Cards extension |
| Opacity | “Black-box collaboration” | called agent 的 internals 对 caller 隐藏 |
| Input-required | “Task pause” | agent 需要更多信息时的 lifecycle state |

## Further Reading / 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — canonical A2A specification
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — reference implementations 和 SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月 governance transfer
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — roadmap 和 partner momentum
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 release notes 和 backward-compat guidance
