# A2A — The Agent-to-Agent Protocol / A2A：Agent-to-Agent 协议

> Google 在 2025 年 4 月宣布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，已有 150+ 组织支持。A2A 是 MCP（Lesson 13）的横向补充：MCP 是纵向的（agent ↔ tools），A2A 是 peer-to-peer（agent ↔ agent）。它定义 Agent Cards（discovery）、带 artifacts 的 tasks（text、structured data、video）、opaque task lifecycle 和 auth。生产系统越来越常把 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年把 A2A 支持纳入 Vertex AI Agent Builder。

**类型：** 学习 + 构建
**语言：** Python（stdlib, `http.server`, `json`）
**前置知识：** 第 16 阶段 · 04（Primitive Model）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 A2A 的四个核心元素：Agent Card、Task、Artifact、Opaque lifecycle
- 构建最小 A2A server/client：discovery、submit、poll、artifact
- 区分 MCP 的 agent-to-tool 与 A2A 的 agent-to-agent
- 判断 A2A 适合跨组织、异构框架、长任务的场景，以及不适合低延迟微调用的场景

## The Problem / 问题

你的 Agent 需要调用另一个系统上的 Agent。怎么做？你可以暴露一个 HTTP endpoint，定义自家 JSON schema，然后祈祷对方也说同一种语言。每一对 Agent 都变成一次定制集成。

A2A 是这类调用的通用 wire protocol。标准 discovery、标准 task model、标准 transport、标准 artifacts。它像 HTTP+REST，只是把 Agent 当作一等公民。

## The Concept / 概念

### The four elements / 四个元素

**Agent Card.** 位于 `/.well-known/agent.json` 的 JSON 文档，描述 Agent：name、skills、endpoints、supported modalities、auth requirements。discovery 通过读取 card 发生。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task.** 工作单元。一个 async、stateful object，生命周期是 `submitted → working → completed / failed / canceled`。client 发送 task，通过 polling 或 subscription 等待更新。

**Artifact.** task 产出的结果类型。text、structured JSON、image、video、audio。Artifact 带类型，modalities 是一等公民。

**Opaque lifecycle.** A2A 不规定 remote agent *如何* 解决任务。client 看到状态转换和 artifacts；实现可以使用任何框架。

### The MCP/A2A split / MCP 与 A2A 的分工

- **MCP**（Lesson 13）：agent ↔ tool。Agent 通过 JSON-RPC 对 tool server 读写。默认无状态。
- **A2A**：agent ↔ agent。peer protocol；双方都是有自己推理能力的 Agent。

生产多 Agent 系统会同时使用两者。一个 A2A peer 可以在自己一侧调用 MCP tools。这个拆分能保持两个关注点清晰。

### Discovery flow / Discovery 流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

如果使用 streaming，则通过 SSE 订阅 `/tasks/{id}/events` 接收 push updates。

### Auth / 认证

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或 opaque token。
- **mTLS** — mutual TLS；组织互相证明身份。
- **Signed requests** — 对 payload 做 HMAC。

Auth 写在 Agent Card 里；client 发现并遵守。

### 150+ organizations by April 2026 / 2026 年 4 月 150+ 组织

企业采用推动了 A2A 规模化。关键结论：A2A 成为 enterprise agent systems 跨 trust boundary 的方式。Google Cloud 在 Vertex AI Agent Builder 中加入 A2A 支持；Microsoft Agent Framework 支持它；多数主流框架（LangGraph、CrewAI、AutoGen）都提供 A2A adapters。

### Where A2A wins / A2A 适合哪里

- **Cross-organization calls.** 公司 A 的 Agent 调用公司 B 的 Agent。没有 A2A，每一对都是 bespoke contract。
- **Heterogeneous frameworks.** LangGraph agent 调 CrewAI agent，再调 custom Python agent。A2A 做归一化。
- **Typed artifacts.** video result、structured JSON、audio 都是一等结果。
- **Long-running tasks.** opaque lifecycle + polling 让小时级任务变得直接。

### Where A2A struggles / A2A 不适合哪里

- **Latency-sensitive micro-calls.** A2A lifecycle 是 async。亚毫秒 agent-to-agent 不适合；用 direct RPC。
- **Tight-coupled in-process agents.** 两个 Agent 在同一 Python process 里时，A2A 的 HTTP round-trip 是过度设计。
- **Small teams.** 规范开销真实存在；只在内部运行的小 Agent 不一定需要这种正式程度。

### A2A vs ACP, ANP, NLIP / A2A 与 ACP、ANP、NLIP

2024-2026 出现了几个相邻规范：

- **ACP**（IBM/Linux Foundation）— A2A 的前身之一，范围更窄。
- **ANP**（Agent Network Protocol）— 强调 peer discovery 与 decentralized-first。
- **NLIP**（Ecma Natural Language Interaction Protocol，2025 年 12 月标准化）— 自然语言 content type。

截至 2026 年 4 月，A2A 是采用最广的 peer protocol。比较见 arXiv:2505.02279（Liu et al., "A Survey of Agent Interoperability Protocols"）。

## Build It / 动手构建

`code/main.py` 用 `http.server` 和 JSON 实现 A2A-minimal server 和 client。server：

- 暴露 `/.well-known/agent.json`，
- 接收 `POST /tasks`，
- 管理 task state，
- 在 `GET /tasks/{id}` 返回 artifacts。

client：

- 获取 Agent Card，
- 提交 task，
- 轮询直到完成，
- 读取 artifact。

运行：

```
python3 code/main.py
```

脚本在后台线程启动 server，然后让 client 调用它。你会看到完整 flow：discovery、submit、poll、artifact。

## Use It / 应用它

`outputs/skill-a2a-integrator.md` 用来设计 A2A integration：Agent Card 内容、task schemas、auth choice、streaming vs polling。

## Ship It / 交付它

Checklist：

- **Pin the spec version.** A2A 仍在演进；Agent Card 应声明 protocol version。
- **Idempotent task creation.** 重复提交（网络 retry）应得到同一个 task。
- **Artifact schemas.** 声明 Agent 返回的数据形态；consumer 应验证。
- **Rate limits + auth.** A2A 是 public-facing；使用标准 Web security。
- **Dead-letter for failed tasks.** 长期观察失败类型模式。

## Exercises / 练习

1. 运行 `code/main.py`。确认 client 能发现 server 并收到正确 artifact。
2. 给 server 增加第二个 skill（例如 "summarize"）。更新 Agent Card。写一个 client，根据 task type 选择 skill。
3. 实现 SSE streaming endpoint：`/tasks/{id}/events`，发出 state changes。client 需要做什么不同处理？
4. 阅读 A2A spec（https://a2a-protocol.org/latest/specification/）。找出规范强制要求但本 demo 没实现的三件事。
5. 比较 A2A（Agent Card discovery）与 MCP（server-side capability listing via `listTools`）。self-describing agents 和 capability-probing 的取舍是什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | “Agent-to-agent” | Agents 跨系统调用其他 Agent 的 peer protocol。Google 2025。 |
| Agent Card | “Agent 的名片” | `/.well-known/agent.json` 的 JSON，描述 skills、endpoints、auth。 |
| Task | “工作单元” | 异步有状态对象，带 lifecycle；完成后产出 artifacts。 |
| Artifact | “结果” | 类型化输出：text、structured JSON、image、video、audio。一等 media。 |
| Opaque lifecycle | “怎么解决是 Agent 的事” | client 只看状态转换；server 可自由选择 framework/tools。 |
| Discovery | “找到 Agent” | `GET /.well-known/agent.json` 返回 card。 |
| MCP vs A2A | “工具 vs peers” | MCP：纵向 agent ↔ tool。A2A：横向 agent ↔ agent。 |
| ACP / ANP / NLIP | “Sibling protocols” | 相邻规范；A2A 是 2026 采用最广的 peer protocol。 |

## Further Reading / 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
