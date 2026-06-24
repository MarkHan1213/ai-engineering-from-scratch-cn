# AutoGen v0.4: Actor Model and Agent Framework / AutoGen v0.4：Actor Model 与 Agent Framework

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 actor model 重新设计了 Agent 编排。异步消息交换、event-driven agents、fault isolation、天然并发。现在该框架进入 maintenance mode，Microsoft Agent Framework（2025 年 10 月 public preview）成为后继者。

**Type / 类型：** Learn + Build / 学习 + 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 actor model：agents as actors，messages 是唯一 IPC，每个 actor 独立隔离失败。
- 说出 AutoGen v0.4 的三层 API：Core、AgentChat、Extensions，以及各自用途。
- 解释为什么 decouple message delivery from handling 能带来 fault isolation 和 natural concurrency。
- 用 Python 实现 stdlib actor runtime，并把 two-agent code-review flow 移植到它上面。

## The Problem / 问题

多数 Agent framework 是同步的：一个 Agent 产出，另一个 Agent 消费，挂在一个 call stack 上。失败会崩掉整条栈。并发是后补的。分布式需要重写。

AutoGen v0.4 的答案是 actor model。每个 Agent 都是拥有 private inbox 的 actor。消息是唯一交互方式。runtime 把 delivery 和 handling 解耦。失败隔离在单个 actor 中。并发是原生的。分布式只是不同 transport。

## The Concept / 概念

### Actors / Actors

一个 actor 有：

- 私有 state（外部不能直接触碰）。
- inbox（message queue）。
- handler：`receive(message) -> effects`，其中 effects 可以是 “reply”、“send to other actor”、“spawn new actor”、“update state”、“stop self”。

两个 actors 不能共享内存。它们只能发送消息。

### Three API layers in AutoGen v0.4 / AutoGen v0.4 的三层 API

1. **Core。** 低层 actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，event-driven。
2. **AgentChat。** task-driven 高层 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成层：OpenAI、Anthropic、Azure、tools、memory。

### Why decoupling matters / 为什么解耦重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 把消息放进 agent_b 的 inbox 然后返回。runtime 稍后投递。三个后果：

- **Fault isolation。** Agent B crash 不会 crash Agent A；runtime 捕获 B handler 的失败并决定如何处理（log、retry、dead-letter）。
- **Natural concurrency。** 多条消息同时 in flight；actors 并发处理 inbox。
- **Distribution-ready。** inbox + transport 在 in-process 和跨 host 下是同一个抽象。

### Topologies / 拓扑

- **RoundRobinGroupChat。** Agents 按固定轮转顺序发言。
- **SelectorGroupChat。** selector agent 基于 conversation context 选择下一个谁发言。
- **Magentic-One。** 面向 web browsing、code execution、file handling 的 reference multi-agent team，构建在 AgentChat 上。

### Observability / 可观测性

内建 OpenTelemetry 支持。每条 message 都发出 span；tool calls 按 2026 OTel GenAI semantic conventions（Lesson 23）携带 `gen_ai.*` attributes。

### Status: maintenance mode / 状态：维护模式

2026 年初：AutoGen v0.7.x 对 research 和 prototyping 足够稳定。Microsoft 已把 active development 转向 Microsoft Agent Framework（2025 年 10 月 1 日 public preview；1.0 GA 目标为 2026 Q1 末）。AutoGen patterns 可以平滑前移，actor model 才是持久思想。

## Build It / 动手构建

`code/main.py` 实现 stdlib actor runtime：

- `Message`：带 `sender`、`recipient`、`topic`、`body` 的 typed payload。
- `Actor`：抽象类，含 `receive(message, runtime)`。
- `Runtime`：带 shared queue、delivery、failure isolation 的 event loop。
- two-actor demo：`ReviewerAgent` review code，`ChecklistAgent` 运行 checklist；二者交换消息直到达成 consensus。

运行：

```
python3 code/main.py
```

trace 会展示 message delivery、某个 actor 的模拟失败不会 crash 另一个 actor，以及最终收敛到 shared verdict。

## Use It / 应用它

- **AutoGen v0.4/v0.7**（maintenance）：适合 research、prototyping、multi-agent patterns。
- **Microsoft Agent Framework**（public preview）：前进路径；同样的 actor-model ideas，新的 API。
- **LangGraph swarm topology**（Lesson 13）：通过 shared-tool handoffs 实现类似模式。
- **Custom actor runtime**：当你需要特定 transport（NATS、RabbitMQ、gRPC）时。

## Ship It / 交付它

`outputs/skill-actor-runtime.md` 会为给定 multi-agent task 生成 minimal actor runtime 和 team template（RoundRobin 或 Selector）。

## Exercises / 练习

1. 增加 dead-letter queue：handler 抛异常时，把失败消息停到队列供 human inspection。在 toy 中 DLQ 会命中几次？
2. 实现 `SelectorGroupChat`：selector actor 基于 conversation state 选择下一个处理消息的人。
3. 增加 distributed transport：把 in-process queue 换成 JSON-over-HTTP server，让 actors 分别跑在不同进程中。
4. 给每条 message 接 OTel span（或 no-op stand-in）。按 Lesson 23 发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 architecture post。把 toy 移植到真实 `autogen_core` API。你跳过了哪些生产里重要的东西？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Actor | “Agent” | private state + inbox + handler；无共享内存 |
| Message | “Event” | typed payload；actors 交互的唯一方式 |
| Inbox | “Mailbox” | 每个 actor 的 pending messages 队列 |
| Runtime | “Agent host” | 路由消息并隔离失败的 event loop |
| Topic | “Channel” | actors 之间的命名 publish-subscribe route |
| Fault isolation | “Let it crash” | 一个 actor 失败不会 crash 其他 actors |
| RoundRobinGroupChat | “Fixed-rotation team” | Agents 按顺序轮流发言 |
| SelectorGroupChat | “Context-routed team” | Selector 选择下一个谁发言 |
| Magentic-One | “Reference team” | 用于 web + code + files 的 multi-agent squad |

## Further Reading / 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — redesign post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graph-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 spans
