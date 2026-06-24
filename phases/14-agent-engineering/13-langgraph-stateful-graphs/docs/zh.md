# LangGraph: Stateful Graphs and Durable Execution / LangGraph：有状态图与持久执行

> LangGraph 是 2026 年低层有状态编排的参考。Agent 是 state machine；nodes 是 functions；edges 是 transitions；state 不可变，并在每一步之后 checkpoint。任何失败都能从停下的位置准确恢复。

**Type / 类型：** Learn + Build / 学习 + 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 LangGraph 的核心模型：带 immutable state、function nodes、conditional edges 和 post-step checkpoints 的 state machine。
- 说出文档强调的四个能力：durable execution、streaming、human-in-the-loop、comprehensive memory。
- 解释 LangGraph 支持的三种 orchestration topologies：supervisor、peer-to-peer (swarm)、hierarchical (nested subgraphs)。
- 实现一个 stdlib state graph，包含 immutable state、conditional edges 和 checkpoint / resume cycle。

## The Problem / 问题

Agent 和 workflow 共享一个问题：一个 40 步 run 在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二等公民的 state model 会让 operators 在一个假设 fresh runs 的库外面硬拼 retry。

LangGraph 的设计答案是：state 是一等 typed object，mutations 是显式的，每个 node 之后都持久化 checkpoint。resume 就是一次 `load_state(session_id)` 调用。

## The Concept / 概念

### The graph / 图

一个 graph 由这些东西定义：

- **State type。** 一个 typed dict（或 Pydantic model），每个 node 都读取并更新它。
- **Nodes。** 纯函数 `(state) -> state_update`。返回后 update 会 merge 进 state。
- **Edges。** 节点之间的 conditional 或 direct transitions。
- **Entry and exit。** `START` 和 `END` sentinel nodes 标记边界。

例子：一个含 `classify`、`refund`、`bug`、`sales`、`done` 节点的 Agent，也就是 graph 形态的 routing workflow。

### Durable execution / 持久执行

每个 node 返回后，runtime 会序列化 state，并写到 checkpointer（SQLite、Postgres、Redis、自定义）。第 N 步失败时，runtime 可以 `resume(session_id)`，用精确 state 从第 N+1 步继续。

LangGraph docs 明确强调这个能力服务于 Klarna、Uber、J.P. Morgan 等生产用户。重点不是 graph shape，而是 graph shape 加 checkpointing 让恢复成本降低。

### Streaming / 流式输出

每个 node 可以 yield partial output。graph 向 caller stream per-node-delta events，因此 UI 能随着 graph 运行持续更新。

### Human-in-the-loop / 人在回路中

在 nodes 之间检查并修改 state。实现方式：在关键 node 前暂停，把 state 展示给 human，接受修改，然后 resume。checkpointer 让它容易，因为 state 已经序列化。

### Memory / 记忆

短期（run 内：state 中的 conversation history）和长期（跨 run：通过 checkpointer + 独立 long-term store 持久化）。LangGraph 通过 tools 集成外部 memory systems（Mem0、自定义）。

### Three topologies / 三种拓扑

1. **Supervisor。** 中央 router LLM 派发到 specialist subagents。`langgraph-supervisor` 中有 `create_supervisor()`，但 LangChain 团队在 2026 年建议通过 tool calls 直接做，以便更好控制 context。
2. **Swarm / peer-to-peer。** Agents 通过共享 tool surface 直接 hand off。没有中央 router。
3. **Hierarchical。** Supervisors 管理 sub-supervisors，通过 nested subgraphs 实现。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Checkpoints too small。** 只 checkpoint conversation turns，会导致 tool state 和 memory writes 无法恢复。必须序列化 full state。
- **Non-deterministic nodes。** Resume 假设相同 node inputs 会产生相同 state update。random seeds、wall-clock、external APIs 必须被捕获。
- **Over-use of conditional edges。** 每条 edge 都 conditional 的 graph，是一个难以推理的 state machine。优先线性链，只在必要处 branch。

## Build It / 动手构建

`code/main.py` 实现一个 stdlib stateful graph：

- `State`：typed dict，含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node`：接收 state 并返回 update dict 的 callable。
- `StateGraph`：nodes + edges + conditional edges + run + resume。
- `SQLiteCheckpointer`（in-memory fake）：每个 node 后序列化 state；`load(session_id)` 恢复。
- demo graph：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

trace 会展示第一次 run 在 human gate 失败、持久化，然后 resume 并产出最终输出。

## Use It / 应用它

- **LangGraph**：参考实现，production-ready。用 `create_react_agent`、`create_supervisor`，或构建自己的 graph。
- **AutoGen v0.4**（Lesson 14）：适合高并发场景的 actor model alternative。
- **Claude Agent SDK**（Lesson 17）：带 built-in session store 的 managed harness。
- **Custom**：当你需要精确控制 state shape 或 checkpointer backend。

## Ship It / 交付它

`outputs/skill-state-graph.md` 会在任意 target runtime 中生成 LangGraph-shaped state graph，并接好 checkpointing 和 resume。

## Exercises / 练习

1. 当 classification confidence 低于阈值时，从 `classify` 增加一条 conditional edge 到 `end`。human 手动设置 `route` 后 resume。
2. 把 SQLite-like fake 换成真实 SQLite checkpointer。测每步 serialization overhead。
3. 实现 parallel edges：两个 nodes 并发运行，由自定义 reducer merge。immutable state 在这里带来什么？
4. 阅读 `langgraph-supervisor` reference。把 toy 移植到 `create_supervisor`。比较 trace shapes。
5. 增加 streaming：每个 node 运行时 yield partial state。实时打印 deltas。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| State graph | “Agent as state machine” | typed state + nodes + edges + reducers |
| Checkpointer | “Persistence backend” | 每个 node 后序列化 state；支持 resume |
| Reducer | “State merger” | 合并当前 state 和 node update 的函数 |
| Conditional edge | “Branch” | 由 state 函数选择的 edge |
| Subgraph | “Nested graph” | 一个 graph 作为另一个 graph 中的 node |
| Durable execution | “Resume from failure” | 用精确 state 从最后成功 node 重启 |
| Supervisor | “Router LLM” | specialist subagents 的中央 dispatcher |
| Swarm | “P2P agents” | Agents 通过共享 tools hand off；没有中央 router |

## Further Reading / 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store and subagents
