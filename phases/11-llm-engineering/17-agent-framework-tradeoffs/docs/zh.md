# Agent Framework Tradeoffs — LangGraph vs CrewAI vs AutoGen vs Agno / Agent Framework 取舍：LangGraph、CrewAI、AutoGen 与 Agno

> 每个 framework 都在卖同一个 demo：research agent 自动生成报告。也都藏着同一个 bug：state schema 会和 orchestration layer 打架。你要选的是抽象形状匹配问题形状的 framework；否则剩下的都是要写两遍的 glue code。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 比较 LangGraph、CrewAI、AutoGen 与 Agno 的核心抽象，而不是只比较 demo
- 根据问题形状判断它是 graph、role play、agent chat，还是 single-agent-with-tools
- 评估 durable state、branching、observability、cost/latency 与 interoperability 对框架选择的影响
- 用 decision matrix 为一个 agent workflow 给出可解释的框架建议

## The Problem / 问题

你有一个任务，需要不止一次 LLM call。它可能是 research workflow（plan、search、summarize、cite），也可能是 code-review pipeline（parse diff、critique、patch、validate），还可能是一个多轮 assistant，负责订机票、写邮件、提交报销。你于是选了一个 framework。

三天后，你发现 framework 的 abstraction leak 了。CrewAI 给了你 roles，但当 “researcher” 需要把结构化 plan 交给 “writer” 时，它开始和你对抗。AutoGen 给了你 agents 之间的 chat，但没有一等 state，你的 checkpoint 变成了一坨 conversation log 的 pickle。LangGraph 给了你 state graph，但要求你在还不知道 agent 会做什么之前就命名每条 transition。Agno 给了你 single-agent abstraction，但当你想 fan out 到三个并发 worker 时，它开始吃力。

修复方法不是“选最好的 framework”。修复方法是让 framework 的核心抽象匹配你的问题形状。本课会画出这张地图。

## The Concept / 概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

2026 年有四个 framework 占主导。它们的核心抽象并不相同。

| Framework | Core abstraction | Best fit | Worst fit |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | Workflows with explicit state and human-in-the-loop interrupts; production agents needing time-travel debugging. | Loose, role-driven brainstorming where the topology is unknown. |
| **CrewAI** | `Crew` — roles (goal, backstory), tasks, process (sequential or hierarchical). | Role-playing or persona-driven workflows with a short linear/hierarchical plan. | Anything stateful beyond the crew's turn history; complex branching. |
| **AutoGen** | `ConversableAgent` pair — two or more agents that speak in turns until an exit condition. | Multi-agent *dialogue* (teacher-student, proposer-critic, actor-reviewer) where the thinking emerges from the chat. | Deterministic workflows with a known DAG; anything needing durable state across restarts. |
| **Agno** | `Agent` — a single LLM + tools + memory, composable into teams. | Fast-to-build single agents and lightweight teams; strong multi-modality and built-in storage drivers. | Deep, explicitly-branched graphs with custom reducers. |

### What "abstraction" actually means / “抽象”到底是什么意思

一个 framework 的核心抽象，就是你在白板上讲架构时画出来的那个东西。

- **LangGraph** → 你画一张 graph。Nodes 是 steps，edges 是 transitions，每个点上的 state object 都是 typed。Mental model 是 state machine。
- **CrewAI** → 你画组织架构图。每个 role 有 job description，manager 负责路由 tasks。Mental model 是一支小型专家团队。
- **AutoGen** → 你画 Slack DM。两个 agents 互相发消息；如果需要 moderator，就加第三个。Mental model 是 chat。
- **Agno** → 你画一个带 tools 的单个 box。把多个 box 放在一起就是 team。Mental model 是 “agent with batteries included.”

### The state question / 状态问题

State 是大多数 framework 选择在生产中崩掉的地方。

- **LangGraph.** Typed state（`TypedDict` 或 Pydantic model）、per-field reducers、一等 checkpointer（SQLite/Postgres/Redis）。Resume、interrupt 和 time-travel 都是内建能力。*(See Phase 11 · 16.)*
- **CrewAI.** State 通过 `context` 字段在 tasks 之间以 strings 流动，或者通过 `output_pydantic` 做结构化输出。默认没有 durable per-crew store；如果 crew 必须跨重启存活，你要自己接。
- **AutoGen.** State 是 chat history 和任意 user-defined `context`。Conversation transcripts 可以持久化，但 arbitrary workflow state 不会自动持久化，除非你写 adapter。
- **Agno.** 通过 `storage=` 挂到 `Agent` 上的 built-in storage drivers（SQLite、Postgres、Mongo、Redis、DynamoDB）会自动持久化 conversation sessions 和 user memories。它不是完整 graph checkpointer，而是 session store。

### The branching question / 分支问题

任何非平凡 agent 都会分支。关键是由谁决定分支。

- **LangGraph**：由你通过 conditional edges 决定。Routing 是一个带 named branches 的 Python function。Branches 是 compiled graph 的一等结构；checkpointer 会记录走过哪条 branch。
- **CrewAI**：hierarchical mode 中由 manager 决定；sequential mode 中由你在构建时决定。Routing 隐含在 task list 里；manager prompt 之外没有一等 `if`。
- **AutoGen**：agents 通过 chat 决定。Branching 由谁下一次发言涌现出来。`GroupChatManager` 选择 next speaker；你可以手写 `speaker_selection_method`，但默认是 LLM-driven。
- **Agno**：agent 通过下一步调用哪个 tool 来决定。Teams 有 coordinator/router/collaborator mode；更复杂的 branching 由开发者自己负责。

### The observability question / 可观测性问题

- **LangGraph**：通过 LangSmith 或任意 OTel exporter 接 OpenTelemetry。每个 node transition 都是 trace span；checkpoints 同时也是可 replay 的 traces。LangSmith 是一方选项，Langfuse/Phoenix 也有 adapter。
- **CrewAI**：自 2025 年末起有一等 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen**：通过 `autogen-core` 接 OpenTelemetry；AgentOps 和 Opik 有 connector。Tracing 粒度是 per-agent-message，不是 per-node。
- **Agno**：内建 `monitoring=True` flag 和 OpenTelemetry exporters；与 Langfuse 的 session traces 集成紧密。

### Cost and latency / 成本与延迟

四个 framework 都会增加 per-call overhead（framework logic、validation、serialization）。从低到高大致是：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由 framework 做了多少额外 LLM routing 决定。CrewAI 的 hierarchical manager 会花 tokens 决定下一步谁执行；AutoGen 的 `GroupChatManager` 也一样。LangGraph 只在你写 `llm.invoke` 的地方花 tokens。Agno 的 single-agent path 很薄。

当每次运行的成本很重要时，优先选择 explicit routing（LangGraph edges、AutoGen `speaker_selection_method`），而不是 LLM-selected routing。

### Interoperability / 互操作性

- **LangGraph** ↔ **LangChain** tools、retrievers、LLMs。一等 MCP adapter（把 MCP servers 作为 tools 导入）。
- **CrewAI** ↔ tools 继承自 `BaseTool`；LangChain tools、LlamaIndex tools 和 MCP tools 都可以适配进来。通过 `allow_delegation=True` 做 crew-to-crew delegation。
- **AutoGen** → `FunctionTool` 可以包装任意 Python callable；有 MCP adapter。与 AG2 ecosystem 的 agent-to-agent patterns 结合紧密。
- **Agno** → `@tool` decorator 或 BaseTool subclass；支持 MCP adapter；tools 可以在 agents 和 teams 之间共享。

## Build It / 动手构建

> 你应该能用一句话解释：为什么某个 framework 对某类 agent problem 是正确选择。

Pre-build checklist：

1. **Draw the shape / 画出形状。** 这是 graph（typed state、named transitions）？Role play（specialists hand off work）？Chat（agents talk until done）？还是 single agent with tools？
2. **Decide who branches / 决定谁来分支。** Developer-decided branching → LangGraph。Manager-agent-decided → CrewAI hierarchical。Chat-emergent → AutoGen。Tool-call-decided → Agno。
3. **Check the state budget / 检查状态预算。** 你需要 resume-from-checkpoint？Time-travel？Human interrupts mid-run？如果需要，LangGraph 是默认选择；Agno sessions 覆盖 conversation-scoped state。
4. **Check the cost budget / 检查成本预算。** LLM-selected routing 每轮都会多花 tokens。如果 agent 每天运行数千次，优先 explicit routing。
5. **Budget the framework overhead / 预算 framework overhead。** 每个 framework 都是额外依赖。如果任务只是两次 LLM call 和一个 tool，写 30 行 plain Python；没有 framework 比任何 framework 都便宜。

拒绝在能画出 graph、org chart、chat 或 agent box 之前就选 framework。拒绝选择一个会让你和它的 state model 对抗的 framework。

## Use It / 应用它

用这张 matrix 把问题形状映射到 framework，而不是用 demo 漂亮程度选型。

| Problem shape | Preferred framework | Why |
|---------------|---------------------|-----|
| Workflow DAG with typed state, human approvals, long-running | LangGraph | First-class state, checkpointer, interrupts, time-travel. |
| Research / writing pipeline with distinct roles | CrewAI (sequential) or LangGraph subgraphs | Role-per-task is cheap to express in CrewAI; scale up with LangGraph when branching gets complex. |
| Proposer-critic or teacher-student dialogue | AutoGen | Two-agent chat is its native shape. |
| Single agent with tools, sessions, memory | Agno | Thinnest setup, built-in storage and memory. |
| Thousands of parallel fanouts with reducers | LangGraph + `Send` | The only one with a first-class parallel-dispatch API. |
| Quick prototype, no framework commitment | Plain Python + provider SDK | No framework is the fastest framework. |

## Ship It / 交付它

把 framework 选择交付成一个可审查的 architecture decision，而不是口头偏好：

- 写出一句话推荐：`Because <problem shape>, choose <framework>; avoid <framework> because <mismatch>.`
- 明确 state owner：谁持久化、能否 resume、checkpoint 或 session store 的粒度是什么。
- 明确 branching owner：developer、manager LLM、chat manager，还是 tool-call loop。
- 明确成本边界：每次 run 有多少 LLM routing calls 是 framework 引入的。
- 保留 plain Python fallback：如果 workflow 只有两三个固定步骤，不要为了框架而框架。

## Exercises / 练习

1. **Easy / 简单。** 用同一个任务 “research Anthropic's headquarters, write a 200-word brief, cite sources” 分别在 LangGraph（four nodes: plan, search, write, cite）和 CrewAI（三个 roles: researcher, writer, editor）中实现。报告每次运行的 token cost 和 lines of code。
2. **Medium / 中等。** 用 AutoGen（researcher ↔ writer chat，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 与 `write_tools` 的 single agent，加 session store）实现同一任务。按三项排序四种实现：(a) cost per run，(b) crash 后 resume 能力，(c) write step 前注入 human approval 的能力。
3. **Hard / 困难。** 构建 decision-tree script `pick_framework.py`，输入一段短问题描述（JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回推荐和一句话理由。用你自己设计的六个 cases 验证。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration | “agents 如何协调” | 决定下一步由哪个 node/role/agent 执行的层。 |
| Durable state | “重启后还能恢复” | 跨进程死亡仍存在的 state，挂在 checkpoint 或 session store 上。 |
| LLM-selected routing | “让模型决定” | Planner LLM 每轮选择下一步；灵活，但每个 decision 都付 tokens。 |
| Explicit routing | “开发者决定” | 由 Python function 或 static edge 选择下一步；便宜且可审计。 |
| Crew | “CrewAI team” | Roles + tasks + process（sequential 或 hierarchical）组合成的 runnable。 |
| GroupChat | “AutoGen 的 multi-agent chat” | 由 speaker selector 管理的 N agents conversation。 |
| Team (Agno) | “Multi-agent Agno” | 对一组 agents 的 route / coordinate / collaborate mode。 |
| StateGraph | “LangGraph 的 graph” | Typed-state、node、conditional-edge、checkpointer abstraction。 |

## Further Reading / 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、teams、tools。
- [Agno documentation](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 与 framework 无关的 pattern library（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 framework 都在包装的 loop。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的 design paper。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI-style persona stacks 所依赖的 role-play foundation。
- Phase 11 · 16 (LangGraph) — 本课用来对比的 framework。
- Phase 11 · 19 (Reflexion) — 一个很适合映射到 LangGraph、但映射到 CrewAI 会别扭的 pattern。
- Phase 11 · 22 (Production observability) — 无论选择哪个 framework，都要知道如何做 instrumentation。
