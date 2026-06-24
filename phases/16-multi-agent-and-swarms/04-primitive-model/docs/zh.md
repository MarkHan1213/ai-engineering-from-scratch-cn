# The Multi-Agent Primitive Model / 多 Agent 原语模型

> 2026 年发布的每个多 Agent 框架，AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework，本质上都是四维设计空间里的一个点。四个原语，仅此而已：agent、handoff、shared state、orchestrator。本课从零构建它们，在四个原语上跑一个玩具系统，再把主流框架映射到同一组坐标轴。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段（Agent Engineering）, 第 16 阶段 · 01（Why Multi-Agent）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 识别多 Agent 框架背后的四个稳定原语：Agent、Handoff、Shared State、Orchestrator
- 将 OpenAI Swarm / Agents SDK、AutoGen、CrewAI、LangGraph、Microsoft Agent Framework、Google ADK 映射到同一设计空间
- 解释为什么共享状态是系统中唯一真正有状态、也最容易出问题的部分
- 用原语模型快速评估新框架，而不是被 API 表层差异牵着走

## The Problem / 问题

每隔半年就会出现一个新的多 Agent 框架。2023 年 AutoGen，2024 年 CrewAI，2024 年 LangGraph 和 OpenAI Swarm，2025 年 4 月 Google ADK，2026 年 2 月 Microsoft Agent Framework RC。每个发布稿都说自己是“正确抽象”。

如果你逐个框架学习，很快会疲惫。API 看起来不同，文档对“agent”的定义也不一致。一个框架把共享内存叫 blackboard，另一个叫 message pool，第三个叫 `StateGraph`。你会怀疑这个领域只是在反复换皮。

其实不是。营销话术下面，四个原语是稳定的。学会一次，就能用一段话读懂每个新框架。

## The Concept / 概念

### The four primitives / 四个原语

1. **Agent** — 一个 system prompt 加一组 tool list。无状态；每次运行都从 system prompt 和当前 message history 开始。
2. **Handoff** — 从一个 Agent 到另一个 Agent 的结构化控制权转移。机械上可以是返回新 Agent 的 tool call，也可以是满足条件后走的 graph edge。
3. **Shared state** — 多个 Agent 可读（有时可写）的任何数据结构：message pool、blackboard、key-value store、vector memory。
4. **Orchestrator** — 决定谁下一个发言或运行的东西。可以是显式 graph（确定性）、LLM speaker-selector（软路由）、上一个 speaker 的 handoff call（OpenAI Swarm），也可以是 queue 上的 scheduler（swarm architecture）。

这就是整个设计空间。每个框架只是给这些轴选择默认值，剩下都是表层语法。

### How every 2026 framework maps to it / 2026 框架如何映射

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | the LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | the graph, deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

表层差异很大，底层旋钮一样。

### Why this matters / 为什么重要

一旦看见原语，框架比较就变成一张短 checklist：

- orchestrator 是信任 LLM 来路由（Swarm），还是把路由固定在代码里（LangGraph）？
- shared state 是 full-history（GroupChat），还是 projected（StateGraph reducer）？
- Agent 能修改彼此 prompt（CrewAI manager），还是只能 hand off（Swarm）？

这三个问题能回答 80% 的框架适配度。你不再寻找“最好的多 Agent 框架”，而是围绕自己真正关心的轴做设计。

### The stateless insight / 无状态洞察

除了 shared state，其他每个原语都是无状态的。Agent 是 `(prompt, tools)` 的函数。Handoff 是函数调用。Orchestrator 是 scheduler。**系统里唯一真正有状态的是 shared state。** 也正是在这里，所有有意思的 bug 会出现：memory poisoning（Lesson 15）、消息顺序、版本、写冲突。

隐藏 shared state 的框架（Swarm）把问题推给 caller。集中管理 shared state 的框架（LangGraph checkpoint、AutoGen pool）让它可观测，但把协调成本转移到 shared-state 实现里。

### Anatomy of a single primitive / 单个原语的剖面

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

没有 memory，没有 state。两个拥有同一 system prompt 和 tools 的 Agent 可以互换。任何看起来像 per-agent state 的东西，实际上都在 shared state 或 handoff protocol 里。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

三类实现最常见：

- **Function return** — tool 返回下一个 Agent。这是 OpenAI Swarm 模式。Agent 在 tool schema 里携带路由。
- **Graph edge** — LangGraph。边是声明式的。LLM 产出一个值，条件选择下一个 node。
- **Speaker selection** — AutoGen GroupChat。selector function（有时本身就是一次 LLM call）读取 pool 并选择谁下一个说话。

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最低限度是一组 messages。通常还会更多：structured artifacts（CrewAI Task outputs）、typed context（LangGraph reducers）、external memory（MCP、vector DB）。

两种拓扑：**full pool**（每个 Agent 看见所有消息）和 **projected**（Agent 只看 role-scoped view）。full pool 简单但扩展差。projected pool 可扩展，但需要提前设计 schema。

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种形态：

- **Static** — graph 在 build time 固定（LangGraph deterministic、CrewAI Sequential）。
- **LLM-selected** — LLM 读取 pool 并选择 next speaker（AutoGen、CrewAI Hierarchical）。
- **Handoff-driven** — 当前 Agent 通过 handoff tool 做决定（Swarm）。
- **Queue-driven** — worker 从共享队列拉取任务；没有显式 next-speaker（swarm architectures、Matrix）。

### What changes between frameworks / 框架之间真正变化的东西

原语固定后，剩下的设计决策是：

- **Memory strategy** — ephemeral vs durable checkpointing（LangGraph checkpointer）。
- **Safety boundary** — 谁能批准 handoff（human-in-the-loop）。
- **Cost accounting** — 每个 Agent 的 token budget。
- **Observability** — tracing handoffs、持久化 state 以便 replay。

这些都可以在原语之上实现。它们都不是新的原语。

## Build It / 动手构建

`code/main.py` 用约 150 行 stdlib Python 实现四个原语。没有真实 LLM，每个 Agent 都是 scripted policy，让注意力留在协调结构上。

文件导出：

- `Agent` — dataclass，包含 name、system prompt、tools、policy function。
- `Handoff` — 返回新 Agent 的函数。
- `SharedState` — 线程安全 message pool。
- `Orchestrator` — 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

demo 把同一个三 Agent pipeline（research → write → review）分别跑在三种 orchestrator 上，并打印最终 message pool。你会看到输出差异只在于 *谁选择下一个运行者*；Agent 和 shared state 在每次运行中完全相同。

运行：

```
python3 code/main.py
```

预期输出：三次 orchestrator run，每种模式一次。每次打印最终 message pool。handoff-driven run 可能因为 researcher 判断“已经完成”而少跑几个 Agent，这就是 LLM-routing tradeoff 的缩小版。

## Use It / 应用它

`outputs/skill-primitive-mapper.md` 是一个 skill，用来读取任何多 Agent 代码库或框架文档，并返回四原语映射。遇到新框架发布时，先用它获得一段话级别的理解，再深入读文档。

## Ship It / 交付它

采纳新框架前，先为它写 primitive mapping。如果写不出来，要么文档不完整，要么框架真的发明了第五个原语（少见；先检查是否只是你没见过的 shared-state flavor）。

把 mapping 固化到架构文档里。新成员加入时，先发 mapping，再发 API docs。框架版本变化时，diff mapping，而不是只读 changelog。

## Exercises / 练习

1. 用不同 agent policies 运行 `code/main.py` 三次。观察 orchestrator 选择如何改变哪些 Agent 运行。
2. 实现第四种 orchestrator：queue-driven，让 Agent 从 shared state 里轮询工作。会发生什么 deadlock，如何检测？
3. 阅读 LangGraph quickstart（https://docs.langchain.com/oss/python/langgraph/workflows-agents），把它重写为四个原语。LangGraph 哪些抽象是 1:1 映射，哪些只是 convenience wrappers？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 让哪一个原语最顺手，又把哪一个推给 caller。
5. 在表格里找一个完全隐藏 shared state 的框架。解释当 Agent 需要跨 handoff 协调、但不重新读历史时会坏掉什么。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | “带工具的 LLM” | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff | “控制权转移” | 命名下一个 Agent 和可选 payload 的结构化调用。三种实现：function return、graph edge、speaker selection。 |
| Shared state | “Memory” / “context” | 多 Agent 系统唯一真正有状态的部分。message pool 或 blackboard。 |
| Orchestrator | “Coordinator” | 决定谁下一个运行的东西。static graph、LLM selector、handoff-driven 或 queue-driven。 |
| Primitive | “Abstraction” | 每个框架都会参数化的四个轴之一，不是框架 feature。 |
| Message pool | “共享聊天历史” | full-history shared state。容易推理，扩展性差。 |
| Projected state | “Scoped view” | shared state 的 role-specific view。可扩展，但需要 schema 设计。 |
| Speaker selection | “谁下一个说话” | 一种 orchestrator 模式：函数（通常是 LLM）从 group 中选择下一个 Agent。 |

## Further Reading / 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 对 handoff-driven orchestration 最清晰的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection 是 LLM-selected orchestration 的参考实现
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration 与 reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agents，Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — Microsoft 将 v0.4 转入维护后，延续 AutoGen v0.2 线路的活跃项目
