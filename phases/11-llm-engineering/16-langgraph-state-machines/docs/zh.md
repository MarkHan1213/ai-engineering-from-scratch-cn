# LangGraph — State Machines for Agents / LangGraph：Agent 的状态机

> 手写的 ReAct loop 是一个 `while True`。用 LangGraph 写的 ReAct loop 是一张可以 checkpoint、interrupt、branch 和 time-travel 的图。Agent 本身没有变，变化的是包住它的 harness。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 理解为什么生产 Agent loop 本质上是带状态、边和副作用的 state machine
- 使用 LangGraph `StateGraph` 声明 typed state、nodes、conditional edges 与 reducers
- 为 ReAct agent 添加 checkpointer、human-in-the-loop interrupt、streaming 和 time-travel debugging
- 识别 `add_messages` reducer、生产 checkpointer 与副作用前 interrupt 的上线要求

## The Problem / 问题

你发布了一个 function-calling agent。前三轮都正常，然后出事了：模型尝试了一个返回 500 的 tool，用户在任务中途改了主意，或者 agent 想在没有人批准的情况下给订单退款。`while True:` loop 没有 hook。你不能暂停它，不能 rewind，也不能分支出一个“如果模型当时选了另一个 tool 会怎样”的调试路径。只要你把它从 demo 推到生产，agent 就变成了一个黑盒：要么成功，要么失败。

一旦你看清这一点，下一步就很直接：agent 本来就是状态机。它由 system prompt、message history、pending tool calls 和 next action 组成。把状态机显式化：用 nodes 表示“模型思考”“工具运行”“人类批准”，用 edges 表示它们之间的条件转移。图一旦显式，harness 就免费获得四件事：checkpointing（step 之间保存 state）、interrupts（暂停等待人类）、streaming（流式输出 token 和中间事件）、time-travel（回到之前的 state，尝试另一个 branch）。

LangGraph 就是提供这层抽象的库。它不是 LangChain 意义上的 agent framework（“这里有个 AgentExecutor，祝你好运”）。它是一个图运行时，拥有一等 state、一等 persistence 和一等 interrupts。Agent loop 是你画出来的，不是你手写出来的。

## The Concept / 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三件事。

1. **State / 状态。** 一个 typed dict（TypedDict 或 Pydantic model）在图中流动。每个 node 接收完整 state，返回 partial update；LangGraph 用每个字段对应的 *reducer* 合并它们，例如 list 用 `operator.add` 累积，默认是 overwrite。
2. **Nodes / 节点。** Python functions，形状是 `state -> partial_state`。每个 node 是一个离散步骤：“调用模型”“运行工具”“做摘要”。
3. **Edges / 边。** 节点之间的转移。Static edges 只去一个地方。Conditional edges 接收 router function `state -> next_node_name`，让图可以根据模型输出分支。

你会 compile 这张图。Compile 会绑定 topology，挂载 checkpointer（可选，但生产必需），并返回一个 runnable。你用 initial state 和 `thread_id` 调用它。执行的每一步都会把 checkpoint 持久化到 `(thread_id, checkpoint_id)`。

### The four superpowers / 四个超能力

**Checkpointing / 检查点。** 每次 node transition 都把新 state 写到 store（测试可用 in-memory，生产可用 Postgres/Redis/SQLite）。用同一个 `thread_id` 再次调用图即可 resume。图会从暂停位置继续。

**Interrupts / 中断。** 给 node 标记 `interrupt_before=["human_review"]`，执行会在该 node 运行前停止。State 已经持久化。你的 API 可以回复用户“awaiting approval”。之后对同一 `thread_id` 携带 `Command(resume=...)` 的请求会恢复执行。

**Streaming / 流式输出。** `graph.stream(state, mode="updates")` 会在 state delta 发生时 yield。`mode="messages"` 会流式输出 model node 内部的 LLM tokens。`mode="values"` yield 完整 snapshot。你决定 UI 展示哪种粒度。

**Time-travel / 时间旅行。** `graph.get_state_history(thread_id)` 返回完整 checkpoint log。把任意旧的 `checkpoint_id` 传给 `graph.invoke`，就能从那个点 fork。它非常适合调试（“如果模型当时选了 tool B 呢？”），也适合 replay 生产 trace 做回归测试。

### Reducers are the point / Reducer 才是关键

每个 state field 都有 reducer。大多数默认行为没问题：新值覆盖旧值。但 message list 需要 `operator.add`，让新 message 追加而不是替换。Parallel edges 也通过 reducer 合并 update。如果两个 nodes 都更新 `messages`，而你忘了 `Annotated[list, add_messages]`，第二个会静默获胜，你会丢掉半轮对话。Reducer 是这个库唯一微妙的点；把它做对，其它东西就能组合。

### The ReAct graph in four nodes / 四个节点的 ReAct 图

生产 ReAct agent 只有四个 node/edge 关系：

1. `agent`：用当前 message history 调用 LLM。返回 assistant message（可能包含 tool_calls）。
2. `tools`：执行最后一条 assistant message 中的 tool_calls，把 tool results 作为 tool messages 追加回去。
3. 从 `agent` 出发的 conditional edge：如果最后一条 message 有 tool_calls，就路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的 static edge。

就是这样。你用大约 40 行代码，拿到了完整 ReAct loop（Thought → Action → Observation → Thought → …），还带 checkpointing、interrupts 和 streaming。

### StateGraph vs Send (fanout) / StateGraph 与 Send（fanout）

`Send(node_name, state)` 允许一个 node 并行派发 subgraphs。例如 agent 决定同时查询三个 retriever。每个 `Send` 都会启动 target node 的一次并行执行；它们的输出通过 state reducer 合并。LangGraph 就是用这种方式表达 orchestrator-workers pattern，而不需要你手写线程原语。

### Subgraphs / 子图

一个 compiled graph 可以作为另一个 graph 的 node。外层 graph 只看到一个 node；内层 graph 有自己的 state 和 checkpoints。团队通常用它构建 supervisor-worker agents：supervisor graph 把用户意图路由到各个 domain worker subgraph。

## Build It / 动手构建

### Step 1: state and nodes / 第 1 步：state 与 nodes

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让 message list 累积而不是覆盖的 reducer。忘记它是最常见的 LangGraph bug。

### Step 2: run with a thread / 第 2 步：带 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个 update 都是 `{node_name: state_delta}` 形态的 dict。你的 frontend 可以把这些事件流式传给 UI，让用户看到“agent is thinking… calling search_web… got result… answering.”

### Step 3: add a human-in-the-loop interrupt / 第 3 步：加入 human-in-the-loop interrupt

标记一个 node，让执行在它运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

State、checkpoint 和 thread 都会跨 interrupt 持久存在。除了执行瞬间，没有东西只存在内存里。

### Step 4: time-travel for debugging / 第 4 步：用 time-travel 调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

把 `None` 作为 input 传入，会从给定 checkpoint replay；传入一个 value，则会先把它作为 update 追加到该 checkpoint 的 state，再继续执行。这就是你复现一个坏 agent run、同时又不重跑整段 conversation 的方式。

### Step 5: swap the checkpointer for production / 第 5 步：替换成生产 checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已支持。`MemorySaver` 适合测试。任何需要跨进程重启持久化的东西，都应该使用真正的 store。

## Use It / 应用它

> 你应该把 Agent 构建成图，而不是构建成 `while True` loop。

在使用 LangGraph 前，先做一个 60 秒设计：

1. **Name the nodes / 命名节点。** 每个离散决策或有副作用的动作都是一个 node。“Agent thinks”“tool runs”“reviewer approves”“response streams”。如果你列不出来，任务还没有 agent-shaped。
2. **Declare the state / 声明状态。** 最小 TypedDict，并为每个 list field 配 reducer。不要把一切都塞进 `messages`；把任务字段（working `plan`、`budget` counter、`retrieved_docs` list）提升到 top level。
3. **Draw the edges / 画边。** 默认 static，除非下一步取决于模型输出。每个 conditional edge 都需要一个带 named branches 的 router function。
4. **Choose a checkpointer up front / 一开始就选择 checkpointer。** 测试用 `MemorySaver`，其它场景用 Postgres/Redis/SQLite。不要在没有 checkpointer 的情况下上线，因为那意味着不能 resume、不能 interrupt、不能 time-travel。
5. **Decide interrupts before tools run, not after / interrupt 放在 tool 执行前，而不是执行后。** Approval 应该在进入 side-effecting node 的边上，这样你可以在造成影响前取消；validation 应该在离开 model 的边上，这样可以低成本拒绝坏调用。
6. **Stream by default / 默认流式输出。** UI 用 `mode="updates"`，model node 内的 token-level streaming 用 `mode="messages"`，eval 期间的完整 snapshot 用 `mode="values"`。

拒绝上线没有 checkpointer 的 LangGraph agent。拒绝上线 side effect 之后才 interrupt 的 agent。拒绝上线没有用 `add_messages` 作为 reducer 的 `messages` field。

## Ship It / 交付它

生产交付前，至少满足这几个 gate：

- 每个 side-effecting node 都有进入前的 approval 或 validation path。
- `messages` 这类累积字段都声明了正确 reducer。
- Checkpointer 使用可跨进程重启的 store，而不是 `MemorySaver`。
- UI 至少展示 `mode="updates"` 的节点级进度，方便用户判断 agent 是否卡住。
- 回归测试能从一条真实 trace 的旧 checkpoint fork，并复现/修正错误分支。

## Exercises / 练习

1. **Easy / 简单。** 用 calculator tool 和 web-search tool 实现上面的 four-node ReAct graph。验证对于 two-turn conversation，`list(app.get_state_history(config))` 至少返回四个 checkpoints。
2. **Medium / 中等。** 增加一个 `planner` node，让它在 `agent` 前运行，并向 state 写入结构化 `plan: list[str]`。让 `agent` 标记 plan steps done。如果 checkpoint resume 后 `plan` 丢失（reducer 错误），测试必须失败。
3. **Hard / 困难。** 构建一个 supervisor graph，使用 `Send` 在三个 subgraphs（`researcher`、`writer`、`reviewer`）之间路由。每个 subgraph 都有自己的 state 和 checkpointer。在外层 graph 上添加 `interrupt_before=["writer"]`，让人类批准 research brief。确认从旧 checkpoint time-travel 时，只重跑 forked branch。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | “LangGraph graph” | Compile 前用于添加 nodes 和 edges 的 builder object。 |
| Reducer | “字段如何合并” | 当 node 返回某字段 update 时执行的 `(old, new) -> merged` 函数；默认 overwrite，`add_messages` 会 append。 |
| Thread | “Conversation ID” | 一个 `thread_id` 字符串，用来限定一个 session 的所有 checkpoints。 |
| Checkpoint | “Paused state” | Node transition 之后持久化的完整 graph state snapshot，以 `(thread_id, checkpoint_id)` 为 key。 |
| Interrupt | “暂停等人” | `interrupt_before` / `interrupt_after` 在 node 边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | “从旧步骤 fork” | `graph.invoke(None, config_with_old_checkpoint_id)` 会从该 checkpoint 向前 replay。 |
| Send | “Parallel subgraph dispatch” | Node 可以返回的 constructor，用于启动 N 个 target node 的并行执行。 |
| Subgraph | “Compiled graph as a node” | 把 compiled StateGraph 作为另一个 graph 的 node 使用；保留自己的 state scope。 |

## Further Reading / 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、reducers、checkpointers 和 interrupts 的主参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的 mental model，来自官方文档。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis stores、checkpoint namespaces 与 thread IDs 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和 edit-state pattern。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph agent 实现的 pattern；读它理解 reasoning trace 的动机。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 什么时候选择 chain、router、orchestrator-workers、evaluator-optimizer 等 graph shape。
- Phase 11 · 09 (Function Calling) — 每个 LangGraph agent node 复用的 tool-call primitive。
- Phase 11 · 14 (Model Context Protocol) — 外部 tool discovery 可以通过 MCP adapter 接入 LangGraph `ToolNode`。
- Phase 11 · 17 (Agent framework tradeoffs) — 什么时候选择 LangGraph 而不是 CrewAI、AutoGen 或 Agno。
