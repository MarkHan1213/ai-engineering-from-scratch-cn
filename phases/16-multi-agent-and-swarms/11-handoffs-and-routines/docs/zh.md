# Handoffs and Routines — Stateless Orchestration / Handoff 与 Routine：无状态编排

> OpenAI 的 Swarm（2024 年 10 月）把多 Agent 编排压缩成两个原语：**routines**（instructions + tools 组成的 system prompt）和 **handoffs**（返回另一个 Agent 的 tool）。没有 state machine，没有 branching DSL，LLM 通过调用正确的 handoff tool 来路由。OpenAI Agents SDK（2025 年 3 月）是生产继任者。Swarm 本身仍然是最干净的概念参考，源码只有几百行。这个模式之所以传播快，是因为 API 表面大致就是 “agent = prompt + tools; handoff = function returning agent”。限制也明确：无状态，所以 memory 是 caller 的问题。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 04（Primitive Model）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 Swarm 的两个核心原语：routine 与 handoff
- 区分 handoff-driven routing 和 GroupChatManager selector
- 识别 stateless tradeoff：memory、parallelism、audit/replay 的边界
- 设计 handoff logging、context transfer、guardrail、loop detection 和 fallback agent

## The Problem / 问题

每个多 Agent 框架都想让你学习自己的 DSL：LangGraph 的 nodes 和 edges、CrewAI 的 crews 和 tasks、AutoGen 的 GroupChat 和 managers。这些 DSL 是真实抽象，但会让事情感觉比实际更重。

Swarm 走相反方向：使用模型已经具备的 tool-calling 能力。Handoff 变成 tool call。orchestrator 就是当前拿着 conversation 的 Agent。state machine 隐含在 Agent 的 system prompts 里。

## The Concept / 概念

### Two primitives / 两个原语

**Routine.** 定义 Agent 角色和可用工具的 system prompt。可以理解为作用域内指令：“你是 triage agent；如果用户问退款，handoff 给 refund agent。”

**Handoff.** Agent 可以调用的工具，返回一个新的 Agent object。Swarm runtime 检测到返回值是 Agent，就把下一轮 active agent 切过去。

这就是整个抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

triage agent 的 system prompt 让它根据用户消息选择合适 handoff。LLM 的 tool-calling 完成路由。

### Why it is viral / 为什么传播快

- **Small API.** 只学两个概念。
- **Uses what the model already does.** tool calling 已经是跨 provider 的生产级能力。
- **No state-machine burden.** 不需要描述 graph；Agent prompt 描述自己会 hand off 给谁。

### The stateless trade / 无状态取舍

Swarm 明确在 run 之间无状态。框架在一次 run 内保存 message history，但不持久化。memory、continuity、long-running tasks 都是 caller 的问题。

生产继任者（OpenAI Agents SDK，2025 年 3 月）主要补上了这些：SDK 增加 session management、guardrails 和 tracing，同时保留 handoff primitive。

### When Swarm/handoffs fit / 适合场景

- **Triage patterns.** 前线 Agent 把用户路由给 specialist。
- **Skill-based handoffs.** “任务需要代码就 call coder；需要研究就 call researcher。”
- **Short, bounded conversations.** 客服、FAQ-to-ticket、简单 workflow。

### When Swarm struggles / 困难场景

- **Long sessions with shared memory.** handoff 会把 conversation state 重置为新 Agent 的 prompt + history。没有 caller-managed memory，就没有跨 Agent 持久状态。
- **Parallel execution.** handoff 一次只能切换一个 active agent。并行需要 caller 编排多个 Swarm runs。
- **Audit and replay.** 无状态 run 很难精确 replay；LLM handoff choice 不是确定性的。

### OpenAI Agents SDK (March 2025) / OpenAI Agents SDK

生产继任者增加：

- **Session state.** 持久化 thread across runs。
- **Guardrails.** input/output validation hooks。
- **Tracing.** 每个 tool call 和 handoff 都记录日志。
- **Handoff filters.** 控制 handoff 时转移什么上下文。

handoff primitive 保留下来，生产所需的工程能力围绕它补齐。

### Swarm vs GroupChat / Swarm 与 GroupChat

二者都用 LLM-driven routing，但区别在于 **谁选择下一个**：

- GroupChat：selector（函数或 LLM）从外部选择 next speaker。
- Swarm：当前 Agent 通过调用 handoff tool 选择继任者。

Swarm 是“Agent 决定下一步”；GroupChat 是“manager 决定下一步”。Swarm 的决策落在 active agent 的 tool call 里；GroupChat 的决策落在 `GroupChatManager` 里。

## Build It / 动手构建

`code/main.py` 从零实现 Swarm：Agent dataclass、handoff mechanism（tool 返回 Agent）和一个检测 Agent switch 的 run loop。

demo：triage agent 路由到 refund、sales 或 support specialists。每个 specialist 有自己的 tools。run loop 打印每次 handoff。

运行：

```
python3 code/main.py
```

## Use It / 应用它

`outputs/skill-handoff-designer.md` 为给定任务设计 handoff topology：有哪些 Agent、它们能调用哪些 handoff、转移什么上下文。

## Ship It / 交付它

Checklist：

- **Handoff logging.** 每次 handoff 写 trace event，包含 from-agent、to-agent、context snapshot。
- **Context transfer rules.** 决定 handoff 时移动什么：full history（贵）、last N messages，还是 summary。
- **Guardrail on handoff.** handoff 到拥有不同 tool permissions 的 specialist 必须认证，否则 prompt injection 可以强迫不该发生的 handoff。
- **Loop detection.** 两个 Agent 来回 handoff 很常见；用 last-K ring check 检测。
- **Fallback agent.** handoff target 不存在时，回到安全默认 Agent。

## Exercises / 练习

1. 运行 `code/main.py`，triage 到 refund agent。确认第二轮 active agent 是 refund。
2. 增加 loop-detection rule：如果同两个 Agent 连续 handoff 3 次，强制退出。设计 fallback。
3. 阅读 OpenAI Agents SDK docs on handoff filters。实现一个 “summarize-on-handoff” 版本：outgoing agent 在 incoming agent 接手前把上下文压缩成 bullet summary。
4. 比较 Swarm handoff 和 GroupChatManager selector。哪种模式让 prompt injection 更糟，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 做出的一个显式设计决策，OpenAI Agents SDK 改变了还是保留了它。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Routine | “The agent prompt” | system prompt + tool list。定义角色和可用 handoffs。 |
| Handoff | “Transfer to another agent” | active agent 可调用的 tool，返回新 Agent。runtime 切换 active agent。 |
| Stateless | “No memory between runs” | Swarm 不持久化任何东西；memory 是 caller 责任。 |
| Active agent | “Who's speaking now” | 当前持有 conversation 的 Agent。handoff 会改变它。 |
| Context transfer | “What moves on handoff” | incoming agent 能看到哪些 history：full、last N 或 summary。 |
| Handoff loop | “Agents ping-pong” | 两个 Agent 反复互相 handoff 的失败模式。 |
| OpenAI Agents SDK | “Production Swarm” | 2025 年 3 月继任者；在 handoff primitive 上增加 sessions、guardrails、tracing。 |
| Handoff filter | “Gate on transfer” | SDK 特性：在 handoff 边界检查和修改上下文。 |

## Further Reading / 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，作为概念参考保留
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带 sessions 和 tracing 的生产继任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagents 如何通过 `Task` 使用类似 handoff 的模式
