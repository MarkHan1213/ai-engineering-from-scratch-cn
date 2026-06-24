# Group Chat and Speaker Selection / GroupChat 与 Speaker Selection

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个 Agent 之间共享一个 conversation；selector function（LLM、round-robin 或 custom）决定下一个 speaker。这是 emergent multi-agent conversation 的原型：Agent 不知道自己在 static graph 里的角色，只是对 shared pool 做反应。AutoGen v0.2 的 GroupChat 语义由 AG2 fork 保留；AutoGen v0.4 改写为 event-driven actor model。Microsoft 在 2026 年 2 月把 AutoGen 转入 maintenance mode，并与 Semantic Kernel 合并成 Microsoft Agent Framework（RC February 2026）。GroupChat primitive 在 AG2 和 Microsoft Agent Framework 中都保留下来，学会一次，到处可用。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 04（Primitive Model）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 GroupChat 的两个核心：shared message pool 和 speaker selector
- 比较 round-robin、LLM-selected、custom 三种 selector
- 识别 termination、context bloat、hot speaker、sycophancy cascade 等问题
- 设计可观测、可停止、带 speaker balance 的 GroupChat 系统

## The Problem / 问题

Static graph（LangGraph）适合已知 workflow。真实对话不是静态的：coder 有时问 reviewer，有时问 researcher，有时问 writer。把所有可能 handoff 都硬编码，会造成 edge explosion。你想要的是 *Agent 对 shared pool 做反应*，再由某个函数决定谁下一个说话。

AutoGen GroupChat 正是这个东西。

## The Concept / 概念

### The shape / 形状

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个 Agent 都看到所有消息。每轮都会调用 selector function，选择下一个 speaker。

### The three selector flavors / 三种 selector

**Round-robin.** 固定循环。确定性。随 N 线性扩展，但无视上下文；即使话题是 legal review，也可能轮到 coder。

**LLM-selected.** 调一次 LLM，读取最近的 pool 并返回最佳 next speaker。上下文敏感，但慢：每轮额外一次 LLM call。AutoGen 默认走这类思路。

**Custom.** 你写的 Python 函数，逻辑随意。典型做法：LLM-selected 加 fallback rules，例如“coder 后一定给 verifier 机会”。

### The ConversableAgent API / ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有 selector。当一个 Agent 完成 turn 后，manager 调 selector，selector 返回下一个 Agent。循环持续到 termination condition。

### Termination / 终止

三种常见模式：

- **Max rounds.** 总轮数硬上限。
- **"TERMINATE" token.** Agent 可以输出 sentinel message；manager 看到后停止。
- **Goal-reached check.** 每轮运行轻量 verifier，完成后停止 chat。

### The AutoGen → AG2 split and the Microsoft Agent Framework merge / AutoGen 到 AG2，再到 Microsoft Agent Framework

2025 年初，Microsoft 开始围绕 event-driven actor model 大改 AutoGen（v0.4）。社区 fork 了 AutoGen v0.2 的 GroupChat 语义，形成 AG2，保留早期采用者已经集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 转入 maintenance mode，event-driven actor model 并入 **Microsoft Agent Framework**（RC February 2026，并已与 Semantic Kernel 合并）。GroupChat 概念在两条路线中都存活；实现细节不同。AG2 是 v0.2-compatible 代码的首选 upstream。

### When GroupChat fits / 适用场景

- **Emergent conversations.** 不想预先连好每种 next-speaker。
- **Role-mixing tasks.** coder 问 researcher，researcher 问 archivist，archivist 又问 coder。流程不是 DAG。
- **Exploratory problem-solving.** 更像“头脑风暴会议”，不是“流水线”。

### When it fails / 失败场景

- **Strict determinism.** LLM selector 会不稳定。同一个 prompt，不同 run，next speaker 可能不同。
- **Sycophancy cascades.** Agent 服从最自信 speaker。需要显式 counter-prompt。
- **Context bloat.** 每个 Agent 读所有消息；10 轮后上下文已经很大。用 projection（Lesson 15）控制 view。
- **Hot speakers.** selector 过度偏好某个 Agent 专长，导致它主导对话。把 speaker balance 加进 selector feature。

### Group chat vs supervisor / GroupChat 与 supervisor

原语相同，默认值不同：

- Supervisor：一个 Agent 计划，其他 Agent 执行。selector 是“问 planner 做什么”。
- Group chat：所有 Agent 是 peers；selector 是 shared pool 上的函数。

二者都使用 Lesson 04 的四个原语。Group chat 默认采用 LLM-selected orchestration 和 full-pool shared state。

## Build It / 动手构建

`code/main.py` 从 stdlib 实现 GroupChat。三个 Agent（coder、reviewer、manager），round-robin 和 LLM-selected 两个变体，并通过 `TERMINATE` token 终止。

demo 打印 conversation transcript 和两个变体里的 selector decision trace。

运行：

```
python3 code/main.py
```

## Use It / 应用它

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat selector：round-robin vs LLM-selected vs custom，以及 selector 应使用哪些输入（recent messages、agent specialties、turn counts）。

## Ship It / 交付它

Checklist：

- **Max rounds cap.** 必须有。典型任务 10-20。
- **Speaker-balance metric.** 跟踪每个 Agent 的发言轮数；超过不均衡阈值就 alert。
- **Termination token.** `TERMINATE` 或 dedicated verifier agent。
- **Projection or scoped memory.** 约 10 条消息后，考虑只给每个 Agent scoped view，避免 context bloat。
- **Selector logging.** LLM-selected 变体必须记录 selector input 和 choice。否则调试不可能。

## Exercises / 练习

1. 运行 `code/main.py`。比较 round-robin 和 LLM-selected 下的对话。每种情况下哪个 Agent 主导？
2. 在 selector 中增加 "max-speaks-per-agent" 规则。它如何影响 transcript？
3. 实现 goal-reached termination：reviewer 返回 "approved" 时停止。它有多大概率在 round cap 前触发？
4. 阅读 AutoGen stable docs on GroupChat（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认 selector。
5. 阅读 AG2 repo（https://github.com/ag2ai/ag2），比较 v0.2 GroupChat 和 v0.4 event-driven version。v0.4 增加了哪项具体属性（throughput、fault-tolerance、composability）？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | “Agents in one chat room” | shared message pool + selector function。AutoGen / AG2 primitive。 |
| Speaker selection | “谁下一个说话” | 选择 next agent 的函数。round-robin、LLM-selected 或 custom。 |
| GroupChatManager | “The meeting host” | AutoGen 组件，拥有 selector 并循环 turn。 |
| ConversableAgent | “The base agent” | AutoGen 基类；能收发消息的 Agent。 |
| Termination token | “停止词” | 结束 chat 的 sentinel string，通常是 `TERMINATE`。 |
| Hot speaker | “一个 Agent 主导” | selector 持续选择同一 Agent 的失败模式。 |
| Context bloat | “Pool 无边界增长” | 每个 Agent 读所有历史消息；上下文随轮数增长。 |
| Projection | “Scoped view” | shared pool 的 role-specific view，用于防止 context bloat。 |

## Further Reading / 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — community AutoGen v0.2 continuation
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并后的继任者，RC February 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — event-driven actor model rewrite 细节
