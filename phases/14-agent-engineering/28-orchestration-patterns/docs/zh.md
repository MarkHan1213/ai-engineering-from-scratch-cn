# Orchestration Patterns: Supervisor, Swarm, Hierarchical / 编排模式：Supervisor、Swarm、Hierarchical

> 2026 年的 frameworks 中反复出现四种 orchestration patterns：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的建议是：“It's about building the right system for your needs.” 从简单开始；只有当 single agent 加五种 workflow patterns 不够时，才引入 topology。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 12（Workflow Patterns）, 第 14 阶段 · 25（Multi-Agent Debate）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出四种反复出现的 orchestration patterns，以及各自何时适用。
- 描述 2026 年 LangChain 的推荐：tool-call-based supervision vs supervisor libraries。
- 解释 Anthropic 的 “build the right system” 规则，以及它如何约束 topology choice。
- 用 stdlib 和一个 common scripted LLM 实现全部四种模式。

## The Problem / 问题

团队常常在真正需要之前就上 “multi-agent”。跨 frameworks 反复出现四种模式；一旦你能命名它们，就能选择正确模式，或者直接跳过 topology。

## The Concept / 概念

### Supervisor-worker

- 一个 central routing LLM 分派任务给 specialist agents。
- 决策：loop back to self、hand off to specialist、terminate。
- Specialists 之间不直接交谈；所有 routing 都经过 supervisor。

Frameworks: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**2026 LangChain recommendation:** 用 direct tool calls 做 supervision，而不是 `create_supervisor`。这样能获得更细的 context engineering control — 你可以精确决定每个 specialist 看到什么。

### Swarm / peer-to-peer

- Agents 通过 shared tool surface 直接 hand off。
- 没有 central router。
- 比 supervisor 延迟更低（hops 更少）。
- 更难推理（没有单一控制点）。

Frameworks: LangGraph swarm topology, OpenAI Agents SDK handoffs（当所有 agents 都能 hand off 给所有其他 agents 时）。

### Hierarchical

- Supervisors 管 sub-supervisors，sub-supervisors 管 workers。
- 在 LangGraph 中实现为 nested subgraphs；在 CrewAI 中实现为 nested crews。
- 以 operational complexity 为代价，扩展到更大的 agent populations。

何时需要：当单个 supervisor 的 context budget 无法容纳所有 specialists 的描述时。

### Debate

- Parallel proposers + iterative cross-critique（Lesson 25）。
- 严格说不算 orchestration，更像 verification，但在 frameworks 中经常作为 topology choice 出现。

### CrewAI Crew vs Flow

CrewAI 形式化了两种部署模式：

- **Flow** 用于 deterministic event-driven automation（推荐作为生产起点）。
- **Crew** 用于 autonomous role-based collaboration。

它与上面四种模式正交，但会映射到 topology：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带 LLM router 的 supervisor。

### Anthropic's guidance / Anthropic 的建议

“Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs.”

决策顺序：

1. Single agent + workflow patterns（Lesson 12）— 从这里开始。
2. Supervisor-worker — 当你有 2-4 个 specialists。
3. Swarm — 当 latency 比 reasoning clarity 更重要。
4. Hierarchical — 只有 supervisor context budget 失败时才用。
5. Debate — 当 accuracy 比 cost 更重要。

### Where this pattern goes wrong / 这种模式容易出错的地方

- **Topology-first thinking.** 在识别 multi-agent 到底解决什么问题之前，就说 “We need multi-agent”。
- **Bouncing handoffs in swarm.** A -> B -> A -> B。使用 hop counters。
- **Fake hierarchy.** 因为 “enterprise” 搞三层结构，但实际只有两个团队。应折叠。

## Build It / 动手构建

`code/main.py` 用 stdlib 和 scripted LLM 实现全部四种模式：

- `Supervisor` — central router。
- `Swarm` — direct handoffs 的 peer-to-peer。
- `Hierarchical` — supervisors of supervisors。
- `Debate` — parallel proposers + critique。

每种模式处理同一个 three-intent task（refund / bug / sales）。Trace shapes 不同。

运行：

```
python3 code/main.py
```

输出：per-pattern trace + op count。Supervisor 最清晰；swarm 最短；hierarchical 最深；debate 最昂贵。

## Use It / 应用它

- **LangGraph** 用于 supervisor 和 hierarchical（nested subgraphs）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（supervisor-shaped）。
- **CrewAI Flow** 用于 production deterministic。
- **Custom** 用于 debate，或你需要精确控制的场景。

## Ship It / 交付它

`outputs/skill-orchestration-picker.md` 会选择 topology 并实现它。

## Exercises / 练习

1. 去掉 router，把 supervisor-worker 转成 swarm。什么坏了？什么改善了？
2. 给 swarm 增加 hop counter：3 次 handoffs 后拒绝。它能抓住 A->B->A bouncing 吗？
3. 为一个 12-specialist domain 构建 two-level hierarchical system。没有 nesting 时，context budget 在哪里失败？
4. 在 production-shaped workload 上 profile 四种模式。它们分别在哪些 metric 上胜出（latency、cost、accuracy、debuggability）？
5. 阅读 Anthropic 的 “Building Effective Agents” 文章。把你的每个 production flow 映射到四种模式之一。有没有无法干净映射的？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Supervisor-worker | “Router + specialists” | Central LLM 分派给 specialists；specialists 之间不交谈 |
| Swarm | “Peer-to-peer” | 通过 shared tools 直接 handoffs；没有 central router |
| Hierarchical | “Supervisors of supervisors” | 面向 large populations 的 nested subgraphs |
| Debate | “Proposer + critique” | Parallel proposers, cross-critique（Lesson 25） |
| Tool-call-based supervision | “Supervisor without a library” | 用 direct tool calls 实现 supervisor，以控制 context |
| Crew | “Autonomous team” | CrewAI 的 role-based collaboration mode |
| Flow | “Deterministic workflow” | CrewAI 的 event-driven production mode |

## Further Reading / 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
