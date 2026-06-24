# Long-Running Background Agents: Durable Execution / 长时间后台 Agent：持久执行

> 生产中的 long-horizon agents 不会跑在 `while True` 里。每一次 LLM call 都会变成带 checkpoint、retry 和 replay 的 activity。Temporal 的 OpenAI Agents SDK integration 于 2026 年 3 月 GA。Claude Code Routines（Anthropic）可以在没有 persistent local process 的情况下运行 scheduled Claude Code invocations。Sessions 会在 human-input 上暂停，跨 deploy 存活，并从以 `thread_id` 为 key 的最新 checkpoint 恢复。新 ergonomics 背后是一个老模式——workflow orchestration——只不过多了一个新输入：LLM calls 是 non-deterministic activities，恢复时必须 deterministically replay。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, minimal durable-execution state machine)
**Prerequisites / 前置知识：** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 workflow、activity、event log、replay 和 checkpoint。
- 解释为什么 LLM calls 适合作为 durable execution 中的 activities。
- 用 `thread_id` 设计多 session 隔离的 checkpoint backend。
- 说明 human-input state 为什么必须是一等状态。
- 判断 durable execution 何时提升可靠性，何时只是掩盖 long-horizon degradation。

## The Problem / 问题

考虑一个运行四小时的 Agent。它调用三个 tools，向用户提问两次，并发起四十次 LLM calls。运行到一半时，宿主机重启。会发生什么？

- 在 naive `while True` loop 中：一切丢失。Run 从头开始。三个 tool calls（带真实 side effects）再次执行。用户再次被询问已经批准过的事项。四十次 LLM calls 重新计费。
- 在 durable execution 中：run 从最近 checkpoint 恢复。已经完成的 activities 不会重新执行；它们的结果从 durable log replay。用户不用重新批准已经批准过的事情。已经发出的 LLM calls 不会重新计费。

这是 workflow engines（Temporal、Cadence、Uber 的 Cherami）交付十年的同一模式。新变化是 LLM calls 现在成为一种 activity——non-deterministic、昂贵、带 side effects——而它们很自然地适配这个模式。

本课贯穿的主题是：long-horizon reliability 会衰减（METR 观察到 “35-minute degradation”：success rate 随 horizon 近似二次下降）。Durable execution 让 runs 能超过可靠性画像支持的长度；如果设计正确，这是安全失败的新方式，如果设计错误，则是危险失败的新方式。

## The Concept / 概念

### Activities, workflows, and replay / Activities、workflows 与 replay

- **Workflow**：确定性的 orchestration code。定义 activities 顺序、branches、waits。必须确定，这样才能从 event log replay 而不发生意外分歧。
- **Activity**：non-deterministic、可能失败的工作单元。LLM call、tool call、file write、HTTP request。每个 activity 都会记录 inputs，并在完成后记录 outputs。
- **Event log**：durable backing store。每个 activity start、complete、fail、retry，以及每个 workflow decision 都会被记录。
- **Replay**：恢复时，workflow code 从头重跑；所有已经完成的 activities 会返回 logged result，而不重新执行。只有尚未完成的 activities 才真正运行。

这和 React 针对 virtual DOM 重新 render、或 Git 从 commits 重建 working tree 是同一形状。Orchestrator 的 determinism 让 durability 变便宜。

### Why LLM calls fit the pattern / 为什么 LLM calls 适合这个模式

LLM calls 具有这些性质：
- Non-deterministic（temperature > 0；即使 temperature 0 也会随 model versions 漂移）。
- Expensive（金钱和 latency）。
- Potentially failing（rate limits、timeouts）。
- Side-effectful（如果它们调用 tools）。

这正是 activity profile。把每次 LLM call 都包成 activity，就能得到 exponential backoff retry、跨重启 checkpointing，以及可 replay 的 debugging trace。

### Checkpoints keyed by `thread_id` / 以 `thread_id` 为 key 的 checkpoint

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到同一 API shape：一个 `thread_id`（或等价物）标识 session；每次 state transition 持久化到 backend（PostgreSQL 默认、SQLite 用于 dev、Redis 用于 cache）；resume 读取最新 checkpoint。

Backend 选择很重要：

- **PostgreSQL**：durable、queryable、survives deploys。LangGraph 的默认选择。
- **SQLite**：只适合 local-dev；跨 hosts 会丢数据。
- **Redis**：快，但除非配置 AOF/snapshot，否则 ephemeral。
- **Cloudflare Durable Objects**：透明分布式；按 unique key scoped；可以存活数小时到数周。

### Human-input as a first-class state / 把 human-input 作为一等状态

Propose-then-commit（Lesson 15）需要一个 durable “waiting on human” state。Workflow 暂停，external queue 持有 pending request，approval 会从那个精确位置恢复。没有 durability，这只是 best-effort；有了它，隔夜 approval 到达后，workflow 会在早上继续。

### The 35-minute degradation / 35 分钟退化

METR 观察到，被测每类 Agent 在连续运行超过约 35 分钟后都会出现可靠性衰减。任务时长翻倍，failure rate 大约变成四倍。Durable execution 不会修复这个问题；它只是让你能跑得比可靠性画像支持的时间更久。安全模式是把 durability 与 re-entry 时需要 fresh HITL 的 checkpoints、以及 cap 总 compute 的 budget kill switches（Lesson 13）结合。

### When durable execution is the wrong answer / 什么时候不该用 durable execution

- 运行短于几分钟且没有 human input 的任务。Overhead > benefit。
- 严格 read-only information retrieval。
- 正确性要求在一个 context window 内端到端完成的任务（某些 reasoning tasks、某些 one-shot generation）。

```figure
memory-consolidation
```

## Build It / 动手构建

本课实现一个最小 durable-execution engine：把 activity inputs/outputs 写入 JSON event log，模拟 crash 后的 replay，并对比 naive retry 会如何重复 side effects。

## Use It / 应用它

`code/main.py` 用 stdlib Python 实现一个 minimal durable-execution engine。它支持：

- `@activity` decorator，把 inputs 和 outputs 记录到 JSON event log。
- 一个排序 activities 的 workflow function。
- 一个 `run_or_replay(workflow, event_log)` function，会 replay completed activities，而不是重新执行。

Driver 会模拟一个 three-activity workflow，在中途 crash，并展示：（a）naive retry 会重跑一切，而（b）replay 只运行缺失 activity。

## Ship It / 交付它

`outputs/skill-durable-execution-review.md` 会审查 proposed long-running agent deployment 是否具备正确 durable-execution shape：activities、determinism、checkpoint backend、human-input state，以及 HITL-on-resume policy。

## Exercises / 练习

1. 运行 `code/main.py`。观察 naive retry 与 replay 在 activity-execution count 上的差异。改变 crash point，展示 replay count 如何随之变化。

2. 把 toy engine 改成显式使用 `thread_id`。模拟两个 concurrent sessions 共享 engine，并确认它们的 event logs 不会冲突。

3. 选 toy engine 中的一个 activity。引入一个 non-determinism（在 workflow decision 内使用 wall-clock timestamp）。展示 replay 时的 divergence。解释真实 engines 如何处理它（side-effect registration、`Workflow.now()` APIs）。

4. 阅读 LangChain “Runtime behind production deep agents” 文章。列出 runtime 持久化的每个 state，并说明各自覆盖哪种 failure mode。

5. 为一个 6 小时 autonomous coding task 设计 checkpoint policy。哪里 checkpoint？Crash 后 resume 是什么样？哪些地方需要 fresh HITL？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Workflow | “Agent 的脚本” | 确定性的 orchestration code；可从 event log replay |
| Activity | “一个步骤” | Non-deterministic unit（LLM call、tool call）；执行前后都会记录 |
| Event log | “Backing store” | 每个 state transition 的 durable record |
| Replay | “恢复” | 重跑 workflow；completed activities 返回 logged results 而不重新执行 |
| Checkpoint | “保存点” | 以 thread_id 为 key 的 persisted state；resume 时 latest-wins |
| thread_id | “Session key” | 限定 durable state scope 的标识符 |
| 35-minute degradation | “可靠性衰减” | METR：success rate 随 horizon 近似二次下降 |
| Non-determinism | “Replay drift” | Wall clock、random、LLM output；必须注册为 side effect |

## Further Reading / 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — budget、turns 和 resume semantics。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent shape。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体 runtime requirements。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM calls 的 activity shape。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35-minute degradation reference。
