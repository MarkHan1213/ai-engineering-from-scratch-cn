# Production Scaling — Queues, Checkpoints, Durability / 生产扩展：队列、检查点与耐久性

> 把多 Agent 系统扩展到数千个并发 run，需要 **durable execution**。LangGraph runtime 在每个 super-step 后按 `thread_id` 写 checkpoint（默认 Postgres）；worker crash 会释放 lease，另一个 worker 接手恢复。Agent 可以无限期 sleep，等待人工输入。**MegaAgent**（arXiv:2408.09955）为每个 Agent 运行 producer-consumer queue，包含三种状态（Idle / Processing / Response）和两层 coordination（intra-group chat + inter-group admin chat）。对 LLM streaming 来说，**fiber/async** 胜过 thread-per-job：线程 99% 时间都在等 token，fiber 在 I/O 上协作 yield。反方观点：Ashpreet Bedi 的 "Scaling Agentic Software" 主张直到负载证明不够前，都用 **FastAPI + Postgres + nothing else**，简单架构比想象走得更远。本课构建 durable checkpoint log、per-agent work queue with state transitions、async-vs-thread demo，并落下务实的“start simple”规则。

**类型：** 学习 + 构建
**语言：** Python（stdlib, `asyncio`, `sqlite3`）
**前置知识：** 第 16 阶段 · 09（Parallel Swarm Networks）, 第 16 阶段 · 13（Shared Memory）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 durable execution、checkpoint、lease、resume、idempotent side effects 的关系
- 比较 workflow engine、message queue + state store、actor model、FastAPI + Postgres 四类生产 substrate
- 用 SQLite checkpoint log 和 per-agent queue 模拟 crash resume
- 判断何时保持简单架构，何时采用 LangGraph / Temporal / Restate 等 durable runtime

## The Problem / 问题

一个原型多 Agent 系统在笔记本上用三个 Agent 和 in-memory event loop 跑得很好。搬到生产：

- Agent 有时运行数小时（长 research、human-in-the-loop waits）。
- worker process 会 crash。重启会丢 state。
- 峰值负载是平均负载 10 倍；需要水平扩展。
- 用户按 Agent run 付费；计费需要 exactly-once semantics。

in-memory event loop 都做不到。你需要底层 durable execution layer。2026 年典型选择：

1. 带 checkpoints 的 workflow engine（Temporal、LangGraph runtime）。
2. message queue + state store（Postgres + SQS/RabbitMQ）。
3. actor-model frameworks（MegaAgent per-agent producer-consumer）。
4. 手写 FastAPI + Postgres（Bedi 的观点）。

本课会构建每种方案的迷你版。

## The Concept / 概念

### Durable execution, the pattern / Durable execution 模式

durable-execution engine 在每个 “step”（LangGraph 叫 super-step）后持久化完整 program state。crash 时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

它要求：

- **Serializable state.** 所有 Agent state 必须可持久化。带 live database connections 的 function closures 无法存活。
- **Deterministic resume.** 给定同样 state 和 inputs，Agent 产生同样 actions（或把 LLM calls 委托给外部 deterministic oracle）。
- **Idempotent side effects.** 外部调用（tool calls、payments）必须幂等，或使用 deduplication key。

LangGraph 在每个 super-step 后写 checkpoint；Temporal 在每个 activity 后写；Restate 使用 event-sourced journals。三者实现的是同一个模式。

### LangGraph's runtime / LangGraph runtime

每个 Agent 有 `thread_id`；state 是 typed dict；每个 super-step 向 checkpoints table 写一行。resume 时，runtime 从最后 checkpoint replay，而不是从头开始。Agent 可以 `interrupt()` 等待人工输入；runtime 持久化并释放 worker。输入到达后，任意 worker 都可以 resume。

这是 2026 年 4 月的参考生产设计。

### MegaAgent's per-agent queue / MegaAgent 的 per-agent queue

arXiv:2408.09955 描述了一个规模实验：一个 cluster 中数千个 concurrent agents。架构：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层 coordination 让 group 内密集对话、group 间稀疏通信，这是数千 Agent 成本保持线性的模式。

### Async vs thread-per-job / Async 与每任务一个线程

LLM call 是 I/O-bound。等待下一个 token 的线程 99% 时间都闲着。线程每个约 1MB RAM；10,000 个并发 call，仅栈就 10GB。

fibers（Python `asyncio`、Go goroutines、Rust `tokio`）在 I/O 上协作 yield。同样 10,000 个 call 可以舒适地放进一个 process。在 LLM-agent 规模下，async 不是优化，而是架构。

例外：CPU-bound post-processing（embedding、tokenizer tricks）仍需要 threads 或 processes。把 I/O layer 和 CPU layer 分开。

### Bedi's counterpoint / Bedi 的反方观点

"Scaling Agentic Software"（Ashpreet Bedi, 2026）认为，多数团队在测到负载前过早工程化。务实默认：

- FastAPI + Postgres。
- 每个 agent run 是一行；state 用 optimistic concurrency 原地更新。
- background jobs 通过 `pg_notify` 或简单 Celery worker。
- retry policy 写在应用代码里。

在少于约 100 个 concurrent agent-runs、任务可控的负载下，这通常足够。测到失败时再升级。

规则：只有遇到简单架构解决不了的具体问题时，才采用 durable-execution frameworks。过早采用会把时间消耗在不产生收益的 ceremony 上。

### Exactly-once semantics / Exactly-once 语义

对付费 Agent runs，你需要 “exactly-once effective”（at-least-once delivery + idempotent consumer）。工程动作：

- **Dedup key per run.** 每次 side-effect call 都带它。
- **Outbox pattern.** side effects 先写表，再由独立 process 执行。两步都幂等。
- **Compensating transactions.** side effect 成功但 tracking write 失败时，安排 compensate。

这些是数据库工程模式，不是 LLM 特有。LLM tax 只是 LLM calls 很慢；其余都是标准分布式系统。

### Rainbow deployment / Rainbow 部署

Anthropic multi-agent research system 使用 “rainbow deployments”：多个 Agent runtime 版本同时运行，这样每次代码部署不需要杀掉长时间运行的 Agent。新版本 canary 一小部分流量；旧版本等自己的 Agent 结束后退役。

这是长时间运行有状态系统的标准做法；2026 年适配点是 Agent 可能存活数小时，所以 deployment cycle 必须容纳它们。

### The canonical production checklist / 标准生产 checklist

- durable state（checkpoints、snapshots，或 outbox + replayable log）。
- idempotent side effects。
- 用 async I/O layer 处理 LLM calls。
- at-least-once delivery with dedup。
- stateful workloads 使用 rainbow/canary deployment。
- observability：per-agent traces、super-step audit、retry counter。

## Build It / 动手构建

`code/main.py` 实现：

- `CheckpointStore` — SQLite-backed checkpoint log，按 thread-id key 存储。每个 super-step append 一行。
- `run_with_checkpoint(agent, thread_id)` — 模拟 mid-run crash；第二个 worker 从最后 checkpoint resume。
- `AgentQueue` — per-agent Idle / Processing / Response state machine，带小 work queue。
- `demo_async_vs_threads()` — 用 asyncio 和 threads 跑 500 个并发模拟 “LLM calls”；报告 wall-clock 和 peak memory（近似）。

运行：

```
python3 code/main.py
```

预期输出：模拟 crash 后 checkpoint resume 成功；async 版本在 < 1s 内处理 500 并发 call；thread 版本耗时数秒，并且每个并发单元内存高几个数量级。

## Use It / 应用它

`outputs/skill-scaling-advisor.md` 为 durable-execution 选型提供建议：FastAPI + Postgres、LangGraph runtime、Temporal 或 custom。依据 load、state-retention needs、deploy frequency 校准。

## Ship It / 交付它

生产加固：

- **Start simple (Bedi's rule).** FastAPI + Postgres，直到你测到它失败。
- **Instrument everything before optimizing.** per-run latency histogram、per-step time、retry count、failure categorization。
- **Outbox pattern for side effects.** 尤其是 payments 和 external API calls。
- **Rainbow deploys.** deploy 时永远不要杀 in-flight agent runs。
- **Adopt durable-execution engines (Temporal / LangGraph / Restate) when** 遇到具体问题：小时级 human-in-the-loop waits、cross-region coordination、复杂 retry/compensation policies。
- **Async for the I/O layer.** threads 只用于 CPU-bound post-processing。

## Exercises / 练习

1. 运行 `code/main.py`。确认 checkpoint resume works；测量 async vs thread concurrency 差异。
2. 实现 **outbox** table：每个 tool call 先写 outbox，再由独立 goroutine/task 执行。通过运行两次同一 tool call 验证 idempotency。
3. 模拟 **rainbow deploy**：两个 runtime versions 并发；把一半新的 thread_ids 路由到每个版本；确认 old version 上的 in-flight threads 不被中断。
4. 阅读 LangGraph runtime doc（见下方链接）。识别哪些 runtime features 最难在 hand-rolled FastAPI + Postgres 版本中复制。这是采纳理由，还是可以推迟？
5. 阅读 MegaAgent（arXiv:2408.09955）Section 3。两层 coordination（intra-group + inter-group admin chat）是显式的。画出你会如何把它映射到两个 queue families。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Durable execution | “持久化 program state” | engine 在每个 super-step 后写 state；crash recovery 是确定性的。 |
| Super-step | “Transactional boundary” | checkpoints 之间的工作单元。LangGraph term。 |
| thread_id | “Agent run identifier” | 绑定 checkpoints 和 resume logic 的 key。 |
| Idempotency | “可安全 retry” | 重复 side effect 与一次尝试产生相同结果。 |
| Outbox pattern | “解耦 side effects” | 把 intent 写表；独立 executor 执行并标记完成。 |
| At-least-once delivery | “可能重复” | message queue 语义；dedup key 让 consumer effective-once。 |
| Rainbow deploy | “多个版本并行” | 长任务期间 runtime 版本重叠运行。 |
| Async fiber | “协作式 yield” | user-mode concurrency；对 I/O-bound 负载比 threads 便宜。 |
| Checkpoint | “State snapshot” | super-step 边界处的 serialized state；resume 的关键。 |

## Further Reading / 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime design
- [MegaAgent](https://arxiv.org/abs/2408.09955) — per-agent producer-consumer queue；数千并发 Agent 的两层 coordination
- [Matrix](https://arxiv.org/abs/2511.21686) — 使用 message queues 作为 coordination substrate 的 decentralized framework
- [Temporal docs](https://docs.temporal.io/) — durable execution 的参考 workflow engine
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 包括 rainbow deployment 的生产教训
