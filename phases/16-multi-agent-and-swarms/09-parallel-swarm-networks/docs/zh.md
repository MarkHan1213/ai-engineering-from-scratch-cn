# Parallel / Swarm / Networked Architectures / 并行、Swarm 与网络化架构

> 与 supervisor 相反：没有中心决策者。Agent 读取共享 event bus，异步领取工作，再把结果写回。LangGraph 明确支持 “Swarm Architecture”，用于去中心化、动态环境。Matrix（arXiv:2511.21686）把 control flow 和 data flow 都表示成经过分布式队列传递的 serialized messages，用来消除 orchestrator 瓶颈。取舍很明确：用确定性和可追踪性换可扩展性。Swarm 适合有大量独立子问题的任务；不适合需要单一连贯计划的任务。

**类型：** 学习 + 构建
**语言：** Python（stdlib, `threading`, `queue`）
**前置知识：** 第 16 阶段 · 05（Supervisor Pattern）, 第 16 阶段 · 04（Primitive Model）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 supervisor-worker 与 swarm architecture 的控制权位置
- 判断任务是否满足 swarm 需要的独立性、可并行性和弱顺序约束
- 识别 starvation、hot-spotting、back-pressure、idempotency 等生产问题
- 用共享 queue 实现 worker pool，并与 sequential / fixed assignment 对比

## The Problem / 问题

Supervisor 能扩展到少数 worker。那几百个呢？supervisor 本身会成为瓶颈：谁做什么的每个决定都要经过一个 Agent。一次缓慢 plan step 会卡住整个系统。

Swarm architecture 反转设计。不是中心 planner 分发工作，而是 worker 从共享队列中自己取工作。“协调”内置在 event bus 语义里。没有 orchestrator；系统扩展到 queue 能承受的边界。

## The Concept / 概念

### The shape / 形状

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

没有 orchestrator。每个 worker 重复：pull task，process，write result（有时再 enqueue follow-ups）。

### When swarm fits / 适合 swarm 的情况

- **Many independent tasks.** 抓取、转换、分类。任务之间不相互依赖。
- **Variable-duration work.** 有的任务 100ms，有的 10s，swarm 会自动 load balance：快 worker 继续拿下一项。supervisor 必须预估 duration。
- **Throughput over determinism.** 你关心总完成时间，而不是严格顺序。

### When swarm fails / swarm 失败的情况

- **Ordered workflows.** 如果第 3 步需要第 2 步输出，swarm 可能让第 3 步在第 2 步完成前触发。
- **Global-plan tasks.** 复杂 research question 受益于 planner。研究员 swarm 会产出独立事实，而不是连贯报告。
- **Debugging.** 没有中心日志且工作异步，复现 bug 成本高。

### Matrix (arXiv:2511.21686)

Matrix 是 2025 年把 swarm 推到自然极限的论文：control flow 和 data flow 都是分布式队列上的 serialized messages。没有中心 coordinator。容错来自 message durability。扩展性是 message broker 的问题，不是系统的。

贡献在于一个编程模型：多 Agent 协调变成“这个 Agent 订阅什么 message topic？”，而不是“supervisor 下一步选哪个 Agent？”系统看起来像 pub/sub event mesh。

### LangGraph's Swarm Architecture / LangGraph 的 Swarm Architecture

LangGraph 2025 文档明确把 “Swarm Architecture” 列为多 Agent 模式之一：Agent 是 node，但 edge 形成带 cycle 的 directed graph，任何 node 都可以从 pool 中被激活。worker 按 condition 从可用工作里选择，而不是由 supervisor 分配。

### Failure mode: starvation and hot-spotting / 失败模式：饥饿与热点

如果所有 worker 都拉取最快可用任务，长任务可能直到最后才被领取。这是经典 queue starvation。

缓解：

- 带 aging 的 priority queue（等待越久 priority 越高）。
- worker specialization：部分 worker 只接 “long” tasks。
- back-pressure：限制进入 queue 的 fast tasks 数量。

### The content-based routing link / 与 content-based routing 的关系

Swarm 很适合 content-based routing（Lesson 22）。不要用一个通用 queue，而是按 message type 建一个 queue。specialist worker 只订阅自己的类型。这是扩展到数千 Agent 的 message-bus architecture 基础。

## Build It / 动手构建

`code/main.py` 实现 4 个 worker threads，从共享 `queue.Queue` 拉取任务。任务 duration 可变，有的快、有的慢。demo 对比：

- **Sequential baseline:** 一个 worker 串行处理所有任务。
- **Fixed assignment:** 每个任务预先分给指定 worker（supervisor-style）。
- **Swarm:** worker 从共享 queue 拉取。

Swarm 自动平衡负载；fixed assignment 会让快 worker 在自己分配到的慢任务阻塞时闲置。

运行：

```
python3 code/main.py
```

输出显示每个 worker 的 task count（swarm 分布不均但整体最优）和 wall-clock time。

## Use It / 应用它

`outputs/skill-swarm-fit.md` 评估一个任务应该使用 swarm 还是 supervisor。输入：task independence、duration variance、ordering requirements、debuggability needs。

## Ship It / 交付它

Checklist：

- **Priority queue with aging.** 防止长任务饥饿。
- **Worker idempotency.** worker crash 中途可能导致任务被拉取多次。worker 必须幂等。
- **Durable queue.** 生产使用 Kafka、Redis Streams 或 database-backed queue。`queue.Queue` 只在内存中。
- **Observability per task.** 每个 task 有 trace ID；每个 worker 记录 start/end。
- **Back-pressure.** 如果 queue 增长快于 worker drain，放慢 producer。

## Exercises / 练习

1. 运行 `code/main.py`。在可变 duration workload 上，swarm 比 sequential 快多少？比 fixed assignment 快多少？
2. 增加 priority queue 版本（用 `queue.PriorityQueue`）。按任务 `importance` 字段设 priority。观察在持续负载下低 priority 任务是否会 starvation。
3. 实现 hot-spot detector：当任一 worker 处理的任务数量达到最慢 worker 的 3× 时记录日志。这说明 task-duration distribution 可能有什么问题？
4. 阅读 Matrix paper（arXiv:2511.21686）abstract 和 Section 3。识别 Matrix 接受的一项具体取舍：获得什么 scalability gain，又放弃什么（traceability、determinism）。
5. 把 swarm demo 改成 `queue.Queue` 中存 `(task_type, payload)` tuple，worker 只订阅特定类型。任务异构时，什么 routing rules 合理？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Swarm architecture | “Decentralized agents” | worker 从共享 queue 拉取；没有中心 orchestrator。 |
| Event bus | “Agents subscribe to topics” | 按类型或内容把任务路由给 worker 的 message broker。 |
| Starvation | “任务永远不运行” | 低 priority 任务因为高 priority 工作持续到来而一直拿不到执行。 |
| Hot-spotting | “一个 worker 被压垮” | 负载不均，一个 worker 拿到大多数任务。 |
| Back-pressure | “让 producer 慢下来” | queue 填满时通知上游停止生产的机制。 |
| Idempotent worker | “可安全重跑” | 任务处理两次仍产生同样结果。worker 可能 crash 中途，所以必须具备。 |
| Durable queue | “崩溃后仍在” | 由磁盘或复制存储支撑的 queue；worker crash 不会丢任务。 |
| Matrix framework | “完整 message-passing swarm” | data flow 和 control flow 都是分布式队列上的 serialized messages。 |

## Further Reading / 延伸阅读

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 显式 swarm 支持
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — 完整 message-passing swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — 为什么特定生产系统明确选择 supervisor 而不是 swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — event-driven actor rewrite，比 v0.2 GroupChat 更接近 swarm
