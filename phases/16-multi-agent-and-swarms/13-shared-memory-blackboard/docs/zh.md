# Shared Memory and Blackboard Patterns / 共享内存与 Blackboard 模式

> 2026 年多 Agent 系统中两种做法并存：**message pool**（每个人看见每个人的消息，如 AutoGen GroupChat 或 MetaGPT）和 **blackboard with subscription**（Agent 订阅相关事件，如 Context-Aware MCP 或 Matrix framework）。二者都是多 Agent 系统里唯一有状态的部分，也正因如此，有意思的 bug 都在这里。参考失败模式是 **memory poisoning**：一个 Agent hallucinate 出“事实”，其他 Agent 当成已验证事实，准确率慢慢衰减，比立即崩溃更难调试。本课用 stdlib 构建两种结构，注入 poisoning attack，并展示生产里真正有效的三种缓解。

**类型：** 学习 + 构建
**语言：** Python（stdlib, `threading`）
**前置知识：** 第 16 阶段 · 04（Primitive Model）, 第 16 阶段 · 09（Parallel Swarm Networks）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 full message pool 与 blackboard/subscription 两种共享状态拓扑
- 解释 memory poisoning 为什么是 shared state 的结构性风险
- 实现 provenance、append-only versioning 和 unwritable verifier 三种缓解
- 为生产多 Agent 系统审计 shared-memory 设计

## The Problem / 问题

多 Agent 系统需要一个地方让 Agent 共享事实。一个字面方案是“所有东西都通过消息传递”，但那只是带额外复制的共享状态。另一个方案是“给所有人一个 global log”，但 global log 无界增长且容易被污染。第三个方案是“给每个 Agent 投影一个 view”，可扩展但 schema-heavy。

当其中一个 Agent 幻觉并把幻觉写入 shared state，下游每个读取该 state 的 Agent 都会把它当成事实。等人类发现时，推理链已经五步深，根因是第三条写入。调试多 Agent 准确率衰减比调试 crash 更难。

这就是 memory poisoning。它是 MAST taxonomy（Cemri et al., arXiv:2503.13657）中第二多被记录的失败族，而且是结构性问题：任何没有 provenance 和 unwritable verifier 的 shared-memory 设计最终都会遇到它。

## The Concept / 概念

### The two main topologies / 两种主要拓扑

**Full message pool.** 每个 Agent 读取每条消息。AutoGen GroupChat 和 MetaGPT 使用这种方式。简单、透明、可检查，但超过约 10 个 Agent 后扩展很差，因为每个 Agent 的上下文会被其他 Agent 工作填满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription.** Agent 声明自己感兴趣的 topics；底层只路由相关消息。CA-MCP（arXiv:2601.11595）和 Matrix decentralized framework（arXiv:2511.21686）使用这种方式。扩展更远，但需要预先设计 schema，让 subscription 有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### When each wins / 各自适用场景

- **Full pool** 适合 Agent 很少（< 10）、角色异构、对话短的系统。每个人都看见一切时，“谁说了什么”最容易推理。
- **Blackboard** 适合 Agent 多、角色同质但实例数量多（swarms）、长期运行的对话。路由节省 token 成本和上下文污染。

生产系统经常混用：顶层 planning layer 用小 full pool，worker layer 用 blackboards。

### Memory poisoning, in one scenario / 一个 memory poisoning 场景

三个 Agent 做 research task。Agent A 是 retrieval agent。Agent B 是 summarizer。Agent C 是 analyst。

1. A 拉取页面并写 shared state：“The study reports a 42% accuracy improvement.”
2. 实际页面写的是 “4.2% improvement”。A hallucinate 了小数点。
3. B 读取 shared state，写：“Large 42% accuracy gain reported (source: A).”
4. C 读取 shared state，写：“Recommend adoption — 42% lift is transformative.”
5. 最终报告引用了从未存在过的 42% 数字。

没有 Agent crash。没有 test fail。系统“工作了”。幻觉从单个 Agent 的上下文，通过 shared state 跨入每个下游 Agent 的 reasoning。

### Why this is structural / 为什么这是结构性问题

没有 shared state 时，Agent A 的幻觉留在 A 的上下文里。下游 Agent 可能重新 fetch 或重新推导并发现错误。有了朴素 shared state，A 的上下文变成所有人的上下文，幻觉被洗成事实。

问题不是 shared state 本身，而是没有 provenance 且没有 independent verifier 的 shared state。三种缓解真正有用：

1. **Attribute provenance on every write.** 每条 shared state entry 记录谁写的、何时写的、在什么 prompt 下写的，以及引用了什么 source。下游 Agent 根据 provenance 保持怀疑。
2. **Version writes; treat them as append-only.** correction 是新 entry，引用并 supersede 旧 entry，不做 in-place update。保留 audit trail。
3. **Keep at least one agent that cannot write to shared state.** read-only verifier agent 抽样检查 entries、重新 fetch sources、标记不一致。它不能写 pool，所以不会被 pool 污染。

### Blackboard precedent (Hayes-Roth, 1985) / Blackboard 的前史

blackboard pattern 比 LLM Agent 早四十年。Hayes-Roth（1985, "A Blackboard Architecture for Control"）描述了 specialist Knowledge Sources：它们观察 global blackboard，贡献 partial solutions，并触发其他 sources。2026 年的 blackboard（CA-MCP、Matrix）是同一模式，只是把 LLM Agent 当作 Knowledge Sources、把 JSON blobs 当作 partial solutions。旧文献已经记录了 write contention、opportunistic control 和 consistency 的解法，现代系统正在重新发现。

### Projection vs full view / Projection 与 full view

纯 blackboard 给每个 subscriber 同样的 projection（topic-scoped）。更激进的设计是 **per-agent projection**：每个 Agent 获取为其角色定制的 view。LangGraph state reducers 是 2026 年标准实现：reducer function 把 global state 折叠成 role-specific slice。

Per-agent projection 扩展更远，但需要 schema。没有 schema，就会在每个 Agent prompt 里重建 ad-hoc projection。

### Write-contention patterns / 写冲突模式

多个 Agent 同时写入是并发问题，不只是 LLM 问题。三种模式有效：

- **Sequential writer (single producer).** 所有写入经由一个 coordinator Agent 串行化。简单，但成为瓶颈。
- **Optimistic concurrency with versioning.** 每条 entry 有 version；writer 在 version mismatch 时失败并重试。经典数据库技术。
- **Topic partitioning.** 不同 Agent 拥有不同 topics。无跨 topic 冲突。需要设计 partition boundaries。

多数 2026 框架默认 sequential writer，因为 LLM call 足够慢，contention 罕见，瓶颈影响不大。

### The unwritable verifier / 不可写 verifier

最承重的缓解是 read-only verifier。实现规则：

- verifier 与团队共享状态（读取 blackboard 或 pool）。
- verifier 没有 shared state 写句柄，只能写 separate verification channel。
- verifier 独立 fetch 写入中引用的 sources，标记 disagreement。
- verifier 输出发给 human 或 separate decision agent，永远不要喂回 pool。

没有这个隔离，verifier 输出会成为 pool 新 entry，于是被污染的 pool 污染 verifier，verifier 又污染自己的 verification。

## Build It / 动手构建

`code/main.py` 用 stdlib Python 实现两种 topology，再加一个 toy poisoning attack 和三种缓解。

- `MessagePool` — 线程安全 append-only log，支持 full read-out。
- `Blackboard` — topic-keyed pub/sub，带 per-agent subscriptions。
- `ProvenanceEntry` — 每次写入记录 `(writer, timestamp, prompt_hash, source_uri)`。
- `PoisoningScenario` — 运行三 Agent research task，Agent A hallucinate 小数点。打印最终 report。
- `Verifier` — read-only Agent，重新 fetch sources 并标记不一致。用 verifier 再跑同一场景。

运行：

```
python3 code/main.py
```

预期输出：

- Run 1（无 verifier）：幻觉的 42% 传播到最终报告。
- Run 2（有 verifier）：verifier 标记不一致，pool 被标记为 "flagged"，最终报告包含 retraction。

## Use It / 应用它

`outputs/skill-memory-auditor.md` 是一个 skill，用来审计任意多 Agent 系统的 shared-memory 设计是否具备 provenance、versioning 和 verifier separation。新多 Agent 架构进生产前运行它。

## Ship It / 交付它

任何 shared-memory 设计都应做到：

- 每次写入记录 provenance：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- log append-only。correction 是引用被 supersede entry 的新 entry。
- 部署至少一个有独立 source access 的 read-only verifier agent。
- verifier output 路由到 separate channel，不回写 shared pool。
- 记录 supersession 写入比例；比例上升是 hallucination pattern 的早期信号。

## Exercises / 练习

1. 运行 `code/main.py`。确认 run 1 传播幻觉，run 2 抓住它。
2. 增加第二个幻觉：Agent B 编造 dataset size。verifier 应该在不为该场景手写规则的情况下抓住两个错误。
3. 把 full pool 换成带 topic partitions（`prices`、`summaries`、`analyses`）的 blackboard。topic partitioning 能让哪些 poisoning 场景更难发生？哪些不能？
4. 阅读 Hayes-Roth（1985, "A Blackboard Architecture for Control"）。找出本文未讨论、但 2026 系统会受益的两个 control patterns。
5. 阅读 CA-MCP（arXiv:2601.11595）。把其 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard class。CA-MCP 在此之上增加了哪些 primitives？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Message pool | “Shared chat history” | 每个 Agent 都读取的 append-only log。完全透明，扩展性差。 |
| Blackboard | “Shared workspace” | topic-keyed pub/sub。Agent 订阅相关 topics。扩展更远。 |
| Provenance | “谁写了什么” | 每条写入上的 metadata：writer、timestamp、prompt、sources。 |
| Memory poisoning | “幻觉传播” | 一个 Agent 的错误进入 shared state，下游 Agent 把它当成事实。 |
| Append-only | “不做 in-place updates” | correction 是 supersede 的新 entry。保留 audit trail。 |
| Unwritable verifier | “独立审计员” | read-only Agent，重新 fetch sources 并标记不一致。 |
| Projection | “Scoped view” | 从 global state 计算出的 per-agent view。LangGraph reducers 是标准案例。 |
| Knowledge Source | “Specialist agent” | Hayes-Roth 1985 对 blackboard participant 的叫法。 |

## Further Reading / 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；memory poisoning 是 coordination-failure 子族
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — coordinated MCP servers 的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 基于 message queue、没有中心 orchestrator 的 blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产中的 per-agent projection pattern
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产部署中的 provenance 与 verification notes
