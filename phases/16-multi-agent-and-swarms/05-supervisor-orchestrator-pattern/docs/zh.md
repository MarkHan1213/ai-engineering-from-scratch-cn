# Supervisor / Orchestrator-Worker Pattern / Supervisor 与 Orchestrator-Worker 模式

> 一个 lead Agent 负责计划和委派；专门化 worker 在并行上下文里执行并回报。这是 Anthropic Research 系统背后的模式（Claude Opus 4 作为 lead，Sonnet 4 作为 subagents），在内部 research eval 上相对单 Agent Opus 4 提升 +90.2%。Anthropic 工程文章还报告，BrowseComp 上 80% 的方差仅由 token usage 解释，多 Agent 胜出很大程度上是因为每个 subagent 获得了新鲜上下文窗口。本课从原语构建 supervisor pattern，并覆盖 2026 年生产部署里的工程教训。

**类型：** 学习 + 构建
**语言：** Python（stdlib, `threading`）
**前置知识：** 第 16 阶段 · 04（Primitive Model）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 supervisor / orchestrator-worker 模式为什么特别适合 research 类任务
- 用 lead plan、worker execution、lead synthesis 三段结构拆解系统
- 识别该模式的核心收益：fresh context、specialized prompt、parallelism
- 设计生产检查项：model pairing、worker timeout、token cap、observability、rainbow rollout

## The Problem / 问题

Research 是单 Agent 最容易失败的典型任务。你问：“2023 到 2026 年，多 Agent 系统发生了什么变化？”单 Agent 会顺序读五篇论文，把一半上下文塞满论文文本，然后还要把它们放在一起推理。读到第五篇时，第一篇已经被遗忘。它也无法并行。

supervisor pattern 解决这个问题：一个 lead Agent 规划搜索，把每个子问题委派给 worker，然后做综合。每个 worker 拥有自己的 200k-token window，只研究一个窄问题。lead 不看原始论文，只看 worker 摘要。

Anthropic 的生产 Research 系统报告，相比单 Agent Opus 4，在内部 research eval 上提升 +90.2%。同一篇文章还指出 BrowseComp 方差的 80% 由 *token usage alone* 解释。每个 subagent 获得新鲜上下文是主要机制。

## The Concept / 概念

### The pattern / 模式形状

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

lead 不读原始材料。worker 在 lead 综合前也不互相看对方工作。每条箭头都是一次带窄 artifact 的 handoff。

### Why it wins / 为什么有效

三个机制：

1. **Fresh context per subagent.** 探索 “FIPA-ACL heritage” 的 worker 不携带 lead 规划时消耗的 40k tokens。它拿到的是为一个问题准备的 200k 窗口。
2. **Specialization via prompt.** lead 的 prompt 是“拆解并综合”，不是“研究”。worker 的 prompt 很窄：“找出 X 发生了什么变化”。聚焦 prompt 产生聚焦输出。
3. **Parallelism.** worker 并发运行。wall-clock 约等于 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### Engineering lessons (Anthropic 2025) / 工程教训

Anthropic 文章列出几条到 2026 年仍然相关的生产教训：

- **Scale effort to query complexity.** 简单 query：一个 Agent，3-10 次工具调用。复杂 query：10+ 个 Agent。应该由 lead 估计，而不是 caller 决定。
- **Broad then narrow.** 先拆成宽子问题；如果答案值得深入，再为子问题启动更多 worker。
- **Rainbow deployments.** Agent 是长时间运行、有状态的。传统 blue-green 不适用。Anthropic 用 rainbow：新版本逐步放量，旧版本自然 drain。
- **Token usage dominates.** 多 Agent 约为单 Agent 的 15 倍 tokens。只有任务价值覆盖成本时才运行。

### The LangGraph turn / LangGraph 的转向

LangGraph 曾经提供 `langgraph-supervisor` 库和高层 `create_supervisor` helper。2025 年，LangChain 把推荐方式改成用 tool-calling 直接实现 supervisor pattern，因为 tool call 能更细地控制 *supervisor 能看见什么*（context engineering）。旧库仍可用，但文档现在推荐 tool-calling 形式。

### The failure modes / 失败模式

- **Lead hallucinates the plan.** 如果 lead 生成的子问题没有真正分解原问题，worker 会在错误目标上做精确研究。
- **Workers over-explore.** 没有明确边界时，worker 会漂出被分配的子问题，污染 synthesis。
- **Synthesis conflicts.** 两个 worker 返回矛盾事实。lead 必须重新询问（增加一轮）或显式标注分歧。静默选择一边是最糟糕失败：用户永远不知道曾经发生过分歧。

### When supervisor is wrong / 什么时候不该用 supervisor

- **Sequential tasks.** 如果第 2 步必须依赖第 1 步输出，并行没有收益。使用 pipeline（CrewAI Sequential、LangGraph linear graph）。
- **Simple queries.** 单 Agent 更快更便宜。启动 worker 前先做 lead 的 “scale effort” 检查。
- **Strict determinism.** supervisor 使用 LLM-selected delegation。当 audit/replay 比适应性更重要时，static graph 更合适。

```figure
supervisor-hierarchy
```

## Build It / 动手构建

`code/main.py` 用 `threading` 实现一个 supervisor 和三个并行 worker。lead 把 query 拆成子问题，worker 并发处理每个子问题，lead 综合。没有真实 LLM；worker 是脚本化的，用来模拟 fetch-and-summarize。

核心结构：

- `Lead.plan(query)` 把 query 拆成 3 个子问题。
- `Worker.run(sub_q)` 返回伪摘要（生产里可以是任何 tool-using Agent）。
- `Lead.run(query)` 在线程里启动 worker、join，然后综合。

运行：

```
python3 code/main.py
```

输出会展示 plan、带开始/结束时间戳的并行 worker trace，以及最终 synthesis。你能看到 wall-clock 收益：三个 0.3 秒 worker 约 0.35 秒完成，而不是 0.9 秒。

## Use It / 应用它

`outputs/skill-supervisor-designer.md` 接收用户 query，产出 supervisor-pattern 设计：lead system prompt、worker roles、子问题拆解规则和 synthesis template。构建新的 research-style Agent 系统前使用它。

## Ship It / 交付它

部署 supervisor pattern 前检查：

- **Model pairing.** Lead 使用 reasoning-tier model（Opus class、`o3` class）。Worker 使用更快更便宜的 model（Sonnet、`o4-mini`）。
- **Worker timeout.** 超过 2× median runtime 的 worker 被杀掉；lead 要么用更窄 scope 重启，要么在缺失结果的情况下继续。
- **Token cap per worker.** 硬限制（例如预期 synthesis 输入的 10×）防止 runaway worker 打爆预算。
- **Observability.** trace lead 的 plan、每个 worker 的工具调用和 synthesis。这是一切事后调试的基础。
- **Rainbow rollout.** 长时间运行、有状态的 Agent 需要渐进版本切换，而不是热替换。

## Exercises / 练习

1. 运行 `code/main.py`，然后修改 lead，让它启动 5 个 worker 而不是 3 个。观察 wall-clock 效果。在这个 demo 里，worker 数量达到多少时，启动开销超过并行收益？
2. 实现 worker timeout：杀掉任何运行超过 0.5 秒的 worker，并让 lead 综合剩余结果。你需要什么 observability 才能知道某个 worker 被截断？
3. 给 lead 的 synthesis 增加 conflict-detection：如果两个 worker 返回矛盾答案，lead 记录分歧，而不是选择一边。不调用 LLM 时如何检测矛盾？
4. 阅读 Anthropic 的 Research-system engineering post。列出这个 toy demo 要进入生产所需采用的三项实践。
5. 比较 LangGraph 的 `create_supervisor`（legacy）和新的 tool-calling 推荐方式。哪一个更能控制 supervisor 看到什么？为什么 Anthropic 明确只把 sub-answers 而不是 raw worker context 传入 synthesis？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Supervisor | “Lead agent” | 负责计划、委派和综合的 orchestrator Agent。不亲自做工作。 |
| Worker | “Subagent” | supervisor 以窄 scope 调用、拥有独立上下文窗口的聚焦 Agent。 |
| Orchestrator-worker | “Supervisor pattern” | 同一模式的另一个名字。2026 文献两个说法都用。 |
| Fresh context | “Clean window” | worker 的上下文从 system prompt 和被分配的问题开始，而不是 lead 的历史。 |
| Rainbow deployment | “Gradual rollout” | 长时间运行的有状态 Agent 需要版本化 drain-and-replace，而不是 blue-green。 |
| Token dominance | “Context is the variable” | 据 Anthropic，research eval 方差 80% 来自总 token usage，而不是 model choice。 |
| Scale effort | “Match agent count to complexity” | lead 估计 query 难度，决定启动 1 个还是 10+ worker。 |
| Synthesis conflict | “Workers disagree” | 两个 worker 返回矛盾事实；lead 必须暴露分歧，而不是静默选边。 |

## Further Reading / 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor pattern 的生产参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 当前推荐的 tool-calling supervisor 形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — legacy helper，2026 生产仍有人使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff-based supervisor 变体
