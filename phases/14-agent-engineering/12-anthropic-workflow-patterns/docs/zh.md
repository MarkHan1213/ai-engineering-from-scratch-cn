# Anthropic's Workflow Patterns: Simple Over Complex / Anthropic Workflow Patterns：简单优先

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了 workflows（预定义路径）和 agents（动态工具使用）。五种 workflow patterns 覆盖了大多数场景。从直接 API 调用开始。只有当步骤无法预先预测时，才添加 Agent。

**Type / 类型：** Learn + Build / 学习 + 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 Anthropic 的五种 workflow patterns：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent-vs-workflow 的区别，以及二者各自的工程成本。
- 判断什么时候选择 workflow，什么时候选择 Agent。
- 基于脚本化 LLM，用 stdlib 实现全部五种 pattern。

## The Problem / 问题

团队经常为本来只需要一次 function call 的问题引入 multi-agent framework。成本是真实的：框架增加层次，遮住 prompts，隐藏控制流，并诱导过早复杂化。Schluntz 和 Zhang 2024 年 12 月的文章，是行业里最常被引用的反向提醒：从简单开始，复杂度必须挣到自己的成本。

## The Concept / 概念

### Workflows vs agents / Workflows 与 agents

- **Workflow。** LLM 和工具通过预定义代码路径编排。工程师拥有 graph。
- **Agent。** LLM 动态决定自己的工具和步骤。模型拥有 graph。

两者都有位置。Workflows 更便宜、更快、更容易 debug。Agents 解锁开放式问题，但 failure modes 更难推理。

### The augmented LLM / 增强 LLM

五种 pattern 的基础都是一个带三种能力的 LLM：search（retrieval）、tools（actions）、memory（persistence）。任何 API call 都可以使用这些能力。

### The five patterns / 五种模式

1. **Prompt chaining。** 第 1 次调用的输出是第 2 次调用的输入。适用于任务有干净线性分解的情况。步骤之间可加 programmatic gates。

2. **Routing。** classifier LLM 选择下游哪个 LLM 或工具。适用于类别差异明显的输入（tier-1 support、refund、bug、sales）。

3. **Parallelization。** 并发运行 N 次 LLM calls，再聚合结果。两种形状：sectioning（不同 chunks）和 voting（同一 prompt 跑 N 次，多数投票或 synthesis）。

4. **Orchestrator-workers。** orchestrator LLM 动态决定运行哪些 workers（也是 LLMs），再综合输出。类似 Agent loop，但 orchestrator 不会无限循环。

5. **Evaluator-optimizer。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到 evaluator 通过。这是 Self-Refine（Lesson 05）的泛化。

### Where workflows beat agents / workflows 胜过 agents 的地方

- **Predictable tasks。** 如果能枚举步骤，就应该枚举。
- **Cost-bound tasks。** Workflows 步数有界；Agents 可能螺旋上升。
- **Compliance-bound tasks。** 审计员想读 graph，而不是从 trajectories 里推断 graph。

### Where agents beat workflows / agents 胜过 workflows 的地方

- **Open-ended research。** 下一步依赖上一步返回什么。
- **Variable-length tasks。** 要运行几分钟到几小时，步数未知。
- **Novel domains。** 你还不知道正确 workflow 是什么；先探索，再固化。

### The context-engineering companion / context engineering 伴生纪律

“Effective context engineering for AI agents”（Anthropic 2025）把相邻纪律形式化：200k window 是预算，不是容器。放什么、什么时候压缩、什么时候让 context 增长。课程稍早的 Phase 14 context compression 课会详细覆盖（本课程曾编号为 Phase 14 lesson 06）。

## Build It / 动手构建

`code/main.py` 基于 `ScriptedLLM` 实现全部五种 workflow patterns：

- `prompt_chain(input, steps)`：顺序执行。
- `route(input, classifier, handlers)`：分类 + 派发。
- `parallel_vote(prompt, n, aggregator)`：N 次运行，聚合。
- `orchestrator_workers(task, workers)`：orchestrator 选择 workers。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`：循环直到通过。

运行：

```
python3 code/main.py
```

每个 pattern 都会打印自己的 trace。每种 pattern 的代码大约 10-15 行；框架成本通常以千行计。

## Use It / 应用它

- 多数任务先用直接 API calls。
- 只有当 pattern 真需要 durable state（LangGraph）、actor-model concurrency（AutoGen v0.4）或 role templating（CrewAI）时才使用框架。
- 当你想要 Claude Code harness shape，但不想重建它时，选择 Claude Agent SDK。

## Ship It / 交付它

`outputs/skill-workflow-picker.md` 会为给定任务描述选择合适 pattern，并给出决策理由，以及当 workflow 不够用时重构为 Agent 的路径。

## Exercises / 练习

1. 实现带 confidence threshold 的 routing。低于阈值 -> 升级给 human。tier-1 support 场景下阈值应该落在哪里？
2. 给 `parallel_vote` 增加 timeout。某个 call 卡住时会怎样？缺失 votes 时如何 aggregate？
3. 把 `evaluator_optimizer` 改成 bandit：跨 iterations 保留 top-2 outputs，避免后来的坏结果覆盖早先的好结果。
4. 组合 prompt chaining 和 routing：router 选择三条 chains 中的一条。测 token cost，并和一个 single big-prompt alternative 对比。
5. 选择一个你的生产 feature。画出 workflow graph。数 step。Agent 真的更好吗？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Workflow | “Predefined flow” | 工程师拥有的 LLM 与工具调用 graph |
| Agent | “Autonomous AI” | 模型拥有的 graph；动态工具决策 |
| Augmented LLM | “LLM with tools” | LLM + search + tools + memory，是原子单元 |
| Prompt chaining | “Sequential calls” | 第 N 次调用输出成为第 N+1 次调用输入 |
| Routing | “Classifier dispatch” | 选择哪条 chain / model 处理输入 |
| Parallelization | “Fan out” | N 个并发调用，按 sectioning 或 voting 聚合 |
| Orchestrator-workers | “Dispatcher agent” | Orchestrator LLM 动态选择 specialist LLMs |
| Evaluator-optimizer | “Proposer + judge” | 迭代直到 evaluator 通过；Self-Refine 泛化 |

## Further Reading / 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 五种 workflow patterns
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 伴生 discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — stateful graphs 何时值得成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — productized orchestrator-workers pattern
