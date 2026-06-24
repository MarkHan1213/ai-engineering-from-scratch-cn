# Generative Agents and Emergent Simulation / 生成式 Agent 与涌现仿真

> Park et al. 2023（UIST '23, arXiv:2304.03442）在 **Smallville** 这个 sandbox 中放入 25 个 Agent，并使用三部分架构：**memory stream**（自然语言日志）、**reflection**（Agent 对自己 memory stream 生成的高层综合）和 **plan**（日级行为，再拆成子计划）。标志性结果是 Valentine's Day party emergence：一个 Agent 被 seed 了“想办情人节派对”，没有后续脚本，邀请在群体中传播，日期被协调，派对真的发生了，而 24 个 Agent 起初都不知道这件事。ablation 显示三个组件都对 believability 必要。已记录失败包括 spatial-norm errors（进入关门商店、共享单人洗手间）。这是 2026 年 Agent simulation 和 multi-agent social evaluation 的参考架构。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 04（Primitive Model）, 第 16 阶段 · 13（Shared Memory）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 Smallville 架构中的 memory stream、reflection、plan 三个组件
- 说明 Valentine's Day party emergence 如何从局部互动产生系统级行为
- 识别 spatial norm、memory overflow、reflection hallucination 等仿真失败
- 构建一个小型生成式 Agent 仿真，并用 trace 验证涌现是否发生

## The Problem / 问题

多数多 Agent 系统是严格脚本化团队：planner 计划，coder 写代码，reviewer review。它适合定义清楚的任务，但不能捕捉 Agent 拥有 memory、priority 和开放世界后产生的非脚本化涌现行为。研究、社会仿真，以及越来越多的游戏 AI，需要第二类系统。

Smallville 是这类架构的基准。Park 2023 之前，最好的 Agent 仿真大多只是浅层脚本跟随器；之后，这个模式成为 open-world generative agents 的默认形态。如果你在 2026 年构建 Agent simulation，要么使用 Smallville 的三个组件，要么明确说明为什么不用。

## The Concept / 概念

### The three components / 三个组件

**Memory stream.** observation、action、reflection、plan 的 append-only log。每条 entry 有 timestamp、type、description（自然语言）和派生 metadata：**recency**、**importance**（Agent 自评 1-10）和 **relevance**（与当前 query 的 cosine similarity）。

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

memory retrieval 组合三个 score：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。Top-k entries 进入当前 prompt。

**Reflection.** 周期性地（每 N 条 memory 或重要事件触发），Agent 从 recent memories 生成更高层综合。reflection entry 会写回 stream，并像其他 memory 一样可检索。这是 Agent 建立“理解”的方式，也是架构里的 long-term beliefs 等价物。

**Plan.** 自顶向下拆解。先生成 day-level plan（“go to work, have dinner with Klaus”），再生成 hour-level plans，最后生成 action-level plans。plan 可修订：当 observation 与 plan 冲突时，只重规划受影响片段，而不是整个 plan。

### Why all three matter (ablation) / 为什么三个都必要

Park et al. 做了去掉 observation、reflection、plan 的 ablation。每个被移除都会伤害 believability：

- 没有 **observation**，Agent 错过上下文，按过期 belief 行动。
- 没有 **reflection**，Agent 无法形成高阶 belief；互动停留在浅层。
- 没有 **plan**，行为变成反应式噪声；目标会消散。

人类评分中，三个组件都存在时 believability 最高；去掉任意一个都有可测下降。

### The Valentine's Day emergence / 情人节派对涌现

一个 Agent，Isabella Rodriguez，被 seed 了目标：“wants to throw a Valentine's Day party at Hobbs Cafe on Feb 14 at 5pm。”其他 24 个 Agent 没有该 seed。经过模拟天数：

1. Isabella 的 plan 包含邀请别人。
2. 每个邀请成为邻居 memory stream 中的 observation。
3. 邻居 reflection 生成 belief：“Isabella is throwing a party。”
4. 邻居 plan 纳入 “attend party on Feb 14”。
5. 邻居告诉其他邻居。邀请在没有中心协调的情况下传播。
6. 2 月 14 日 5 点，若干 Agent 聚集到 Hobbs Cafe。

这是技术意义上的 emergence：系统级行为（派对）来自局部互动（双边邀请 + 个体 planning），没有中心 orchestrator。

### The documented failure modes / 已记录失败模式

Park et al. 明确记录了：

- **Spatial norm errors.** Agent 走进关门商店；尝试共用单人洗手间；在不该吃饭的房间吃饭。模型不会仅凭环境推断社会-物理规范。
- **Memory overflow.** 深度仿真会让 memory-retrieval 成本增长。实用缓解是周期性 memory compaction（summarize-and-prune）和对低 importance entry 做 decay。
- **Reflection hallucination.** reflection 可能编造 memory stream 中不存在的关系。缓解：在 reflection prompt 中包含 source memory ids，并在 retrieval 时验证。

这些都是生产相关失败：任何 2026 Agent simulation 都会继承它们。

### Three-component implementation rules / 三组件实现规则

1. **Memory is append-only.** 永远不修改 memory entry。correction 是新 entry。
2. **Importance scores are cheap.** 写入时调用 LLM 评 1-10 importance，并缓存。
3. **Retrieval is ranked, not filtered.** 按 combined score 取 top-k；不要硬过滤，否则会丢上下文。
4. **Reflection runs periodically.** 当未处理 memory 的 importance 总和超过阈值时触发（例如 150）。
5. **Plans are revisable.** 新 observation 与 plan 冲突时，只重新生成受影响片段。

### Generative agents beyond Smallville / Smallville 之后

2024-2026 后续文献扩展了该架构：

- **Multi-agent social simulation for policy / market research.** Smallville-like populations 模拟用户对 feature 的反应。比 A/B test 快；准确性仍有争议。
- **NPC AI for games.** RPG 使用 Smallville agents 产生 emergent storylines，而非脚本任务。
- **Generative-agent evaluation benchmarks.** 指标不再只是任务准确率，而是长期行为的 believability + coherence。

架构是参考点。扩展会替换组件（vector store for memory、retrieval-augmented reflection、neurosymbolic plan），但保留三部分结构。

### Why this matters for multi-agent engineering / 对多 Agent 工程的意义

Smallville 证明了：组件正确时，多 Agent emergence 可以很便宜。该架构已在开源模型上复现（小模型 believability 是平滑下降，不是断崖式）。任何需要 **emergent social behavior** 的生产系统都用这个形状。任何需要 **tight task execution** 的系统，则使用本 phase 前面讲的 supervisor / roles / primitives 模式。

## Build It / 动手构建

`code/main.py` 用 stdlib Python 和 scripted agent policies（无真实 LLM）实现三个组件。demo 以迷你形式复现 Valentine's-party emergence：

- `MemoryStream` — 带 recency/importance/relevance retrieval 的 append-only log。
- `reflect(stream)` — 对 recent high-importance memories 的脚本化 reflection。
- `plan(agent_state)` — 根据当前 beliefs 生成 day-level 和 hour-level plans。
- Scenario：5 个 Agent。Agent 1 起始目标是 "throw party at 5pm"。经过模拟 ticks，邀请传播，Agent 聚集。

运行：

```
python3 code/main.py
```

预期输出：tick-by-tick trace。最终 tick 时，5 个 Agent 中至少 3 个的 plan 包含 party，并聚集到 party location。单个 seed 在没有 orchestrator 的情况下产生协调到达。

## Use It / 应用它

`outputs/skill-simulation-designer.md` 设计 generative-agent simulation：Agent 数量、memory schema、reflection cadence、plan horizon 和 evaluation metric。

## Ship It / 交付它

生产仿真规则：

- **Memory is the database.** 扩展时选择真实 store（vector DB、Postgres）。stdlib in-memory 只适合 prototype。
- **Log the retrieval trace.** 每个 action 都记录驱动它的 top-k memories。这是你的 debug 能力。
- **Budget per-agent tokens.** 每个 Agent 每 tick 的 retrieve + reflect + plan 是 O(k) LLM calls。N agents × T ticks × calls-per-tick 会吞掉预算。
- **Compact memory periodically.** summarize-and-prune low-importance entries。retention policy 是设计决策，不是细节。
- **Detect spatial / social norm violations** explicitly。架构不会自己学会它们。

## Exercises / 练习

1. 运行 `code/main.py`。确认 3+ 个 Agent 聚集到派对。把 Agent 增加到 10，emergence 还会发生吗？
2. 移除 reflection step。行为看起来如何？把结果映射到 Park 2023 的 ablation finding。
3. 引入 competing seeded goal（“Klaus wants to give a research talk at 5pm”）。Agent 分裂，还是一个目标占优？决定因素是什么？
4. 增加 spatial constraints：Hobbs Cafe 最多容纳 4 个 Agent。simulation 能优雅处理 overflow，还是会出现 “single-person bathroom” 失败模式？
5. 阅读 Park et al.（arXiv:2304.03442）Section 6（emergent behavior experiments）。找一个你的 miniature 无法复现的行为。需要增强架构的哪个组件？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Memory stream | “Agent 的日记” | observations、actions、reflections、plans 的 append-only log。 |
| Recency | “memory 有多新” | 按 age 指数衰减的 score。 |
| Importance | “Agent 有多在意” | 写入时自评 1-10，并缓存。 |
| Relevance | “与当前 query 多相关” | cosine similarity（embedding-based）。 |
| Reflection | “高阶 belief” | 从 recent memories 生成的综合，再作为新 memory 写回。 |
| Plan | “day/hour/action decomposition” | 自顶向下 plan tree。observation 冲突时可修订。 |
| Smallville | “Park 2023 的 sandbox” | 25-Agent simulation，产生 Valentine's Day emergence。 |
| Believability | “质量指标” | 人类评分：行为是否像一个合理 Agent。 |

## Further Reading / 延伸阅读

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — 参考架构
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — 发表场地
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — Python 参考实现
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — structured-memory agents 的先例
