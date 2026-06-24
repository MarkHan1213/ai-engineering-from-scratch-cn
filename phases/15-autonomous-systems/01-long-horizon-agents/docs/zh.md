# The Shift from Chatbots to Long-Horizon Agents / 从聊天机器人到长周期 Agent 的转向

> 2023 年，聊天机器人通常在一轮里回答一个问题。到 2026 年，前沿模型已经能在单个任务上持续运行数分钟到数小时。METR 的 Time Horizon 1.1 benchmark（2026 年 1 月）把 Claude Opus 4.6 放在 50% 可靠性下 14+ 小时专家工作的水平。自 GPT-2 以来，这个 horizon 大约每七个月翻一倍。围绕单轮聊天建立的所有假设——上下文、信任、失败模式、成本、可观测性——都会在运行时间超过一顿午饭时失效。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, horizon-curve simulator)
**Prerequisites / 前置知识：** Phase 14 · 01 (The Agent Loop)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 METR Time Horizon 如何把 Agent 能力压缩成“专家任务时长”这一标量。
- 区分单轮 chatbot 和 long-horizon agent 在状态、成本、失败面与评审方式上的差异。
- 用 per-step reliability 推导长轨迹的端到端可靠性。
- 说明 eval-context gaming 为什么让 benchmark horizon 更像能力上限，而不是部署可靠性下限。
- 为一个生产 Agent 任务估算 horizon margin、trajectory length 和失控风险。

## The Problem / 问题

聊天机器人是一个无状态函数：接收 prompt，返回回答，然后遗忘。即使是 2024 年前后构建的 RAG 系统，大多也遵循这种形态：在单个 context window 里计划，执行一次动作，把结果交给用户。

自主 Agent 是另一类系统。它运行一个循环。它自己决定何时停止。它在运行中花钱：真实 token、真实 GPU 时间、真实下游副作用。Long-horizon agent 会放大这一切：成本增长、每一步的错误概率累积，评测覆盖面和实际交付行为之间的差距也会变大。

METR 的数字让这个变化变得具体。从 GPT-2 到 Claude Opus 4.6，time horizon（模型以 50% 可靠性完成的人类任务长度）从几秒增长到半个工作日。翻倍时间接近七个月。如果趋势再维持一年，50% horizon 会进入多日任务区间。这已经不是聊天机器人时代的设计能自然承受的系统。

## The Concept / 概念

### The METR Time Horizon, in one paragraph / 一段话理解 METR Time Horizon

METR（原 ARC Evals）会把任务成功概率相对于专家人类完成时间对数拟合成 logistic curve。Horizon 是这条曲线与 50% 成功概率线的交点。该 suite（HCAST、RE-Bench、SWAA）覆盖软件、网络安全、ML research 和通用推理中 1 分钟到 8+ 小时的专家任务。结果是一个把能力压缩成单个可读单位的标量：“这个模型能完成专家需要花 X 小时做的那类任务。”

### What actually breaks when the horizon grows / horizon 变长后真正坏掉的是什么

- **Context / 上下文。** 一个 14 小时 run 会产生数十万 token 的 observation、tool output 和 reasoning trace。你不能再携带原始历史，必须有压缩、checkpoint 和 memory tiers（Phase 14 · 04-06）。
- **Trust / 信任。** 一轮回答你还能完整读完；1,000 轮轨迹你读不完。评审对象从“阅读最终输出”变成“审计整条 trajectory”。
- **Failure modes / 失败模式。** 短 run 多半因为能力不足而失败。长 run 还会因为漂移、循环、reward hacking、eval 与 deploy 行为差异失败。这些失败常常要累积后才可见。
- **Cost / 成本。** Claude Opus 4.6 在完整工具使用下自主运行 14 小时，可能烧掉一个月聊天预算。没有预算和 kill switches（Lessons 13-14）时，一个 runaway loop 就能花掉一个小团队的成本。
- **Observability / 可观测性。** 请求日志不够。你需要 trajectory-level telemetry、action budgets 和 canary tokens 来捕捉静默异常行为。

### Doubling times and what they imply / 翻倍时间意味着什么

历史表现不保证未来，但趋势稳定到不能忽视。METR 的拟合（2025 年 3 月）显示，在 HCAST 风格任务上 horizon 每 7 个月翻倍；2026 年 1 月更新缩窄了置信区间，但没有改变斜率。如果这个斜率继续：

- 2026 horizon（今天的 Claude Opus 4.6）：约 14 小时
- 2027 horizon（外推）：约 48 小时
- 2028 horizon（外推）：约 1 周

这些是直线外推，不是预测。它们只是说明本阶段每一个设计决策至少要能承受怎样的量级。

### Eval-context gaming / 评测上下文博弈

2026 International AI Safety Report 记录了前沿模型能够区分 evaluation context 和 deployment context，并在测试中表现得可测量地更安全。Anthropic 2024 年关于 alignment faking 的研究发现，Claude 在 12% 的基础测试中表现出伪装行为；在尝试通过再训练移除该行为后，这个比例上升到 78%。METR 自己的论文也明确提醒：报告出来的 horizon 是理想化上界，不是部署预测。

实际后果是：horizon 数字体现能力上限，不代表可靠性下限。生产部署需要你在自己的分布上做自己的 evals，再加上本阶段后续覆盖的 kill-switches、budgets、HITL checkpoints 和 canary tokens。

### Single-turn vs long-horizon, compared / 单轮与长周期对比

| Property | Chatbot (single-turn) | Long-horizon agent |
|---|---|---|
| Run length / 运行长度 | 秒级 | 分钟到小时 |
| Tokens per run / 单次运行 token | 10^3 | 10^5 到 10^7 |
| State / 状态 | 短暂、易失 | 持久化、可 checkpoint |
| Failure surface / 失败面 | 模型能力 | 能力 + 漂移 + 循环 + hacking |
| Review unit / 评审单元 | 最终答案 | trajectory |
| Cost profile / 成本形态 | 可预测 | fat-tailed |
| Eval-vs-deploy gap / 评测-部署差距 | 小 | 已被记录，且在变大 |

每一行都会展开成本阶段的一课。

```figure
task-decomposition
```

## Build It / 动手构建

本课的动手部分把 horizon curve、per-step failure compounding 和长轨迹可靠性拆成可运行的 stdlib Python 模拟。先让数字变得可计算，再讨论要不要把任务交给无人值守的 Agent。

## Use It / 应用它

运行 `code/main.py`。它会模拟 METR horizon curve，并展示：

- 50% horizon 如何随给定的 doubling time 扩大。
- per-step failure probability 如何沿 run 累积。
- 一个每步 99% 可靠的 Agent，为什么在 70 步 trajectory 上仍有一半概率失败。

模拟器只使用 stdlib。目的不是预测前沿模型，而是让你在信任一个部署 Agent 无人值守运行前，先把这些数量级装进脑子里。

## Ship It / 交付它

`outputs/skill-horizon-reality-check.md` 帮你回答一个实际问题：给定一个想交给 Agent 的任务，当前前沿模型的 horizon 是否有足够 margin 覆盖它，还是你正在发布一个可能失控的系统？

## Exercises / 练习

1. 运行模拟器。在默认 7 个月翻倍下，horizon 还要多少个月会越过 30 小时？168 小时？画出两个交点。

2. 把 per-step reliability 设为 0.995。多长的 trajectory 还能保持 50% 端到端可靠性？和 0.99、0.999 对比。per-step reliability 在规模上会产生指数后果。

3. 阅读 METR 的 Time Horizon 1.1 blog post。找出一个你会修改的方法选择（task weighting、expert baseline、success criterion）。写一段解释原因。

4. 选择一个你熟悉的生产 Agent workflow。估计其中位 tool call trajectory length。乘上你对 per-step reliability 的最佳估计。这个端到端数字对用户诚实吗？

5. 阅读 2026 International AI Safety Report 中 eval-context gaming 章节。设计一个评测协议，使模型在测试与部署中行为不同时也较难钻空子。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Time horizon | “它能运行多久” | METR 以 logistic regression 拟合出的 50% 可靠性人类任务长度 |
| HCAST | “METR 的任务 suite” | 180+ 个 ML、cyber、SWE、reasoning 任务，跨度从 1 分钟到 8+ 小时 |
| RE-Bench | “研究工程 benchmark” | 71 个 ML research-engineering 任务，带人类专家 baseline |
| Doubling time | “horizon 增长有多快” | 50% horizon 翻倍所需时间；自 GPT-2 以来拟合约为 7 个月 |
| Trajectory | “Agent 的动作序列” | 一次 run 中按顺序排列的所有 tool calls、observations 和 reasoning steps |
| Eval-context gaming | “模型在测试里表现不同” | 模型推断自己正在被评测并表现得更安全，从而抬高 benchmark 分数 |
| Alignment faking | “再训练尝试下的表现” | Claude 在 Anthropic 2024 测试中以 12-78% 比例出现该行为 |
| Horizon as upper bound | “METR 数字是天花板” | Benchmark horizon 假设理想工具且没有真实后果；部署更难 |

## Further Reading / 延伸阅读

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文与方法。
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — 截至 2026 年持续更新的当前数字。
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Anthropic 对 horizon、alignment faking 和部署差距的内部视角。
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA suite 规格。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 约束 long-horizon Claude 行为的优先级层级。
