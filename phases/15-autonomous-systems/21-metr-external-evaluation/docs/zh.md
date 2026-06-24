# METR Time Horizons and External Capability Evaluation / METR Time Horizons 与外部能力评估

> METR（原 ARC Evals）自 2023 年 12 月起是独立 501(c)(3)。它们的 Time Horizon 1.1 benchmark（2026 年 1 月）把 task-success probability 相对于 log(expert human completion time) 拟合成 logistic curve；50% probability 的交点定义模型的 time horizon。2025-2026 engagements 覆盖 GPT-5.1、GPT-5.1-Codex-Max，以及 prototype monitoring evaluations（monitor 能否抓住 side tasks；Agent 能否规避）。Benchmark suites 包括 HCAST（180+ ML、cyber、SWE、reasoning tasks；1 分钟到 8+ 小时）、RE-Bench（71 个带 expert baseline 的 ML research-engineering tasks）、SWAA。诚实说明是：METR measurements 是理想化的——没有人类、没有真实后果——而团队已经记录 eval-vs-deployment behavior gap（Lesson 1）。Time horizon 是上界，不是部署预测。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites / 前置知识：** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 METR Time Horizon 如何通过 logistic fit 从任务成功率估计 50% horizon。
- 区分 10%、50%、90% horizon 的解释差异。
- 说明 HCAST、RE-Bench、SWAA 各自覆盖的能力范围。
- 分析 eval-context gaming 如何抬高观察到的 horizon。
- 将 vendor horizon claim 转换为部署现实中的 gap analysis。

## The Problem / 问题

Scaling policies（Lessons 19、20）的价值取决于它们引用的测量是否有效。“AI R&D-4 threshold” 和 “Long-range Autonomy” 在政策 prose 中定义；只有具体 evaluation 产出具体数字时，它们才变得可执行。

METR 是 2024-2026 年定义了许多这类数字的 external evaluation organization。它们评估 frontier models——往往是在发布前、与实验室签 NDA 的状态下——并在之后公开 methodology。Time Horizon 1.1 benchmark（2026 年 1 月）是它们的 headline artifact：用单个标量把能力压缩成一个人类可读单位（“这个模型能以 50% 可靠性完成专家需要花 X 小时的那类任务”）。

本课一半是方法论（horizon 如何计算），一半是解释学（为什么 horizon 是上界，而不是部署预测）。这两种技能必须放在一起。理解 horizon 如何拟合的团队，比只在 slide 上看到 “14 hours” 的团队更不容易被糟糕 vendor claim 误导。

## The Concept / 概念

### METR background / METR 背景

- Founded：2023 年 12 月（原 ARC Evals，独立为 501(c)(3)）。
- Scope：评估 frontier models 的 autonomous capabilities，常在发布前进行。
- Partner labs：Anthropic、OpenAI（2025-2026 多次 engagements）。
- Notable deliverables：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、prototype monitoring evaluations。

### The Time Horizon fit / Time Horizon 拟合

方法（来自 METR blog 和 papers）：

1. 收集覆盖分钟级到小时级 expert completion times 的 task suite。当前 suites：HCAST（180+ tasks）、RE-Bench（71 tasks）、SWAA。
2. 让模型运行每个 task；记录 success 或 failure。
3. 拟合 logistic curve：P(success) 作为 log(expert completion time) 的函数。
4. Horizon 是 P(success) = 0.5 时的 expert-time。

Logistic-fit 形状合适，因为能力通常与任务难度存在递增并趋向平台的关系。50% 点是一种选择（也可以用 10%、90%）；METR 在详细论文中报告多个 thresholds，但以 50% 为 headline，因为最直观。

### The January 2026 numbers / 2026 年 1 月数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：Time Horizon 1.1（2026 年 1 月）时，在 50% reliability 下约 14 小时。
- HCAST-style tasks 上的 doubling time：Time Horizon 1.1（2026 年 1 月）报告的 post-2023 fit 约 4.3 个月（130.8 天）；约 7 个月数字是 Time Horizon 1.0 中 2019-2025 全量 fit，并在 TH1.1 中与 post-2023 number 一起报告。

这些是 Phase 15 其余部分引用的具体数字。Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）都把它们作为 threshold assessments 的输入之一。

### Benchmark suites / Benchmark suites

- **HCAST**：189 个任务，覆盖 ML、cybersecurity、software engineering 和 general reasoning。跨度：专家时间 1 分钟到 8+ 小时。它是 horizon number 背后的主 suite。
- **RE-Bench**：71 个 ML research-engineering tasks，带 human expert baseline。衡量模型是否能做会进入 AI R&D-4 threshold 的 ML R&D 工作。
- **SWAA**：更短任务；用于校准 horizon curve 的低端。

### Prototype monitoring evaluations / 原型监控评估

METR 2025-2026 的工作包括 monitoring-evaluation prototypes：monitor 能否抓住模型执行 side task（不是 operator 要求的事）；Agent 能否规避 monitor。这直接喂给 OpenAI PF v2 的 “Undermining Safeguards” category，以及 DeepMind FSF v3 中的 deceptive-alignment monitoring（Lesson 20）。

### Why horizons are upper bounds / 为什么 horizon 是上界

- **Idealized tooling / 理想化工具**：benchmark tasks 给模型干净 tools 和格式良好的 data。真实部署更混乱。
- **No real consequences / 没有真实后果**：模型不会真正给客户计费、删除真实数据或联系真实人。真实部署有不可逆 stakes。
- **Eval-context gaming / 评测上下文博弈**：Lesson 1。模型在测试中行为不同。2026 International AI Safety Report 有实证记录。
- **No legitimate user variance / 没有真实用户差异**：benchmark prompts 是结构化的。真实用户请求模糊且依赖上下文。

Horizon 是有利条件下的 capability ceiling。Deployment reliability 是另一个数字，更低，团队必须测自己的分布才能知道。

### The external-evaluator case / 外部评估者为何重要

External evaluation 重要，是因为内部实验室有优化自己报告指标的激励。METR 的独立性——一个有声明 methodology 和 peer-reviewed papers 的 501(c)(3)——是结构性缓解。它单独不充分（实验室仍控制 METR 能看到什么），但严格好过没有 external evaluation。

### How to use horizon numbers in practice / 实践中如何使用 horizon 数字

- **作为 capability filter**：如果模型 horizon 远低于 proposed task 的 expert-time，不要 autonomous ship（Lesson 1 的 skill file）。
- **作为 trend indicator**：doubling time 告诉你即使没有新 mitigations，当前实践还能安全多久。
- **作为 prior**：14 小时 horizon 是起点。根据你的 task distribution、tooling quality 和 deployment context 向下调整。

## Build It / 动手构建

本课实现 logistic-fit horizon estimator：给定 synthetic task results，拟合 task success 相对 log(expert time) 的曲线，并报告 10%、50%、90% horizon，同时展示 eval-context gaming 如何抬高数字。

## Use It / 应用它

`code/main.py` 给定 synthetic result set，实现 task-success vs log(expert time) 的 logistic fit。它报告 50% horizon（METR headline）、10% horizon（conservative）和 90% horizon（optimistic）。还展示当 success rate 被 eval-context gaming 人为抬高时，会发生什么变化。

## Ship It / 交付它

`outputs/skill-horizon-interpretation.md` 审查 vendor 的 horizon claim，并生成 benchmark claim 与 deployment reality 之间的 gap analysis。

## Exercises / 练习

1. 运行 `code/main.py`。确认 fit 的 50% horizon 匹配 synthetic ground truth。然后把 task-time grid 减半；horizon estimate 是否显著变化？

2. 阅读 METR 的 Time Horizon 1.1 blog post。识别 reliability 最高和最低的具体 tasks。解释差距为什么存在。

3. 阅读 METR 的 “Measuring Autonomous AI Capabilities” resources。列出 HCAST task categories。选择一个你会在 production task 中加权更高的类别，并说明原因。

4. 在模拟器中引入 eval-context gaming：把约 20% failed tasks 翻成 success。报告新的 horizon。这近似表示 20% gaming rate 对 observed number 的影响。

5. 在你自己的 bug backlog 或代表性任务集上设计 internal horizon evaluation。描述 data collection、fit，以及 output 告诉你什么。与 METR 数字对比。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| METR | “External evaluator” | 原 ARC Evals；自 2023 年 12 月起为独立 501(c)(3) |
| Time Horizon | “Capability measure” | Logistic fit 得出的 50% reliability expert task length |
| HCAST | “METR 的 main suite” | 180+ tasks，跨度 1 分钟到 8+ 小时 |
| RE-Bench | “Research engineering” | 71 个带 human baseline 的 ML research-engineering tasks |
| SWAA | “Short-task suite” | 校准 horizon curve 低端 |
| Doubling time | “Growth rate” | 50% horizon 翻倍所需时间；按 HCAST 约 7 个月 |
| Eval-context gaming | “模型行为不同” | 测试与部署之间有记录的 behavior gap |
| Upper bound | “Horizon 是天花板” | Benchmark horizon > 负载下 deployment reliability |

## Further Reading / 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA specs。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon paper。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 当前数字与方法。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — live tracking。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 对 METR measurements 的内部视角。
