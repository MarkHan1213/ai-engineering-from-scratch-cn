# Anthropic Responsible Scaling Policy v3.0 / Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代 2023 年政策。它采用 two-tier mitigation：Anthropic 会单方面做什么，以及被表述为 industry-wide recommendation 的内容（包括 RAND SL-4 security standards）。它把 Frontier Safety Roadmaps 和 Risk Reports 变成常设文件，而不是一次性交付物。它移除了 2023 年的 pause commitment。它引入 AI R&D-4 threshold：一旦跨过，Anthropic 必须发布 affirmative case，识别 misalignment risks 和 mitigations。Claude Opus 4.6 没有跨过该阈值。Anthropic 在 v3.0 公告中说，“confidently ruling this out is becoming difficult.” SaferAI 给 2023 RSP 打 2.2 分；v3.0 被降到 1.9，使 Anthropic 与 OpenAI、DeepMind 一样落入 “weak” RSP category。定性阈值取代了 2023 年的定量承诺；移除 pause clause 是最尖锐的退步。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, RSP threshold decision engine)
**Prerequisites / 前置知识：** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 读懂 RSP v3.0 中 Anthropic unilateral actions 与 industry-wide recommendations 的区别。
- 解释 AI R&D-4 threshold 为什么与 AAR 和 RSI 直接相关。
- 识别 Frontier Safety Roadmaps、Risk Reports 和 affirmative case 的作用。
- 分析移除 2023 pause commitment 带来的政策可信度变化。
- 使用第三方 rubric（如 SaferAI）审查 scaling policy 的具体承诺强度。

## The Problem / 问题

Frontier labs 发布的 scaling policies 同时是技术文件、治理文件，也是给监管者的信号。RSP v3.0 是 Anthropic 当前文件。仔细读它重要，不是因为它具备强制合规效力（它没有），而是因为这种 framing 会塑造实验室如何理解 catastrophic risk，以及如何向公众沟通 trade-offs。

有用的单位是 v3.0 与 v2.0 的 diff。新增了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 threshold。移除了什么：2023 pause commitment。重新 framing 了什么：把 mitigation schedule 分成 Anthropic-unilateral 和 industry-recommendation 两层。外部 review——SaferAI——把分数从 2.2（v2）降到 1.9（v3.0）。这展示了 scaling policy 如何在看起来更精致的同时变得更不严格。

## The Concept / 概念

### The two-tier mitigation schedule / 两层 mitigation schedule

- **Anthropic unilateral actions / Anthropic 单方面行动**：无论其他实验室做什么，Anthropic 承诺会做什么。包括超过阈值时停止训练、具体 security measures、具体 deployment gates。
- **Industry-wide recommendations / 行业建议**：Anthropic 认为行业应该集体做什么。包括 RAND SL-4 security standards。这些不是 Anthropic 自身的 commitments；它们是 policy advocacy。

V2 中没有这个 two-tier structure。它意味着读者必须看每个 commitment 位于哪一列。一个 security measure 如果在 “industry-wide recommendation” 列，就不是 Anthropic 的承诺，而是 Anthropic 的希望。

### The AI R&D-4 threshold / AI R&D-4 阈值

这是 RSP v3.0 命名的下一重要 capability level。具体来说：一个模型能够以有竞争力的成本自动化相当大比例的 AI research。一旦 Anthropic 认为模型跨过它，继续扩展前必须发布 affirmative case，识别 misalignment risks 和 mitigations。

按照 v3.0 公告，Claude Opus 4.6 没有跨过该阈值。文档补充说：“confidently ruling this out is becoming difficult.” 这句话重要，因为它承认阈值已经足够接近，是现实问题，而不是纯猜测上限。

Lesson 6（Automated Alignment Research）和 Lesson 7（Recursive Self-Improvement）直接喂给这个阈值。Automated alignment researchers 跨过 research-quality bars，是 AI R&D-4 threshold 正在接近的证据。

### Frontier Safety Roadmaps and Risk Reports / Frontier Safety Roadmaps 与 Risk Reports

v3.0 把两类 artifacts 提升为常设文件：

- **Frontier Safety Roadmap**：前瞻文件，描述计划中的 safety work、capability expectations 和 mitigation research。
- **Risk Report**：具体模型发布后的回顾文件，描述 observed capability 和 residual risk。

两者都是公开的。两者都按声明节奏更新。它们的作用是：读者可以追踪 Anthropic 在 Roadmap 里说会做什么，以及后来在 Risk Report 里报告了什么。

### Removing the pause clause / 移除 pause clause

2023 RSP 包含明确 pause commitment：如果模型跨过特定 capability thresholds，training 会暂停，直到 mitigations 到位。v3.0 用更软的表述替代明确暂停（发布 affirmative case，并在 mitigations adequate 时继续）。SaferAI 和其他分析者直接指出，这是新文档中最强的退步。

支持这一变化的政策论点是：2023 年的定量阈值到了 2026 年 capability benchmarks 时代变得难以抵达，因为 benchmark 本身被重新缩放。反方论点是：scaling policy 中的 pause clause 是 commitment device；移除它会移除政策可信度。

### SaferAI's downgrade / SaferAI 降级

SaferAI 是一个独立组织，会给 RSP-style documents 打分。其公开 rating：2023 Anthropic RSP 得 2.2（评分体系中 4.0 是当前最好 RSP，1.0 是 nominal）。v3.0 得 1.9。这让 Anthropic 从 “moderate” 移到 “weak”，与 OpenAI 和 DeepMind 一起处于 weak category。

SaferAI 给出的降级因素包括：
- 定性阈值取代定量阈值。
- Pause commitment 被移除。
- AI R&D-4 threshold mitigations 被描述为 “affirmative case”，而不是具体措施。
- Review mechanisms 依赖 Anthropic 的 Safety Advisory Group，独立监督有限。

### What this lesson is not / 本课不是什么

这不是合规课。RSP v3.0 不是 regulation；没有外部力量强制 Anthropic 遵守它。本课训练的是用它应得的具体性和怀疑态度阅读文档。Scaling policies 是 frontier labs 关于 catastrophic-risk posture 发出的主要公共信号。读好它们，是任何依赖 frontier capabilities 的工作者都需要的实用技能。

## Build It / 动手构建

本课把 RSP threshold-evaluation 写成一个小型 decision engine：输入 candidate model 的 capability measurements，输出是否跨过 AI R&D-4、需要哪些 affirmative-case sections，以及 deployment 是否能继续。

## Use It / 应用它

`code/main.py` 实现一个小型 decision engine，模拟 RSP threshold-evaluation 形状：给定 candidate model 和一组 capability measurements，返回是否跨过 AI R&D-4 threshold、required affirmative-case sections，以及 deployment 是否可继续。它故意很简单；重点是把文档逻辑显式化。

## Ship It / 交付它

`outputs/skill-scaling-policy-review.md` 会基于 v3.0 reference 审查一个 scaling policy（Anthropic、OpenAI、DeepMind 或内部政策）：two-tier structure、thresholds、pause commitments、independent review。

## Exercises / 练习

1. 运行 `code/main.py`。输入三个不同 capability level 的 synthetic models。确认 threshold evaluator 行为符合预期，并生成正确 affirmative-case template。

2. 完整阅读 RSP v3.0（32 页）。找出每一个位于 “industry-wide recommendation” tier 的 commitment。哪些 commitment 在 v2 中会属于 “Anthropic unilateral”？

3. 阅读 SaferAI 的 RSP grading methodology。把 rubric 应用到文档，复现 v3.0 的 1.9 分。哪个 rubric row 对降级影响最大？

4. 2023 pause commitment 被移除。提出一个 replacement commitment，在承认 2026 benchmark-rescaling problem 的同时保留政策可信度。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（Lesson 20）对比。选择一个 v3.0 更强的地方；再选择一个 Preparedness Framework 更强的地方。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| RSP | “Anthropic 的 scaling policy” | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | “Research-automation threshold” | 以有竞争力成本自动化相当大比例 AI research 的能力 |
| Affirmative case | “Safety justification” | 公开论证：risks 已识别，mitigations 足够 |
| Frontier Safety Roadmap | “Forward plan” | 关于 planned safety work 和 expected capabilities 的常设文件 |
| Risk Report | “Retrospective on a model” | 发布后关于 observed capability 和 residual risk 的常设文件 |
| Two-tier mitigation | “Unilateral vs industry” | 分离 Anthropic commitments 与 industry recommendations |
| Pause commitment | “2023 clause” | 明确承诺暂停 training；v3.0 中移除 |
| SaferAI rating | “Independent RSP grade” | 第三方 rubric；v3.0 得 1.9（v2 为 2.2） |

## Further Reading / 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整 32 页政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 与 v2 相比的变化摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 链接的常设文件。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前 frontier model 的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与 measured autonomy 连接起来。
