# Frontier Safety Frameworks — RSP, PF, FSF / Frontier Safety Frameworks：RSP、PF、FSF

> 三个 major-lab frameworks 定义了 2026 年 frontier capability 的行业治理。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入分层 AI Safety Levels（ASL-1 到 ASL-5+），类比 biosafety levels；ASL-3 于 2025 年 5 月为 CBRN-relevant models 激活。OpenAI Preparedness Framework v2（2025 年 4 月）定义 tracked capabilities 的五个 criteria，并区分 Capabilities Reports 与 Safeguards Reports。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入 Critical Capability Levels，包括新的 Harmful Manipulation CCL。三者现在都包含 competitor-adjustment clauses：如果 peer labs 没有 comparable safeguards 就发布，自己可以 defer 或降低要求。Cross-lab alignment 是结构性的，不是术语性的：“Capability Thresholds”、“High Capability thresholds” 和 “Critical Capability Levels” 指的是类似构造。

**Type / 类型：** Learn / 学习
**Languages / 语言：** none
**Prerequisites / 前置知识：** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 Anthropic 的 ASL tier structure，以及什么触发了 ASL-3。
- 说出 OpenAI Preparedness Framework v2 的 tracked capabilities 五个 criteria。
- 描述 DeepMind 的 Critical Capability Level structure 与 Harmful Manipulation CCL。
- 解释 competitor-adjustment clauses，以及它们为什么影响 race dynamics。
- 定义 safety case，并描述 three-pillar structure（monitoring、illegibility、incapability）。

## The Problem / 问题

Lessons 7-17 说明 deception 可能存在、dual-use capability 已存在、evaluation 有边界。拥有 frontier-capable model 的 lab 需要一个内部 governance structure，能够：
- 定义什么时候需要新 safeguards 的 thresholds。
- 定义 scaling 之前必需的 evaluations。
- 描述 safety case 长什么样。
- 处理 race-dynamic problem（如果竞争者无 safeguards 发布，自己怎么办）。

2025-2026 的三个 frameworks 是当前 state of the art：不完美、仍在演进，并且跨 lab 已足够对齐，以至于 governance question 现在不是 frameworks 是否存在，而是它们是否充分。

## The Concept / 概念

### Anthropic Responsible Scaling Policy v3.0 (February 2026) / Anthropic RSP v3.0（2026 年 2 月）

ASL structure：
- ASL-1：不是 frontier model（被 weaker-than-frontier baseline 覆盖）。
- ASL-2：当前 frontier baseline；使用常规 safeguards 部署。
- ASL-3：catastrophic misuse 风险显著更高；CBRN-relevant capabilities。2025 年 5 月激活。
- ASL-4：跨越 AI R&D-2 threshold；能 automate entry-level AI research 的模型。
- ASL-5+：advanced AI R&D；能显著加速 effective scaling 的模型。

v3.0 新增：
- Frontier Safety Roadmaps（公开 redacted 版本）。
- Risk Reports（季度发布，部分外部 review）。
- AI R&D 被拆成 AI R&D-2 和 AI R&D-4。
- 一旦跨越 AI R&D-4，需要 affirmative safety case，识别 pursuing misaligned goals 的模型带来的 misalignment risks。

### OpenAI Preparedness Framework v2 (April 15, 2025) / OpenAI PF v2（2025 年 4 月 15 日）

Tracked capabilities 的五个 criteria：
- **Plausible。** 存在合理 threat model。
- **Measurable。** 可以 empirical evaluation。
- **Severe。** Harm 很大。
- **Net-new。** 不是已有风险的放大。
- **Instantaneous-or-irremediable。** Harm 发生很快或无法 undo。

同时满足五条的 capabilities 会被 tracked。其他不会。

PF v2 其他结构：
- 分离 Capabilities Reports（模型能做什么）与 Safeguards Reports（有哪些 controls）。
- Safety Advisory Group review。
- Leadership approve；Board's Safety & Security Committee oversee。
- “Adjustment clause”：如果其他 lab 没有 comparable safeguards 就发布，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0 (September 2025) / DeepMind FSF v3.0（2025 年 9 月）

按 domain 定义 Critical Capability Levels（CCLs）：
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation（v3.0 新增）：能在 high-stakes contexts 中实质改变 beliefs/behavior 的模型。

v2.0（2025 年 2 月）增加 Deceptive Alignment section，并为 ML R&D CCLs 增加更高 security levels。

### Cross-lab alignment / 跨实验室对齐

- Anthropic “Capability Thresholds”。
- DeepMind “Critical Capability Levels”。
- OpenAI “High Capability thresholds”。

没有行业统一术语。结构上对齐：三层 frontier capability，并有公开 evaluation criteria。三者（2025 年以后）都包含 competitor-adjustment clauses。

### Safety cases / Safety cases

Safety case 是书面 argument：在 worst-case assumptions 下，deployment 可接受地安全。标准结构 targeting 三个 pillars：

- **Monitoring。** 如果 bad behaviour 发生，我们能否 detect？
- **Illegibility。** 模型是否缺乏执行 coherent plan 造成 harm 的能力？
- **Incapability。** 模型是否缺乏造成该 harm 的 capability？

不同 safety cases targeting 不同 pillars。对 ASL-3 CBRN case，incapability（通过 unlearning）是主要目标。对 deceptive alignment，monitoring 与 illegibility 是目标。对 cyber uplift，三者都相关。

### The race-dynamic problem / 竞赛动态问题

Competitor-adjustment clauses 很有争议。批评者认为它们制造 race to the bottom：如果三家 lab 都会在 competitor defect 时降低要求，equilibrium 会转向 defection。支持者认为 alternative（单边 safeguards）在 defecting lab 更不重视安全时会导致更糟结果。

UK AISI、US CAISI 和 EU AI Office（Lesson 24）是外部 governance counterparts。Lab frameworks 是自愿的；regulatory frameworks 正在形成。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 17-18 是 deception 与 red-team analyses 之上的 measurement-and-governance layer。Lessons 19-24 覆盖 welfare、bias、privacy、watermarking 与 regulatory structure。Lesson 28 映射 operationalize evaluations 的 research ecosystem（MATS、Redwood、Apollo、METR）。

## Build It / 动手构建

本课没有代码实现。动手工作是把三份 primary sources 读成一个 threshold map：对同一 capability，比较三家 lab 的分类、触发条件与所需 safety evidence。

## Use It / 应用它

本课没有代码。阅读三份 primary sources：RSP v3.0、PF v2、FSF v3.0。把每家 lab 的 tier structure 映射到其他两家，并识别每家定义而另外两家没有定义的一个 threshold。

## Ship It / 交付它

本课产出 `outputs/skill-framework-diff.md`。给定 safety framework 或 release note，它会把 framework 的 threshold definitions、required evaluations 和 safety-case structure 与 RSP v3.0、PF v2、FSF v3.0 比较，并标记 cross-lab gaps。

## Exercises / 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编译一个表格，列出每家 lab 的 CBRN threshold、AI R&D threshold 和 required pre-deployment evaluation。

2. Competitor-adjustment clause 出现在三份 frameworks（2025+）中。写一段支持它的论证；再写一段反对它的论证。指出每个立场依赖的 assumption。

3. 为跨越 Anthropic AI R&D-4 threshold 的模型设计 safety case。说出 three pillars（monitoring、illegibility、incapability）各自需要什么 evidence。

4. DeepMind FSF v3.0 引入 Harmful Manipulation CCL。提出三个 empirical measurements，用于指示模型跨越该 threshold。

5. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”（2025）。说出三个最强 cross-lab convergences 和两个最大 divergences。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| RSP | “Anthropic's framework” | Responsible Scaling Policy；ASL tiers；v3.0 February 2026 |
| PF | “OpenAI's framework” | Preparedness Framework；five criteria；v2 April 2025 |
| FSF | “DeepMind's framework” | Frontier Safety Framework；CCLs；v3.0 September 2025 |
| ASL-3 | “biosafety level 3-analog” | Anthropic 针对 CBRN-relevant capabilities 的 tier；2025 年 5 月激活 |
| CCL | “critical capability level” | DeepMind 的 threshold construct；按 domain 定义 |
| Safety case | “the formal argument” | 论证 deployment 在 worst-case U 下可接受安全的书面 argument |
| Adjustment clause | “competitor defection allowance” | 如果 competitors 无 comparable safeguards 发布，framework 允许降低要求的条款 |

## Further Reading / 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL tiers、roadmaps、AI R&D disaggregation。
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — five criteria、adjustment clause。
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、Harmful Manipulation。
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — cross-lab comparison。
