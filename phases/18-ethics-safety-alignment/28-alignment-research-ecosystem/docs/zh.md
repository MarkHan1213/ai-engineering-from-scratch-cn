# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR / 对齐研究生态：MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年 non-lab alignment research layer。MATS（ML Alignment & Theory Scholars）：自 2021 年末以来 527+ researchers、180+ papers、10K+ citations、h-index 47；2024 summer cohort 以 501(c)(3) 注册，约 90 scholars 和 40 mentors；pre-2025 alumni 中 80% 从事 safety/security，200+ 在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：Buck Shlegeris 创立的 applied alignment lab；提出 AI Control（Lesson 10）；与 UK AISI 合作 control safety cases。Apollo Research：为 frontier labs 做 pre-deployment scheming evaluations；作者包括 In-Context Scheming（Lesson 8）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：task-based capability evaluations、autonomous-task time-horizon studies；“Common Elements of Frontier AI Safety Policies” 对比 lab frameworks。Eleos AI Research：model-welfare pre-deployment evaluations（Lesson 19）；完成 Claude Opus 4 welfare assessment。

**Type / 类型：** Learn / 学习
**Languages / 语言：** none
**Prerequisites / 前置知识：** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 识别 non-lab alignment research ecosystem 中的五个组织及其 core output。
- 描述 MATS 的规模（scholars、papers、h-index）及其作为 talent pipeline 的作用。
- 描述 Redwood 的 AI Control agenda 与其和 UK AISI 的合作。
- 描述 METR 的 task-based evaluation methodology。

## The Problem / 问题

Frontier labs（Lesson 18）内部做 safety evaluations，并发布选定结果。Labs 之外的 ecosystem 是 evaluations 被验证、novel failure modes 首次被发现、talent 被训练的地方。理解这个生态，有助于判断哪些 research findings 被谁信任。

## The Concept / 概念

### MATS (ML Alignment & Theory Scholars) / MATS

2021 年末开始。Research mentorship program；scholars 与 senior researcher 一起在具体 alignment problem 上工作 10-12 周。

规模（2026）：
- 自 inception 以来 527+ researchers。
- 180+ papers published。
- 10K+ citations。
- h-index 47。
- 2024 summer：90 scholars + 40 mentors；注册为 501(c)(3)。

Career outcomes：pre-2025 alumni 中约 80% 在 safety/security 工作。200+ 在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research / Redwood Research

Applied alignment lab。由 Buck Shlegeris 创立。提出 AI Control agenda（Lesson 10）。与 UK AISI 合作 control safety cases。为 DeepMind 和 Anthropic 的 evaluation design 提供建议。

Canonical papers：Greenblatt, Shlegeris et al., “AI Control”（arXiv:2312.06942, ICML 2024）；Alignment Faking（Greenblatt, Denison, Wright et al., arXiv:2412.14093，与 Anthropic 合作）。

风格：specific threat models、worst-case adversaries、可以 stress-tested 的 concrete protocols。

### Apollo Research / Apollo Research

为 frontier labs 做 pre-deployment scheming evaluations。作者包括 In-Context Scheming（Lesson 8，arXiv:2412.04984）。2025 OpenAI anti-scheming training collaboration 的 partner。产出 Towards Safety Cases for AI Scheming（2024）。

风格：agentic-setting evaluations，让 deception 可以 emerge；three-pillar decomposition（misalignment、goal-directedness、situational awareness）。

### METR (Model Evaluation and Threat Research) / METR

Task-based capability evaluations。Autonomous-task completion time-horizon studies。“Common Elements of Frontier AI Safety Policies”（metr.org/common-elements，2025）对比 lab frameworks。

与 Apollo 共同参与 AI Scheming safety-case sketch。

风格：long-horizon task evaluations、empirical capability measurement、framework synthesis。

### Eleos AI Research / Eleos AI Research

Model-welfare pre-deployment evaluations。完成 Claude Opus 4 welfare assessment，并记录在 system card section 5.3 中。为 Lesson 19 的 welfare-relevant claims 提供 external methodology check。

### The flow / 流动路径

MATS 训练 researchers。Graduates 去 Anthropic、DeepMind、OpenAI（lab safety teams），或 Redwood、Apollo、METR、Eleos（external evaluation）。External evaluators 与 labs、UK AISI / CAISI 合作。Publications 再反馈到 MATS，供下一 cohort 使用。

### Why this layer matters / 为什么这一层重要

Single-source evaluations 不可靠：labs 评估自己的 models 有结构性 conflict of interest。External evaluators 可以提出和验证 lab 可能低估的 failure modes。2024 Sleeper Agents paper（Lesson 7）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。Multi-org structure 是质量控制。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 7-11 引用了 Redwood 与 Apollo 工作；Lesson 18 引用了 METR 的 framework comparison；Lesson 19 引用了 Eleos。Lesson 28 显式绘制了整个 Phase 依赖的生态组织图。

## Build It / 动手构建

本课没有代码实现。动手工作是把一个 alignment claim 或 evaluation 放回组织生态：谁做的、方法论风格是什么、是否有 counterpart 或 external validation。

## Use It / 应用它

没有代码。阅读 METR 的 “Common Elements of Frontier AI Safety Policies”，把它作为 external synthesis 如何为 lab-internal policy work 增加价值的例子。

## Ship It / 交付它

本课产出 `outputs/skill-ecosystem-map.md`。给定 alignment claim 或 evaluation，它会识别 organisation、publication venue 和 methodological style，并与 known-counterpart organisations cross-check。

## Exercises / 练习

1. 从 Lessons 7-15 中选一篇论文，识别 involved organisations。把 authors 与 MATS alumni 和当前 ecosystem affiliations cross-check。

2. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”。识别他们强调的三个 cross-lab convergences 和两个最大 divergences。

3. MATS career outcomes 是约 80% safety/security。论证这种 selection pressure 是 adaptive（训练领域）还是 biased（过滤 heterodox positions）。

4. Redwood 和 Apollo 都做 control/scheming work，但风格不同。选择一个 failure mode，描述各自会如何研究。

5. Eleos AI 是唯一纯 model-welfare organisation。设计一个聚焦不同 welfare-adjacent question（cognitive liberty、robotic embodiment 等）的 hypothetical second organisation，并阐明其 methodology。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| MATS | “the mentorship program” | ML Alignment & Theory Scholars；自 2021 年以来 527+ researchers |
| Redwood Research | “the control lab” | Applied alignment；AI Control authors；UK AISI partner |
| Apollo Research | “the scheming evals” | 为 frontier labs 做 pre-deployment scheming evaluations |
| METR | “the task-horizon evals” | Task-based capability evaluations；framework synthesis |
| Eleos AI | “the welfare lab” | Model-welfare pre-deployment evaluations |
| Talent pipeline | “MATS -> labs” | MATS graduates 流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | “non-lab check” | 非 model producer 完成的 evaluation；增加可信度 |

## Further Reading / 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — mentorship program。
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control papers。
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations。
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — framework comparison。
- [Eleos AI Research](https://www.eleosai.org/research) — model welfare methodology。
