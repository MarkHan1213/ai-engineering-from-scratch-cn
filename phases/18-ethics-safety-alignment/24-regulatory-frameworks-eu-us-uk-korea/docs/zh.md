# Regulatory Frameworks — EU, US, UK, Korea / 监管框架：欧盟、美国、英国、韩国

> 四个主要监管 regime 定义了 2026 年 AI governance landscape。EU AI Act（2024 年 8 月 1 日生效）——prohibited practices 与 AI literacy 从 2025 年 2 月 2 日适用；GPAI obligations 从 2025 年 8 月 2 日适用；full applicability 与 Article 50 transparency 于 2026 年 8 月 2 日适用；legacy GPAI 与 embedded high-risk systems 于 2027 年 8 月 2 日适用；罚款最高 15M EUR 或全球 turnover 的 3%。GPAI Code of Practice（2025 年 7 月 10 日）：三章——Transparency、Copyright、Safety and Security——12 commitments；2026 年 8 月开始 enforcement。UK AISI -> AI Security Institute（2025 年 2 月）：改名 signal 了更窄 scope。US AISI -> CAISI（2025 年 6 月）：NIST 下的 Center for AI Standards and Innovation；转向 pro-growth posture。Korean AI Framework Act（2024 年 12 月通过，2026 年 1 月生效）：Article 12 在 MSIT 下设立 AISI；要求 foreign AI companies 设置 local representatives，并对 high-impact 与 generative AI 做 risk assessment 与 safety measures。

**Type / 类型：** Learn / 学习
**Languages / 语言：** none
**Prerequisites / 前置知识：** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 EU AI Act risk tiers（prohibited、high-risk、general-purpose、limited-risk）以及 August 2025 / August 2026 / August 2027 timeline。
- 描述 GPAI Code of Practice 的三章，以及各自约束哪些 providers。
- 描述 2025 年 rebrands：UK AISI -> AI Security Institute；US AISI -> CAISI；以及每次改名意味着什么 policy direction。
- 说出 Korea AI Framework Act 的核心条款。

## The Problem / 问题

Lab frameworks（Lesson 18）是 voluntary。Regulatory frameworks 是 compulsory。2024-2026 年，第一波 comprehensive AI regulation 开始生效。Deployers 必须把 technical controls 映射到 regulatory obligations；这个映射因 jurisdiction 而异。

## The Concept / 概念

### EU AI Act / 欧盟 AI 法案

**2024 年 8 月 1 日生效。** Risk-tier structure：

- **Prohibited practices**（Article 5）。Social scoring、public 中 real-time remote biometric identification（带 law-enforcement exceptions）、对 vulnerable groups 的 exploitative manipulation。2025 年 2 月 2 日适用。
- **High-risk systems**（Annex III）。Employment、education、credit、law enforcement、justice、migration。要求 conformity assessment、risk management、logging、transparency。
- **General-Purpose AI (GPAI) models。** 2025 年 8 月 2 日适用。所有 GPAI providers 有 obligations；systemic-risk GPAI（>1e25 FLOP training compute）有额外 obligations。
- **Limited-risk systems。** Article 50 下的 transparency obligations（AI-generated content labelling）。2026 年 8 月 2 日适用。

Timeline：
- 2025 年 2 月 2 日：prohibited practices + AI literacy。
- 2025 年 8 月 2 日：GPAI + governance。
- 2026 年 8 月 2 日：full applicability + Article 50 transparency + penalties up to 15M EUR / 3% global turnover。
- 2027 年 8 月 2 日：legacy GPAI + embedded high-risk。

Commission 在 2025 年末提议把 high-risk timeline 调整为 16 months。

### GPAI Code of Practice / GPAI 实践准则

2025 年 7 月 10 日发布。三章：

- **Transparency。** 所有 GPAI providers。
- **Copyright。** 所有 GPAI providers。
- **Safety and Security。** Systemic-risk GPAI providers（估计 5-15 家公司）。

共 12 commitments。AI Office 负责的 Signatory Taskforce 管理实施。2026 年 8 月 2 日开始 enforcement；在此之前接受 good-faith compliance。

### Transparency Code for Article 50 / Article 50 透明度准则

第一稿 2025 年 12 月 17 日。第二稿 2026 年 3 月。最终版本 2026 年 6 月。覆盖 AI-generated content labelling，包括 deepfakes——这是要求 Lesson 23 watermarking technology 的 regulatory layer。

### UK AI Security Institute (February 2025) / 英国 AI Security Institute（2025 年 2 月）

从 AI Safety Institute 改名。这个 rebrand 缩窄 scope：弱化 algorithmic bias 和 free-speech framing，聚焦 frontier capability security。开源 Inspect evaluation tool（2024 年 5 月）。与 Redwood（Lesson 10）合作 control safety cases。

### US CAISI (June 2025) / 美国 CAISI（2025 年 6 月）

Trump administration 将 NIST 的 AI Safety Institute 转为 Center for AI Standards and Innovation。根据 VP Vance 在 Paris AI Action Summit 的表态，转向 “pro-growth AI policies”。减少 pre-deployment evaluation 重点，强调 standards 与 innovation support。作为对 EU AI Act regulatory posture 的美国国内 counterweight。

### Korean AI Framework Act / 韩国 AI Framework Act

2024 年 12 月通过。2025 年 1 月颁布。2026 年 1 月生效。整合 19 个分散 AI bills。

Article 12 在 Ministry of Science and ICT（MSIT）下设立 AISI。要求：
- 在韩国运营的 foreign AI companies 设置 local representatives。
- 对 “high-impact” AI systems 做 risk assessment。
- 对 generative AI 与 high-impact AI 采取 safety measures。

这是亚洲首个 comprehensive horizontal AI regulation。

### Cross-jurisdiction dynamics / 跨司法辖区动态

- EU：严格、risk-tiered、罚款重。privacy-adjacent regulation 的 benchmark。
- US：偏 innovation、decentralized，由 states（例如 California AB 2013——Lesson 27）补 federal gaps。
- UK：聚焦 narrow security，evaluation infrastructure 强。
- Korea：MSIT-led，重视 foreign-provider obligations。

监管哲学竞争。跨 jurisdictions 的 deployers 必须遵守最严格规则；2026 年通常是 EU AI Act。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lesson 18 是 lab-voluntary governance；Lesson 24 是 regulatory；Lesson 25 是 AI systems 的新兴 CVEs；Lessons 26-27 覆盖 documentation（cards）与 training-data governance。

## Build It / 动手构建

本课没有代码实现。动手工作是把一个 deployment 映射到各 jurisdiction 的 obligations：risk tier、provider type、deadline 与 required artifacts。

## Use It / 应用它

没有代码。阅读 EU AI Act primary sources：regulation text、GPAI Code of Practice、UK AISI Inspect framework。把你的 deployment 映射到每个 jurisdiction 的 applicable obligations。

## Ship It / 交付它

本课产出 `outputs/skill-regulatory-map.md`。给定 deployment description，它会映射 applicable jurisdictions、各 jurisdiction 的 tier classifications、per-jurisdiction obligations 和 deadline structure。

## Exercises / 练习

1. 阅读 EU AI Act（regulation 2024/1689）和 GPAI Code of Practice（2025 年 7 月 10 日）。识别适用于所有 GPAI providers 的三个 obligations，以及只适用于 systemic-risk GPAI 的三个 obligations。

2. 一个 deployment 由美国公司开发、运行在欧盟 infrastructure、服务韩国用户。哪些三个 jurisdictions 的规则适用？每个 substantive question 上哪条规则约束？

3. UK AI Security Institute 的 rename 缩窄 scope。分别支持和反对这个 narrower framing。指出每个立场依赖的 policy assumption。

4. CAISI 的 “pro-growth” framing 偏离了 2022-2024 AI safety institute model。识别两个可测量的 policy shifts。

5. Korea AI Framework Act 要求 foreign providers 设置 local representatives。描述一个 Bay Area company 服务韩国用户时的 operational implications。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| EU AI Act | “the regulation” | 基于 risk tiers 的 horizontal AI regulation；2024 年 8 月生效 |
| GPAI | “general-purpose AI” | 大型 foundation models；systemic-risk subset 有额外 obligations |
| Article 50 | “transparency obligations” | AI-generated content labelling；2026 年 8 月适用 |
| UK AISI | “AI Security Institute” | 2025 年 2 月改名；更窄的 frontier-security focus |
| CAISI | “US center for AI standards” | 2025 年 6 月从 AI Safety Institute 改名；pro-growth posture |
| Korean AI Framework Act | “MSIT horizontal regulation” | 亚洲首个 comprehensive AI law；2026 年 1 月生效 |
| Systemic-risk GPAI | “the 1e25 FLOP threshold” | additional obligations tier；估计约束 5-15 家公司 |

## Further Reading / 延伸阅读

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — regulation 与 timeline。
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — three-chapter code。
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — official page。
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — Korean framework analysis。
