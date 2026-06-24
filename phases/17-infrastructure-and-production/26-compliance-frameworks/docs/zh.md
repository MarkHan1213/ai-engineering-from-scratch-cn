# Compliance — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001 / 合规：SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> Multi-framework coverage 是 2026 年 enterprise deals 的基本门槛。**EU AI Act**：2024 年 8 月 1 日生效。多数 high-risk requirements 于 2026 年 8 月 2 日执行。High-risk-system obligations（Art. 99(4)）最高罚 €15M 或 global annual turnover 3%；prohibited AI practices（Art. 99(3)）最高罚 €35M 或 7%。只要服务 EU users，就全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（由 SB25B-004 从 2026 年 2 月延期），要求 high-risk systems 做 impact assessments，并提供 AI decisions appeal 权利。Virginia 对 credit/employment/housing/education 类似。**SOC 2 Type II**：B2B AI 的事实要求（fintech 要 Type II，不是 Type I）。**GDPR**：最大 documented AI-specific fine 是 Dutch DPA 2024 年 9 月对 Clearview AI 的 €30.5M；Italy's Garante 2024 年 12 月对 OpenAI 开出 €15M（2026 年 3 月 appeal 被 overturn）。Real-time PII redaction at inference 是可辩护标准；post-processing cleanup 不够。**HIPAA**：healthcare bound；没有 BAA，不能把 PHI 发给 external AI services。**PCI-DSS**：AI-interaction-layer coverage 需要 configuration + contractual agreements，不是自动拥有。**ISO 42001**：新兴 AI governance standard，正与 ISO 27001 一起成为 procurement requirement。参考 profile：OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA、ChatGPT payment components 的 PCI-DSS。Cross-framework mapping 可以减少 audit fatigue：access controls 映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**类型：** 学习
**语言：**（Python optional — compliance is policy + process, not code）
**前置知识：** 第 17 阶段 · 25（Security）, 第 17 阶段 · 13（Observability）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 枚举 2026 年 LLM products 相关的七个 frameworks，并把它们匹配到 customer segment。
- 引用 EU AI Act enforcement timeline（2024 年 8 月生效；2026 年 8 月 high-risk enforcement），以及两级 fine ceiling（high-risk obligations 为 €15M / 3%，prohibited practices 为 €35M / 7%）。
- 解释为什么 post-processing PII cleanup 不够，并说出 real-time inference-layer redaction 是可辩护标准。
- 描述 cross-framework control mapping（例如 access control 映射到 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## The Problem / 问题

一个 enterprise customer 的 procurement 要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及“EU AI Act compliance statement”。你的团队只有 SOC 2 Type I。离 Type II 还有六个月，也没开始 GDPR Article 30 records。

Multi-framework coverage 不是 LLM 专属问题，而是 enterprise-SaaS 问题，只是叠加了 LLM-specific overlays。2026 年 procurement teams 要的是一张 matrix：每个 framework 一行，每个 control 一列，而不是 PDF。

## The Concept / 概念

### The seven frameworks / 七个 frameworks

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required; PHI cannot leave infrastructure without signed agreement |
| GDPR | EU users | Real-time PII redaction; data subject rights; Article 30 records |
| PCI-DSS | Payment data | Configuration + contracts for AI touching payment |
| EU AI Act | Serving EU users | Risk tier classification; high-risk systems: conformity assessment, documentation, logging |
| Colorado AI Act | Serving CO residents | Impact assessments; right to appeal |
| ISO 42001 | AI governance | Emerging; pairs with ISO 27001 |

### EU AI Act timeline / EU AI Act 时间线

- 2024 年 8 月 1 日：in force。
- 2025 年 2 月 2 日：prohibited-AI practices enforced。
- 2026 年 8 月 2 日：high-risk systems enforced（conformity assessment、documentation、logging）。
- 2027 年 8 月：harmonized legislation 下产品中的 high-risk systems。

Risk tiers：Unacceptable（banned）、High-risk（conformity + logging）、Limited-risk（transparency）、Minimal-risk（no constraint）。多数 B2B LLM SaaS 是 limited-risk；employment、credit、education、law enforcement、migration、essential services 会触发 high-risk。

Fines（Article 99）：breaches of high-risk-system obligations（Art. 99(4)）最高 €15M 或 global annual turnover 3%；prohibited AI practices（Art. 99(3)）最高 €35M 或 7%；以较高者为准。

### GDPR — real-time redaction is the standard / GDPR：real-time redaction 是标准

Post-processing cleanup（LLM 看过后再 redact PII）不是可辩护姿态：模型已经看到数据。2026 年标准是 real-time inference-layer redaction：

- LLM call 前 entity recognition。
- Consistent tokenization（Mesh approach）保留语义。
- 只存 redacted prompts + consented opt-in raw。

近期 enforcement：Dutch DPA 2024 年 9 月对 Clearview AI 开出 €30.5M，是目前最大 documented AI-specific GDPR fine；Italy's Garante 2024 年 12 月对 OpenAI 开出 €15M，是最大 LLM-specific fine，虽然后者在 2026 年 3 月 appeal 中被 overturn，裁决仍处进一步 review。Post-processing claims 在 audit 中站不住。

### HIPAA — BAA is not optional / HIPAA：BAA 不是可选项

没有签署 Business Associate Agreement，不能把 PHI 发给 external AI services。三家 hyperscaler LLM platforms（Bedrock、Azure OpenAI、Vertex）都提供 BAAs。OpenAI direct API 提供 BAA。Anthropic direct API 提供 BAA。发送 PHI 前确认。

### SOC 2 Type II / SOC 2 Type II

Type I：controls 已设计并记录。
Type II：controls 在 6-12 个月内有效运行。

2026 年 B2B procurement 默认 Type II。Type I 是起点；Type II 才是门槛。

常见 audit drivers：access logs（谁看了什么）、change management（如何部署）、risk assessments（季度）、incident response（是否测试）。Phase 17 · 25 的 audit log 可直接复用。

### Cross-framework mapping / Cross-framework mapping

一条 access control policy 能满足多个 framework controls：

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Compliance tools（Drata、Vanta、Secureframe）会自动化这种 mapping。规模化后值得付费。

### ISO 42001 — emerging / ISO 42001：新兴标准

2023 年末发布。正与 ISO 27001 一起成为越来越常见的 procurement requirement。它覆盖 AI governance，包括 risk management、data quality、transparency、human oversight。

### OpenAI's reference profile / OpenAI 的参考 profile

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。这大致是 2026 年 enterprise table stakes。

### Numbers you should remember / 你应该记住的数字

- EU AI Act fines：high-risk obligations（Art. 99(4)）最高 €15M / 3%；prohibited practices（Art. 99(3)）最高 €35M / 7%。
- EU AI Act high-risk enforcement：2026 年 8 月 2 日。
- 最大 documented AI-specific GDPR fine：€30.5M，Clearview AI（Dutch DPA，2024 年 9 月）。
- 最大 LLM-specific GDPR fine：€15M，OpenAI（Italy's Garante，2024 年 12 月；2026 年 3 月 appeal overturn）。
- SOC 2 Type II window：6-12 个月 operated controls。
- Colorado AI Act effective date：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月延期）。

## Build It / 动手构建

在 `code/main.py` 中把 controls 映射到多个 frameworks，练习用一条 evidence 同时支撑 SOC 2、GDPR、HIPAA 或 ISO 控制项。

## Use It / 应用它

`code/main.py` 是一个用 Python 写的 compliance-mapping spreadsheet：给定一个 control，列出它满足的 frameworks。

## Ship It / 交付它

本课产出 `outputs/skill-compliance-matrix.md`。给定 customer segment 和 geography，它会指定 required frameworks 和 controls。

## Exercises / 练习

1. 你的第一个 enterprise customer 要求 SOC 2 Type II、HIPAA BAA、EU AI Act statement。赢下这单的 minimum viable compliance posture 是什么？
2. 按 EU AI Act risk tiers 分类三个假设 LLM products。进入 high-risk 后会改变什么？
3. 你意外把 PHI 发给没有 BAA 的 provider。走一遍 incident response。
4. 论证 ISO 42001 对 mid-market AI vendor 在 2026 年是否“必要”。
5. 把你的 LLM audit log fields（Phase 17 · 25）映射到至少三个 framework controls。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| SOC 2 Type II | “audited controls” | 6-12 个月运行中的 controls，经独立 attestation |
| HIPAA BAA | “healthcare contract” | Business Associate Agreement；PHI 必需 |
| GDPR | “EU privacy” | Real-time PII redaction 是 2026 年可辩护标准 |
| EU AI Act | “EU AI rules” | High-risk enforcement August 2026；€15M / 3%（high-risk obligations）— €35M / 7%（prohibited practices） |
| Colorado AI Act | “US AI state law” | 2026 年 6 月 30 日生效（SB25B-004 延期）；impact assessments |
| ISO 42001 | “AI governance” | AI risk + transparency 的新兴 framework |
| ISO 27001 | “security ISMS” | Information Security Management System baseline |
| Conformity assessment | “EU AI doc package” | High-risk requirement：docs、testing、logging |
| Cross-framework mapping | “one control, many frames” | 一条 policy 满足多个 framework controls |

## Further Reading / 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard。
