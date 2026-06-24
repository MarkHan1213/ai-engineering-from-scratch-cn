# Data Provenance and Training-Data Governance / 数据来源与训练数据治理

> EU AI Act 要求 GPAI 在 2025 年 8 月前支持 machine-readable opt-out standards（通过 EU Copyright Directive TDM exception）。California AB 2013（2024 年签署）——Generative AI training-data transparency 要求 developers 发布带 12 个 mandated fields 的 dataset summary。2025 年 DPA 对 legitimate interest 的趋同：Irish DPC（2025 年 5 月 21 日）在 EDPB opinion 后接受 Meta 在 first-party public EU/EEA adult content 上训练 LLM 的计划及 safeguards；Cologne Higher Regional Court（2025 年 5 月 23 日）驳回 injunction；Hamburg DPA 放弃 urgent procedure；UK ICO（2025 年 9 月 23 日）对 LinkedIn 的 AI-training safeguards（transparency、simplified opt-out、extended objection windows）给出 positive regulatory response 并继续 monitoring——不是 formal clearance。Brazilian ANPD（2024 年 7 月 2 日）因 insufficient information transparency 暂停 Meta processing；Meta 提交 compliance plan 后，preventive measure 于 2024 年 8 月 30 日解除。关键 irreversibility problem：cookie-consent frameworks 面向 real-time、reversible tracking；数据一旦进入 model weights，就无法 surgical erasure——trained neural networks 没有 practical GDPR right-to-erasure。Compliance window 在 collection time。Data Provenance Initiative（dataprovenance.org，Longpre, Mahari, Lee et al.，“Consent in Crisis”，2024 年 7 月）：大规模 audit 显示 AI data commons 随 publishers 添加 robots.txt restrictions 而快速收缩。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites / 前置知识：** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 California AB 2013 对 Generative AI training-data transparency 的 12 个 mandated fields。
- 说明 2025 年 DPAs 对 legitimate-interest LLM training 的立场（Irish DPC、UK ICO、Hamburg、Cologne）。
- 描述 irreversibility problem：为什么 trained neural networks 没有 practical GDPR right-to-erasure。
- 说出 Data Provenance Initiative “Consent in Crisis” 的发现。

## The Problem / 问题

Training-data governance 是每个 model card（Lesson 26）和 regulatory obligation（Lesson 24）的 upstream。2024-2025 年，监管 landscape 收敛到三个原则：opt-out infrastructure、per-dataset disclosure、对 publicly available data 的 legitimate-interest accommodations。Providers 如果在 collection time 不合规，后续很难 remediate。

## The Concept / 概念

### California AB 2013 / 加州 AB 2013

2024 年签署。对于 2022 年 1 月 1 日或之后发布的 systems，documentation 必须在 2026 年 1 月 1 日或之前发布。Section 3111(a) 要求 developers 发布用于 training 的 datasets high-level summary，包含 12 个法定 items：
1. Datasets 的 sources 或 owners。
2. Datasets 如何推进 AI system intended purpose 的说明。
3. Datasets 中 data points 数量（可用 general ranges；dynamic datasets 可估计）。
4. Data points 类型说明（labeled datasets 的 label types；unlabeled 的 general characteristics）。
5. Datasets 是否包含受 copyright、trademark、patent 保护的数据，或完全属于 public domain。
6. Datasets 是否 purchased 或 licensed。
7. Datasets 是否包含 personal information（按 Cal. Civ. Code §1798.140(v)）。
8. Datasets 是否包含 aggregate consumer information（按 Cal. Civ. Code §1798.140(b)）。
9. Developer 做过的 cleaning、processing 或其他 modification，以及 intended purpose。
10. 数据收集 time period；若 ongoing collection 需说明。
11. Datasets 在 development 中首次使用的 dates。
12. System 是否使用或持续使用 synthetic data generation。

Item 12（synthetic data）是相对 Gebru et al. 2018 datasheets 的新增项。Item 7（personal information）触发 Privacy Rights Act（CPRA）obligations。该 statute 豁免 security/integrity、aircraft-operation 和 federal-only national-security systems（Section 3111(b)）。

### EU AI Act (Lesson 24) and TDM opt-out / EU AI Act 与 TDM opt-out

EU Copyright Directive 的 text-and-data-mining exception 允许在 publicly available content 上训练，除非 rightholder opt out。EU AI Act GPAI Code of Practice Copyright chapter 要求 GPAI providers 尊重 machine-readable opt-out signals（robots.txt、C2PA “No AI Training” claim 等）。

### 2025 DPA convergence on legitimate interest / 2025 DPAs 对 legitimate interest 的收敛

Irish DPC（2025 年 5 月 21 日）：在 EDPB opinion 后，接受 Meta 计划用 first-party public EU/EEA adult-user content 训练并带 safeguards。Cologne Higher Regional Court（2025 年 5 月 23 日）驳回针对 Meta 的 injunction：opt-out 足够。Hamburg DPA 为 EU-wide consistency 放弃 urgency procedure。UK ICO（2025 年 9 月 23 日）对 LinkedIn 以类似 safeguards 恢复 AI training 给出 positive regulatory response——不是 formal clearance——并继续 monitoring。

收敛原则：legitimate interest 可以为 publicly available first-party content 上的 training 提供正当性，只要有 opt-out。Consent 不是必需。

### Brazilian ANPD (June 2024) / 巴西 ANPD（2024 年 6 月）

因 information transparency 不足，暂停 Meta 对 Brazilian user data 的 AI training processing。结果不同于 EU DPAs——ANPD 优先考虑 transparency，而不是 legitimate-interest admissibility。

### The irreversibility problem / 不可逆问题

Cookie-consent 设计用于 real-time、reversible tracking。Training data 不同：数据一旦进入 model weights，surgical erasure 不可能。完全 remediation 只能从头 retraining，而成本高到不可行。

部分 remediation：
- **Unlearning。** 近似移除；用 MIA 测量（Lesson 22）。
- **Influence function-based localization。** 找到受该数据影响最大的 weights；选择性 update。
- **Fine-tune-suppression。** 训练模型拒绝输出源自该数据的内容。

没有一种完全解决问题。Compliance window 在 collection time。

### Data Provenance Initiative / Data Provenance Initiative

dataprovenance.org。Longpre, Mahari, Lee et al. “Consent in Crisis”（2024 年 7 月）：对 AI training data commons 做大规模 audit。发现：publishers 正在加速添加 robots.txt restrictions。可公开训练的数据 commons 快速收缩。2023 -> 2024 年间，top training sources 中约 25% 添加了某种 restriction。含义：未来 training-data availability 取决于新的 acquisition paradigms（licensing、synthetic generation、incentivized participation）。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lesson 26 是 model-level documentation。Lesson 27 是 dataset-level governance。二者共同定义 transparency layer。Lesson 28 映射研究这些问题的 research ecosystem。

## Build It / 动手构建

本课构建 California AB 2013 12-field dataset summary scaffold。你会填充 toy dataset，并看到哪些字段触发 privacy、copyright 或 opt-out follow-on obligations。

## Use It / 应用它

`code/main.py` 为 toy dataset 生成 California AB 2013-compliant 12-field dataset summary scaffold。你可以填充字段，并观察哪些字段触发 privacy 或 copyright follow-on obligations。

## Ship It / 交付它

本课产出 `outputs/skill-provenance-check.md`。给定用于 training 的 dataset，它会检查 AB 2013 12-field coverage、opt-out infrastructure compliance、DPA alignment 和 irreversibility-risk assessment。

## Exercises / 练习

1. 运行 `code/main.py`。为 toy dataset 生成 12-field summary，并识别哪些 fields under-specified。

2. EU Copyright Directive TDM opt-out 是 machine-readable。提出一种 opt-out signal 标准格式，并与 robots.txt 和 C2PA “No AI Training” 比较。

3. 阅读 Data Provenance Initiative 的 “Consent in Crisis”（2024 年 7 月）。描述 restrictions 增长最快的三类 content categories，并论证一个经济后果。

4. 2025 DPA alignment 接受 public-content training 的 legitimate interest。构造一个 legitimate interest 不足够的场景，并识别 provider 需要的 legal basis。

5. Sketch 一个 training-data-provenance manifest，把 AB 2013 fields 与每个 dataset 的 C2PA-signed provenance chain 组合起来。识别一个 technical barrier 和一个 legal barrier。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| AB 2013 | “the California law” | Generative AI training-data transparency；12 个 mandated fields |
| TDM exception | “text-and-data-mining” | 带 opt-out 的 EU Copyright Directive training-data exception |
| Legitimate interest | “the EU basis” | GDPR Article 6 中可正当化 public content training 的 legal basis |
| Opt-out signal | “machine-readable no-train” | robots.txt、C2PA “No AI Training”、TDM.Reservation |
| Irreversibility | “cannot un-train” | model weights 中的数据不能 surgical removal |
| Unlearning | “approximate removal” | post-training interventions，用于降低模型对特定数据的依赖 |
| Consent in Crisis | “the DPI audit” | 2024 年 7 月关于 robots.txt restrictions 加速增长的发现 |

## Further Reading / 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Generative AI training-data transparency law。
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright chapter。
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI audit。
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — regulatory context。
