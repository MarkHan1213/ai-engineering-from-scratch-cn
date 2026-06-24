# Model, System, and Dataset Cards / Model、System 与 Dataset Cards

> 三种 documentation formats 组织了 AI transparency。Model Cards（Mitchell et al. 2019）——模型的 nutrition labels：training data、quantitative disaggregated analyses、ethical considerations、caveats；只有 0.3% 的 Hugging Face model cards 记录 ethical considerations（Oreamuno et al. 2023）。Datasheets for Datasets（Gebru et al. 2018, CACM）——motivation、composition、collection process、labeling、distribution、maintenance；类比 electronics datasheet。Data Cards（Pushkarna et al., Google 2022）——modular layered detail（telescopic、periscopic、microscopic），作为 diverse readers 的 boundary objects。2024-2025 进展：LLM 自动生成（CardGen, Liu et al. 2024）；model-card detail 与 HF downloads 最多 29% increase 相关（Liang et al. 2024）；verifiable attestations（Laminator, Duddu et al. 2024）；sustainability reporting additions for carbon/water（Jouneaux et al. 2025 年 7 月）；EU/ISO regulatory cards 正在出现。System Cards（Sidhpurwala 2024；Meta system-level transparency；“Blueprints of Trust” arXiv:2509.20394）——end-to-end AI system documentation，覆盖 security capabilities、prompt-injection protection、data-exfiltration detection、alignment with human values。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites / 前置知识：** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 Mitchell et al. 2019 原始 model card 与 Gebru et al. 2018 datasheet。
- 描述 Data Cards 的 telescopic/periscopic/microscopic layering。
- 描述 System Cards 及其 end-to-end coverage。
- 说出三个 2024-2025 进展（automated generation、verifiable attestations、sustainability reporting）。

## The Problem / 问题

Regulatory frameworks（Lesson 24）和 lab safety policies（Lesson 18）都要求 documentation。Documentation formats 从 model-specific（model cards）演进到 dataset-specific（datasheets），再到 system-specific（system cards）。每一种都处理不同 scope 的 transparency。2024-2025 的 automation 与 verifiable-attestation work，解决的是长期 adoption problem。

## The Concept / 概念

### Model Cards (Mitchell et al. 2019) / Model Cards

Sections：
- Model details。
- Intended use。
- Factors（用于 evaluation 的 relevant demographic 或 environmental factors）。
- Metrics。
- Evaluation data。
- Training data。
- Quantitative analyses（按 factors disaggregated）。
- Ethical considerations。
- Caveats and recommendations。

Adoption problem：Oreamuno et al. 2023 对 Hugging Face model cards 的 audit 发现，只有 0.3% 记录 ethical considerations。

### Datasheets for Datasets (Gebru et al. 2018) / Datasheets for Datasets

Electronics-datasheet analogy。Sections：
- Motivation（为什么创建 dataset）。
- Composition（里面有什么）。
- Collection process（如何组装）。
- Labeling（如果适用）。
- Uses（intended、prohibited、risks）。
- Distribution。
- Maintenance。

2021 年发表于 CACM。Datasheet 是 upstream documentation；model card 依赖 datasheet 准确。

### Data Cards (Pushkarna et al., Google 2022) / Data Cards

Modular layered detail。三个 zoom levels：
- **Telescopic。** 给 non-experts 的 high-level summary。
- **Periscopic。** 给 ML practitioners 的 middle-level overview。
- **Microscopic。** 给 auditors 的 detailed feature-level documentation。

Boundary-object framing：不同读者从同一文档中提取不同信息。

### System Cards / System Cards

Scope：end-to-end AI system，包括 model + safety stack + deployment context。典型 sections 包括：
- Security capabilities。
- Prompt-injection protection。
- Data-exfiltration detection。
- Alignment with stated human values。
- Incident response。

Sidhpurwala 2024 和 Meta system-level transparency work。“Blueprints of Trust”（arXiv:2509.20394）把 System Card 形式化为 Model Cards 在 deployment layer 的 complement。

### 2024-2025 developments / 2024-2025 进展

- **CardGen（Liu et al. 2024）。** 通过 LLMs 自动生成 model cards；在标准 Mitchell 2019 fields 上，objectivity 高于许多人类撰写 cards。
- **Download correlation（Liang et al. 2024）。** Detailed model cards 与 HF 上最多 29% 更高 downloads 相关——adoption pressure 现在不仅来自 compliance，也来自 market。
- **Laminator（Duddu et al. 2024）。** 通过 hardware TEE / cryptographic signatures 做 verifiable attestations，让 model card 携带 proof-of-claim，而不只是 claim。
- **Sustainability（Jouneaux et al. 2025 年 7 月）。** 增加 carbon、water 和 compute-energy footprint；emerging ISO standards。
- **Regulatory cards。** EU AI Act（Lesson 24）GPAI Code of Practice Transparency chapter 要求 model cards 作为 compliance artifact。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 24-25 是 regulatory 与 CVE layers。Lesson 26 是 documentation layer。Lesson 27 是 training-data governance，也就是 datasheet 的 upstream。Lesson 28 是生成 cards 中 evaluations 的 research ecosystem。

## Build It / 动手构建

本课构建一个 model card、datasheet 与 system card generator。你会看到三个 scopes 的 section structure 如何不同，以及哪些字段需要外部证据而不是 placeholder。

## Use It / 应用它

`code/main.py` 为 toy deployment 生成 minimal model card、datasheet 和 system card。每个都遵循 canonical section structure。你可以检查格式，并比较三个 scopes。

## Ship It / 交付它

本课产出 `outputs/skill-card-audit.md`。给定 model card、datasheet 或 system card，它会审计 section coverage、numerical disaggregation，以及是否存在 verifiable attestations。

## Exercises / 练习

1. 运行 `code/main.py`。检查 generated cards。识别 weak sections（placeholder-only），并说明什么 evidence 能强化它们。

2. 用两个 demographic groups 上的 quantitative disaggregated analysis 扩展 model card（Lesson 20）。

3. 阅读 Oreamuno et al. 2023 关于 0.3% adoption rate 的研究。提出一个 model card specification 的结构性改动，用于提高 ethical-considerations adoption。

4. Laminator（Duddu et al. 2024）使用 TEEs 做 verifiable attestations。设计一个 model-card field，携带 evaluation result 的 cryptographic attestation，并描述 verifier 的角色。

5. 为你过去的一个项目或一个 hypothetical deployment 写 System Card（是 System Card，不是 Model Card）。识别对 third-party auditors 最高价值的 section。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Model Card | “the Mitchell card” | Mitchell et al. 2019 标准 ML model documentation |
| Datasheet | “the Gebru datasheet” | Gebru et al. 2018 标准 dataset documentation |
| Data Card | “the Pushkarna card” | Google 2022 modular layered data documentation |
| System Card | “the deployment card” | 包含 safety stack 的 end-to-end AI system documentation |
| Boundary object | “different readers, one doc” | Data Cards framing：同一文档服务多类受众 |
| Verifiable attestation | “the Laminator attestation” | 附在 documentation claim 上的 cryptographic 或 TEE proof |
| Sustainability field | “carbon / water footprint” | 2025 年 emerging addition，用于 environmental accounting |

## Further Reading / 延伸阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — canonical model card。
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — datasheet paper。
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — layered data documentation。
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card formalization。
