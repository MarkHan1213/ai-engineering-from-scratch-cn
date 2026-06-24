# OpenAI Preparedness Framework and DeepMind Frontier Safety Framework / OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入 Research Categories——Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards——并把它们与 Tracked Categories 区分开。Tracked Categories 会触发 Capabilities Reports 和 Safeguards Reports，并由 Safety Advisory Group review。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日加入 Tracked Capability Levels）把 autonomy 折叠进 ML R&D 和 Cyber domains（ML R&D autonomy level 1 = 以相对 human + AI tools 有竞争力的成本完全自动化 AI R&D pipeline）。FSF v3 明确通过 automated monitoring for instrumental-reasoning misuse 处理 deceptive alignment。诚实说明是：PF v2 中的 Research Categories（包括 Long-range Autonomy）不会自动触发 mitigations；政策用语是 “potential”。DeepMind 自己也说，如果 instrumental reasoning 变强，automated monitoring “will not remain sufficient long-term”。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites / 前置知识：** Phase 15 · 19 (Anthropic RSP)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 对比 Anthropic RSP、OpenAI PF v2 和 DeepMind FSF v3 对同一能力的分类方式。
- 区分 OpenAI 的 Tracked Categories 与 Research Categories。
- 解释 DeepMind 为什么把 autonomy 放入 ML R&D 和 Cyber 等具体 domains。
- 分析 sandbagging / deceptive alignment 如何削弱所有 capability thresholds。
- 为一个具体 capability 生成 cross-policy comparison。

## The Problem / 问题

Lesson 19 仔细读了 Anthropic 的 scaling policy。本课补齐 OpenAI 和 DeepMind。三份文件是同一类 artifact，回答同一个问题：frontier lab 应在何时 pause 或 gate 一个模型。它们在少数 categories 上收敛，也在一些重要细节上分歧。

收敛点：三者都把 long-range autonomy 视为值得跟踪的 capability class。三者都承认 deceptive behavior（alignment faking、sandbagging）是具体风险类别。三者都有 internal review body。分歧点：OpenAI 把 categories 拆成 “Tracked”（mandatory mitigation）和 “Research”（不自动触发）。DeepMind 没有单独命名 autonomy，而是把它折叠进两个 domains。实验室命名为 Tracked vs Research、Critical vs Moderate、Tier-1 vs Tier-2；但 capability 位于哪个 bucket，在不同实验室中有不同操作后果。

把它们放在一起读才有价值。同一能力在 Anthropic 可能是 “mandatory mitigation”，在 OpenAI 可能是 “monitored but not triggering”，在 DeepMind 可能是 “tracked in a specific domain”。Policy posture 很重要。

## The Concept / 概念

### OpenAI Preparedness Framework v2 (April 2025) / OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发 Capabilities Reports（模型能做什么）加 Safeguards Reports（有哪些 mitigations）。部署前由 Safety Advisory Group review。
- **Research Categories**：实验室正在跟踪、但尚未承诺具体 mitigations 的模型能力。包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

V2 的关键 framing：Research Categories 不会自动触发 mitigations。政策语言是 “potential” mitigations。这里是需要盯住的具体 delta：一个 category 位于 Tracked 还是 Research，是 “gated” 和 “observed” 的分界。

### DeepMind Frontier Safety Framework v3 (September 2025; Tracked Capability Levels added April 17, 2026) / DeepMind FSF v3（2025 年 9 月；2026 年 4 月 17 日加入 Tracked Capability Levels）

结构：

- **Critical Capability Levels (CCLs)**：横跨五个 domains 的 capability thresholds：Cyber、Bio、ML R&D、CBRN、Autonomy（被折叠进 ML R&D 和 Cyber）。
- **Tracked Capability Levels**：2026 年 4 月加入的额外粒度。具体例子：ML R&D autonomy level 1 = 以相对 human + AI tools 有竞争力的成本完全自动化 AI R&D pipeline。
- **Deceptive alignment monitoring**：明确承诺针对 instrumental-reasoning misuse 做 automated monitoring。

Autonomy framing 与 OpenAI 不同。DeepMind 没有保留 “Autonomy” 作为 top-level domain，而是把它折叠进 autonomy 会造成伤害的 domains（ML R&D 和 Cyber）。支持论点是：没有 domain 的 autonomy 是 capability without risk；反方论点是：跨 domain autonomy 是 framework 应命名的 meta-risk。

### What all three converge on / 三者收敛在哪里

- Internal Safety Advisory Group（Anthropic SAG、OpenAI SAG、DeepMind internal committee）。High-capability models 部署前 review。
- 明确提到 deceptive alignment / alignment faking 作为 risk class。
- 有 declared cadence 的 standing artifacts（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities and Safeguards Reports；DeepMind：FSF update cycle）。
- 承认 monitoring-only defenses 有上限。DeepMind 说得很明确：“automated monitoring will not remain sufficient long-term.”

### Where they diverge / 分歧在哪里

- **Anthropic**：v3.0 移除了 pause commitment；AI R&D-4 threshold 是命名的下一 gate。
- **OpenAI**：Tracked vs Research split；Research Categories（包括 Long-range Autonomy）不会自动 gate。
- **DeepMind**：autonomy 折叠进其他 domains；2026 年 4 月 Tracked Capability Levels 增加粒度。

### Sandbagging: a specific capability that complicates all three / Sandbagging：让三者都复杂化的具体能力

Sandbagging（模型在 evaluation 上策略性低表现）位于 OpenAI 的 Research Categories。Anthropic 的 RSP v3.0 通过 evaluation-context gap（Lesson 1）处理它。DeepMind 通过 FSF v3 中的 deceptive alignment monitoring 处理它。

如果模型在 evaluations 上 sandbag，每个 framework 的 capability thresholds 都会被低估。Framework 只有在 measurement 有效时才有效。这就是为什么除了 lab self-assessment 之外，还需要 external measurement（Lesson 21, METR）和 adversarial evaluation。

### The policy-reading skill / 读政策的技能

- Locate：你关心的每种 capability 都应该能在政策中找到。找不到，就说明政策没有覆盖。
- Classify：它是 Tracked（触发 mitigation）还是 Research（跟踪但不触发）？OpenAI 这样命名；Anthropic 和 DeepMind 有各自等价物。
- Cadence：政策是按声明节奏更新，还是只在特定事件后更新？声明 cadence 更强。
- Independence：外部 review 是 mandatory 还是 optional？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR 合作；DeepMind 主要依赖 internal SAG。

## Build It / 动手构建

本课把三份政策写成 decision-table diff：输入一个 capability（autonomy、deceptive alignment、R&D automation、cyber uplift 等），输出三份政策如何分类它，以及会触发哪些 mitigations。

## Use It / 应用它

`code/main.py` 实现一个小型 decision-table diff tool。给定一个 capability（autonomy、deceptive alignment、R&D automation、cyber uplift 等），它输出三份政策分别如何分类该 capability，以及触发哪些 mitigations。它是阅读辅助，不是政策工具。

## Ship It / 交付它

`outputs/skill-cross-policy-diff.md` 使用三份 frameworks 作为 reference，为具体 capability 生成 cross-policy comparison。

## Exercises / 练习

1. 运行 `code/main.py`。至少选择两个可以对照 source documents 验证的 capabilities，确认 diff tool 输出与政策一致。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每个 Research Category。对每个写一句话解释为什么它在 Research 而不是 Tracked。

3. 完整阅读 DeepMind FSF v3，以及 2026 年 4 月 Tracked Capability Levels update。识别 ML R&D autonomy level 1 的具体 evaluation criteria。你会如何外部测量它？

4. Sandbagging 位于 OpenAI Research Categories。设计一个评测，迫使 sandbagging model 暴露真实能力。引用 Lesson 1 中 eval-context-gaming 的讨论。

5. 选择一个具体 capability，对比三份政策。说出你认为哪个政策分类最严格，哪个最弱，并用 source text 说明。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Preparedness Framework | “OpenAI 的 scaling policy” | PF v2（2025 年 4 月）；Tracked vs Research categories |
| Tracked Category | “Mandatory mitigation” | 触发 Capabilities + Safeguards Reports；SAG review |
| Research Category | “Monitored only” | 跟踪但不自动 mitigation；包括 Long-range Autonomy |
| Frontier Safety Framework | “DeepMind 的 scaling policy” | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | “Critical Capability Level” | DeepMind 按 domain 定义的 threshold（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1 | “R&D automation” | 以有竞争力成本完全自动化 AI R&D pipeline |
| Sandbagging | “策略性低表现” | 模型在 evals 上故意表现较差；位于 OpenAI Research Categories |
| Instrumental reasoning | “Means-ends reasoning” | 关于如何达成目标的推理；DeepMind monitoring 的目标 |

## Further Reading / 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 announcement。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 announcement。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels addition。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF-format Risk Report 示例。
