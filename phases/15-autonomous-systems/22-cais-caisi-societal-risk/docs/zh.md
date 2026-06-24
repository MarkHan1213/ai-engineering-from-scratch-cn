# CAIS, CAISI, and Societal-Scale Risk / CAIS、CAISI 与社会尺度风险

> Center for AI Safety（CAIS，San Francisco，由 Hendrycks 和 Zhang 于 2022 年创立）发布 four-risk framework——malicious use、AI races、organizational risks、rogue AIs——以及 2023 年 5 月关于 extinction risk 的声明，数百名教授和公司领导签署。CAIS 2026 releases 包括：frontier-model evaluation 的 AI Dashboard、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。另一个不同实体是 NIST Center for AI Standards and Innovation（CAISI）——面向美国政府的 voluntary agreements 和 unclassified capability evaluations，聚焦 cyber、bio、chemical-weapons risks。CAIS 把 organizational risk 标为四大顶层风险之一：safety culture、rigorous audits、multi-layered defenses、information security 是基础，却经常被 deployment speed 牺牲。如果签署，California SB-53 将成为美国首个州级 catastrophic-risk regulation。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites / 前置知识：** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分 CAIS 与 NIST CAISI 的组织性质和任务边界。
- 描述 CAIS four-risk framework 的四个顶层类别。
- 解释 organizational risk 为什么最贴近从业者可控范围。
- 将 lab scaling policies、external evaluations、civil society 和 government programs 放进同一 defense-in-depth 图景。
- 用 four-risk inventory 审查一个 proposed deployment 的 societal-scale-risk posture。

## The Problem / 问题

Lessons 19 和 20 覆盖了 lab-internal scaling policies。Lesson 21 覆盖了 independent capability evaluation。本课覆盖第三个视角：影响公共讨论与 catastrophic AI risk 监管 baseline 的 civil society 和 government organizations。

两个不同实体很重要。CAIS 是非营利 research org，发布 AI risk 思考框架并协调公共声明。CAISI 是 NIST 内的美国政府中心，运行与实验室的 voluntary agreements 和 unclassified capability evaluations。名字相近，使命不重叠。从业者应同时了解二者。

实际内容是：CAIS 的 four-risk framework 是文献中最常被引用的 societal-scale-risk taxonomy。Safety culture 和 organizational risk 是四类之一，也是最直接受 practitioner 控制的一类。SB-53（California）如果签署，将成为美国首个州级 catastrophic-risk regulation；它的 framing 很重要，因为美国科技政策中，州级监管历来可能领先联邦行动。

## The Concept / 概念

### CAIS — Center for AI Safety / CAIS：Center for AI Safety

- Founded：2022 年，San Francisco，由 Dan Hendrycks 和同事创立（“Zhang” 指早期 collaborator，不是当前 co-founder；当前 leadership 见 CAIS website）。
- Status：501(c)(3) non-profit。
- Notable 2023 output：关于 extinction risk 的声明，由数百名 researchers 和 CEOs 共同签署。核心表述是：降低 AI 带来的 extinction risk，应当像 pandemics 和 nuclear war 这类社会尺度风险一样成为 global priority。
- 2026 outputs：frontier-model evaluation 的 AI Dashboard、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。

### The four-risk framework / 四风险框架

CAIS framework 把 catastrophic AI risk 分成四个顶层类别：

1. **Malicious use / 恶意使用**：恶意行为者使用 AI 造成伤害（bioweapons synthesis、disinformation、cyberattacks）。
2. **AI races / AI 竞赛**：实验室、公司或国家之间的竞争压力推动部署越过安全点。
3. **Organizational risks / 组织风险**：实验室内部动态（safety-culture failures、insufficient audit、under-resourced security）导致糟糕部署。
4. **Rogue AIs / 失控 AI**：足够强的 AI 追求与 human welfare 冲突的目标。

这不是唯一 taxonomy；它是被引用最多的 taxonomy。类别之间并非互斥——一个组织在竞赛中牺牲 audit 换速度，最终产出 rogue AI，四类都触及。

### Where organizational risk lives / organizational risk 在哪里

四类中，organizational risk 对从业者最可操作。一个实验室的 safety culture、audit rigor、defense layering 和 information security，决定了模型发布时 Lessons 10-18 的 controls 是真的到位，还是没人验证的 checklist items。

具体 organizational-risk levers：

- **Safety culture / 安全文化**：团队成员能否在没有职业代价的情况下升级 concern？CAIS surveys 发现它强预测其他 levers。
- **Rigorous audits / 严格审计**：外部和内部都需要。Internal-only audits 容易产生乐观报告。
- **Multi-layered defenses / 多层防御**：没有单层足够（Phase 15 的贯穿主题）。
- **Information security / 信息安全**：model weights leaking、eval data leaking、monitor-bypass techniques leaking。Lesson 19 的 RAND SL-4 是具体标准。

### CAISI — Center for AI Standards and Innovation / CAISI：Center for AI Standards and Innovation

- 隶属于 NIST。
- 运行与 frontier labs 的 voluntary agreements。
- 发布聚焦 cyber、bio 和 chemical-weapons risks 的 unclassified capability evaluations。
- 与 CAIS 不同；acronyms 碰撞；阅读时检查 URL（nist.gov）来确认是哪一个。

CAISI 的角色是 METR 私有实验室 engagements（Lesson 21）的公开、政府面 counterpart。CAISI reports 是 unclassified；METR reports 经常受 NDA 限制。两者都读，从业者能得到更完整图景。

### California SB-53 / 加州 SB-53

California Senate bill（2025-2026 session）处理 frontier models 带来的 catastrophic risk。草案关键条款包括：

- 触发州级义务的具体 capability thresholds。
- AI lab employees 的 whistleblower protections。
- Catastrophic failures 的 incident reporting requirements。

如果签署，它会成为美国首个州级 catastrophic-risk regulation。不论签署状态如何，该法案的 framing 都会影响其他州议会如何处理这个问题。California 从业者应该跟踪法案状态；其他地区从业者也应阅读它，理解美国州级监管可能长什么样。

### Societal-scale risk is not a single-layer problem / 社会尺度风险不是单层问题

Phase 15 的贯穿主题——defense in depth——同样适用于社会层。没有单个组织、regulation 或 framework 能关闭 catastrophic risk。生态只有在这些层一起工作时才有效：

- Labs 发布 scaling policies（Lessons 19、20）。
- External evaluators 产出 measurements（Lesson 21）。
- Civil society 跟踪并公开传播（CAIS）。
- Government 运行 voluntary programs 和 baseline regulation（CAISI、SB-53）。
- Practitioners 构建 multi-layered controls（Lessons 10-18）。

这是本阶段的最终综合：前面每一课都是一个栈中的一层；栈的完整性比任何单层强度更重要。

## Build It / 动手构建

本课构建一个 four-risk inventory tool：输入 proposed deployment，输出它触及哪些 risk categories，以及相应 mitigation checklist。它是阅读框架的辅助工具，不替代人类判断。

## Use It / 应用它

`code/main.py` 实现一个小型 risk-inventory tool。给定 proposed deployment，它会把 deployment 标到 four-risk categories，并返回 mitigation checklist。它是 framework 的阅读辅助，不是 human judgment 的替代品。

## Ship It / 交付它

`outputs/skill-societal-risk-review.md` 审查 deployment 的 societal-scale-risk posture：触及四类中的哪些，已有 mitigations 是什么，organizational-risk exposure 有多大。

## Exercises / 练习

1. 运行 `code/main.py`。输入三个不同规模的 synthetic deployments。确认 four-risk tags 符合预期；找出一个 tool under- 或 over-tags 的案例。

2. 完整阅读 CAIS four-risk paper。选择一个 risk category，写两段说明你认为该类别中 2026 年最重要的发展是什么。

3. 阅读 California SB-53 当前草案。找出一个你认为强化 catastrophic-risk posture 的条款，以及一个你认为削弱它的条款。说明理由。

4. 选择一个你熟悉的 production AI deployment（自己的或公开的）。按 organizational-risk sub-levers 打分：safety culture、audit rigor、multi-layered defenses、information security。哪一项最弱？补齐到同等水平需要什么成本？

5. 草拟 2028 版 four-risk framework，反映额外一年的 capability 和部署经验。你会增加、移除或重组什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| CAIS | “Center for AI Safety” | 非营利组织；four-risk framework；2023 extinction statement |
| CAISI | “US government AI safety” | NIST Center；voluntary agreements；unclassified evals |
| Four-risk framework | “CAIS 的 taxonomy” | malicious use、AI races、organizational risks、rogue AIs |
| Malicious use | “Bad actor uses AI” | Bioweapons、disinformation、cyberattacks |
| AI races | “Competitive pressure” | Labs/companies/nations 推动部署越过 safety |
| Organizational risk | “Lab internal failure” | Safety culture、audit、defenses、infosec |
| Rogue AI | “Misaligned agent” | 有能力 AI 追求与 human welfare 冲突的目标 |
| California SB-53 | “State-level regulation” | 2025-2026 bill；若签署，将成为美国首个州级 catastrophic-risk regulation |

## Further Reading / 延伸阅读

- [Center for AI Safety](https://safe.ai/) — four-risk framework 的机构主页。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — four-risk paper。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 简短联合声明。
- [NIST CAISI](https://www.nist.gov/caisi) — 面向政府的 AI standards and innovation center。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 把 lab-level commitments 与 societal-scale framing 连接起来。
