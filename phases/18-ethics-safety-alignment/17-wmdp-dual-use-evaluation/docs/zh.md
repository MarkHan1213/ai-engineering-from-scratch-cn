# WMDP and Dual-Use Capability Evaluation / WMDP 与双用途能力评估

> Li et al.，“The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning”（ICML 2024，arXiv:2403.03218）。4,157 道 multiple-choice questions，覆盖 biosecurity（1,520）、cybersecurity（2,225）和 chemistry（412）。问题位于 “yellow zone”：接近 harmful capability 的 enabling knowledge，经过多位专家审查和 ITAR/EAR legal compliance 过滤。双重用途：dual-use capability 的 proxy evaluation，以及 unlearning benchmark（companion RMU method 降低 WMDP performance，同时保留 general capability）。2024-2025 叙事：OpenAI/Anthropic 2024 早期 evaluations 报告相对 internet search 只是 “mild uplift”；到 2025 年 4 月，OpenAI Preparedness Framework v2 称模型 “on the cusp of meaningfully helping novices create known biological threats”。Anthropic bioweapon-acquisition trial 显示 2.53x uplift，不足以 rule out ASL-3。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites / 前置知识：** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 WMDP 的三个 domains、question counts 与 “yellow zone” filter criterion。
- 解释 RMU，以及为什么 WMDP 同时是 evaluation 和 unlearning benchmark。
- 描述 2024-2025 uplift narrative：“mild uplift” -> “on the cusp” -> “insufficient to rule out ASL-3”。
- 区分 novice-relative uplift 与 expert-absolute capability。

## The Problem / 问题

Dual-use capability 是每个 lab frontier safety framework（Lesson 18）下面的测量问题。问题是：model X 是否实质性提升 novice 在 bio、chem 或 cyber 中造成大规模伤害的能力？直接测量（让模型真实地产生 harm）既违法也不合伦理。Proxy measurement 需要一个 benchmark：模型不能拒答（才能得到 honest capability numbers），但问题本身也不能成为 harmful publication。

## The Concept / 概念

### The "yellow zone" / “黄色区域”

这些问题要求 harmful process 附近的 proximate, enabling knowledge，但不是直接 synthesis recipe。“What reagent catalyzes step 4 of [published pathway]?”，而不是 “how do I make [dangerous compound]?” 每个问题由多个领域专家审查，并过滤 ITAR/EAR export-control compliance。

共 4,157 道题：
- Biosecurity：1,520
- Cybersecurity：2,225
- Chemistry：412

Multiple-choice format。模型并没有被要求 assist anything；能力可以在不 eliciting harmful behaviour 的情况下测量。

### RMU — Representation Misdirection for Unlearning / RMU：用于遗忘的表征误导

配套 unlearning method。应用到 LLaMa-2-7B 后，把 WMDP scores 降到接近随机，同时让 MMLU 和其他 general-capability benchmarks 只下降几个百分点。这个 published method 是后续每篇 bio-chem-cyber unlearning paper 的 unlearning baseline。

### The 2024-2025 uplift narrative / 2024-2025 uplift 叙事

三个阶段：

1. **2024 “mild uplift”。** 早期 OpenAI 和 Anthropic Preparedness/RSP evaluations 报告 novices 尝试 bio-adjacent tasks 时相对 internet search 有小优势。公开 framing：frontier models 有帮助，但不比 Google 实质强太多。

2. **2025 年 4 月 “on the cusp”。** OpenAI Preparedness Framework v2 报告模型 “on the cusp of meaningfully helping novices create known biological threats”。这不是 capability claim，而是警告 cusp 已接近。

3. **Anthropic 2025 bioweapon-acquisition trial。** 对 novice participants 的 controlled study，测量 acquisition-phase tasks 上的相对成功。报告 2.53x uplift。不足以 rule out ASL-3（Lesson 18）——意味着 Anthropic Responsible Scaling Policy tier 3 threshold 已达到或接近。

### Novice-relative vs expert-absolute / 新手相对提升与专家绝对能力

关键区别：

- **Novice-relative uplift。** 模型对 non-expert 有多大帮助。是 multiplicative。Relative advantage 高，因为 novices 知识很少；即使 modest information 也有用。
- **Expert-absolute capability。** 模型在最大努力下能产出多少信息。Expert 能比 novice 提取更多。Absolute ceiling 高。

Safety cases（Lesson 18）必须同时覆盖二者：“模型不能给 novice 足够 uplift 去执行” 以及 “expert 无法从模型提取未公开的信息”。

### The measurement pitfall / 测量陷阱

WMDP 是 capability proxy，不是 deployment measurement。WMDP 分数高的模型，在实践中是否可被 novice exploitable 取决于：
- Elicitation resistance（不触发 safety filters 的情况下拿到能力有多难）
- Tacit knowledge（需要 wet-lab skill 而非 information 的能力）
- Execution barriers（procurement、equipment）

Anthropic 2025 bioweapon-acquisition trial 在 WMDP-style capability 之上增加 novice-elicitation layer：它测量实际 task success，而不只是 multiple-choice capability。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 12-16 是 model outputs 上的 attack 与 defense tooling。Lesson 17 是 dual-use capability layer，也就是 frontier safety frameworks（Lesson 18）会评估的 measurement。Lesson 30 用 2026 年当前 cyber/bio/chem/nuclear uplift evidence 收束。

## Build It / 动手构建

本课构建一个 WMDP-shaped evaluation harness：按 domain 统计 mock model 的 scores，再加入简单 unlearning intervention，观察 dual-use score 与 general capability 的 trade-off。

## Use It / 应用它

`code/main.py` 构建一个 toy WMDP-shaped evaluation harness。Mock model 会在按 category 分组的问题上测试，并报告每个 domain 的 scores。一个简单 unlearning intervention（zero out domain-specific representation）会降低 scores；你可以测量它与 general capability 的 trade-off。

## Ship It / 交付它

本课产出 `outputs/skill-wmdp-eval.md`。给定 dual-use capability claim（“our model does not meaningfully help with bioweapons”），它会审计：跑了哪些 benchmarks、evaluation 使用哪条 refusal path（raw completion vs policy-gated），以及是否用 novice-elicitation studies 补充 multiple-choice result。

## Exercises / 练习

1. 运行 `code/main.py`。报告 toy unlearning step 前后的 per-domain accuracy。解释 general-capability trade-off。

2. 给 toy WMDP 添加第四个 domain（例如 radiological）。指定两个 yellow zone 中的 illustrative question types。解释为什么设计这类问题比添加 MMLU-shaped questions 更难。

3. 阅读 WMDP 2024 Section 5（RMU methodology）。Sketch 一个更简单的 unlearning approach（例如 suppress domain content 的 top-k neurons），并描述预期 general-capability cost。

4. Anthropic 2025 bioweapon-acquisition trial 报告 2.53x uplift。描述这个数字可能 upward biased 的两种方式（novice sample size、task fidelity），以及 downward biased 的两种方式（elicitation ceiling、model safety gating）。

5. 阐明 ASL-3 safety case 除了通过 WMDP unlearning 之外还需要什么。至少说出两个 complementary elicitation studies。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| WMDP | “the dual-use benchmark” | yellow zone 中横跨 bio/cyber/chem 的 4,157 道 MCQ |
| Yellow zone | “enabling but not synthesis” | 接近 harmful capability、但不是 synthesis recipe 的 proximate knowledge |
| RMU | “the unlearning baseline” | Representation Misdirection for Unlearning；降低 WMDP scores 并保留 general capability |
| Novice-relative uplift | “how much it helps non-experts” | 对 novice 相对 status-quo internet search 的 multiplicative advantage |
| Expert-absolute capability | “ceiling for experts” | motivated expert 能从模型提取的最大 information |
| Acquisition-phase task | “steps before synthesis” | Procurement、equipment、permits 等 harm pathway 最早部分 |
| ITAR/EAR | “export-control compliance” | 限制发布某些 enabling knowledge 的 legal frameworks |

## Further Reading / 延伸阅读

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark 与 RMU paper。
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — “on the cusp” language。
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold 与 acquisition trial results。
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL。
