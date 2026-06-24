# Bias and Representational Harm in LLMs / LLM 中的偏见与表征伤害

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed（Computational Linguistics 2024，arXiv:2309.00770）。这篇 2024 foundational survey 区分 representational harms（stereotypes、erasure）与 allocational harms（unequal resource distribution），并把 evaluation metrics 分为 embedding-based、probability-based、generated-text-based。2024-2025 经验研究：An et al.（PNAS Nexus，2025 年 3 月）在 automated resume evaluation 中测量 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 对 20 个 entry-level jobs 的 intersectional gender x race bias。WinoIdentity（COLM 2025，arXiv:2508.07111）引入 uncertainty-based fairness evaluation for intersectional identities。Yu & Ananiadou 2025 识别 MLP layers 中的 gender neurons；Ahsan & Wallace 2025 用 SAEs 揭示 clinical racial bias；Zhou et al. 2024（UniBias）通过操纵 attention heads 做 debiasing。Meta-critique（arXiv:2508.11067）：10 年文献过度关注 binary-gender bias。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, toy embedding-based bias probe)
**Prerequisites / 前置知识：** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 定义 representational harm 与 allocational harm，并各给一个 LLM deployment 中的例子。
- 说出 Gallegos et al. 2024 的三类 evaluation metrics，并描述每类中的一个 metric。
- 描述 intersectionality，以及 WinoIdentity 的 uncertainty-based fairness measurement 为什么能补足 single-axis bias evaluation 的缺口。
- 描述两种 bias 的 mechanistic-interpretability approaches（gender neurons、SAE features、attention-head manipulation）。

## The Problem / 问题

之前课程覆盖 deliberate harm（jailbreaks、scheming）和 safety governance。Bias 是没有意图也会出现的 harm：来自 training data distributions、prompt framing、积累的 design choices。测量和降低它，是一个不同于 adversarial robustness 的方法论挑战。

## The Concept / 概念

### Representational vs allocational / 表征伤害与分配伤害

- **Representational harm。** Stereotypes、erasure、demeaning portrayals。一个 LLM 如果把 nurses 描述成全是女性，就是 representational harm。
- **Allocational harm。** 不平等的 material outcomes。一个 LLM 如果系统性给 Black applicants 的 resumes 打低分，就是 allocational harm。

二者不是一回事。模型可以 “representationally unbiased”（生成多样 portrayals），同时 “allocationally biased”（做出不平等 recommendations）。Evaluations 需要同时测量二者。

### Three evaluation-metric categories (Gallegos et al. 2024) / 三类评估指标

- **Embedding-based。** 在 pre-RLHF embeddings 上做 WEAT-style tests。测量 identity terms 与 attribute terms 的 statistical associations。限制：测的是 representation，不是 behaviour。
- **Probability-based。** stereotype-confirming 与 stereotype-violating completions 的 log-likelihood。Decoder-side measurement。捕捉一部分 behavioural bias。
- **Generated-text-based。** 在 generated text 上做 downstream-task measurement。Resume-scoring、recommendation writing、dialogue。生态有效性最高，也最难复现。

### Intersectionality / 交叉性

只评估 “gender” 会漏掉只在 (gender, race) pairs 上触发的 bias。An et al. 2025 发现 GPT-4o 在 resume scoring 中对 Black women 的惩罚大于 Black men，也大于 white women。Single-axis evaluation 无法捕捉这一点。

WinoIdentity（COLM 2025）引入 uncertainty-based intersectional fairness。它测量的是模型对 outcomes 的 uncertainty 是否在 intersectional identity tuples 之间不同，而不只是 point prediction。这能抓住模型在各组同样错误、但对某些组更不确定的情况；这种差异会产生不同 downstream allocation behaviour。

### Mechanistic approaches / 机制性方法

2024-2025 interpretability work 让 bias 可以被机制性干预：

- **Gender neurons（Yu & Ananiadou 2025）。** 特定 MLP neurons 与 gender-specific behaviours 相关。Ablating 这些 neurons 能降低 gender-gap metrics，capability cost 有限。
- **Clinical racial bias via SAEs（Ahsan & Wallace 2025）。** Sparse autoencoder features 把 internal representation 分解成可解释 dimensions；race-correlated features 可以被识别并压制。
- **UniBias（Zhou et al. 2024）。** Attention-head manipulation for zero-shot debiasing。特定 heads 放大 identity-class sensitivity；zeroing 或 re-weighting 这些 heads 能在不 fine-tuning 的情况下降低 bias。

### The meta-critique / 元批评

10 年文献综述（arXiv:2508.11067，2025）发现，领域过度聚焦 binary-gender bias。其他轴线——disability、religion、migration status、multi-lingual identity——获得的关注少得多。Meta-critique 认为 narrow focus 会因 neglect 而伤害 marginalized groups：一个在 binary gender 上 debiased 得很好的模型，可能在没人检查的维度上偏得很严重。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 20-21 形式化覆盖 bias 与 fairness。Lesson 22 覆盖 privacy。Lesson 23 覆盖 watermarking。这些是 user-harm layer，与前面的 deception/safety layer 互补。

## Build It / 动手构建

本课构建一个 toy embedding-based bias probe：用简单 co-occurrence embedding 测量 identity terms 与 attribute terms 的 WEAT-style distance，注入 bias 后观察 metric 触发，再做简单 debiasing。

## Use It / 应用它

`code/main.py` 构建一个 toy embedding-based bias probe：在简单 co-occurrence embedding 中测量 identity terms 与 attribute terms 的 WEAT-style distance。你可以注入一个 bias 并观察 metric 触发；应用一个简单 debiasing operation 并观察部分恢复。

## Ship It / 交付它

本课产出 `outputs/skill-bias-eval.md`。给定 model card 或 fairness claim，它会审计 evaluation 是否覆盖三类 metric（embedding、probability、generated-text）、intersectionality coverage，以及任何 debiasing intervention 的机制。

## Exercises / 练习

1. 运行 `code/main.py`。报告 debiasing step 前后的 WEAT-style bias scores。解释为什么 metric 不会降到零。

2. 扩展 probe，加入 intersectional test：(gender, race) x (career, family)。报告 cross-axis bias scores。

3. 阅读 An et al. 2025（PNAS Nexus）。识别他们报告的两个 single-axis gender evaluation 会漏掉的 intersectional effects。

4. Yu & Ananiadou 2025 识别 gender neurons。Sketch 一个 falsification experiment，用于区分 “这些 neurons cause gender bias” 与 “这些 neurons 只是 correlate with gender bias”。

5. Meta-critique 认为领域过窄地关注 binary gender。选择一个 under-studied axis，并描述它的 representational-harm measurement protocol。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Representational harm | “stereotypes / erasure” | 对某个群体的 biased portrayal |
| Allocational harm | “unequal decisions” | 对某个群体造成 biased material outcome |
| WEAT | “the embedding test” | Word Embedding Association Test；co-occurrence-based bias probe |
| Intersectionality | “combined identity effects” | 在多个 identity axes 的交叉处出现的 bias |
| Gender neurons | “MLP bias neurons” | activations 与 gender-specific behaviour 相关的特定 neurons |
| SAE feature | “interpretable dimension” | sparse-autoencoder 识别出的 feature，可用于 mechanistic bias analysis |
| UniBias | “attention-head debiasing” | 通过 reweighting attention heads 做 zero-shot debiasing |

## Further Reading / 延伸阅读

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — canonical survey。
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — five-model intersectional study。
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — new benchmark。
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — zero-shot debiasing。
