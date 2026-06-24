# A/B Testing LLM Features — GrowthBook, Statsig, and the Vibes Problem / LLM 功能 A/B 测试：GrowthBook、Statsig 与感觉主义问题

> 传统 A/B testing 不是为 non-deterministic LLMs 设计的。关键区分：evals 回答“模型能不能做这件事？”A/B tests 回答“用户在不在乎？”两者都需要；靠 vibe checks 上线的时代结束了。2026 年应测试的轴：prompt engineering（wording）、model selection（GPT-4 vs GPT-3.5 vs OSS；accuracy vs cost vs latency）、generation parameters（temperature、top-p）。真实案例：一个 chatbot reward-model variant 带来 +70% conversation length 和 +30% retention；Nextdoor AI subject-line experiments 在 reward-function refinement 后带来 +1% CTR；Khan Academy Khanmigo 沿 latency-vs-math-accuracy 轴迭代。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）提供 sequential testing、CUPED、all-in-one。**GrowthBook** 是 open-source、warehouse-native，支持 Bayesian + Frequentist + Sequential engines、CUPED、SRM checks、Benjamini-Hochberg + Bonferroni corrections。你根据 warehouse-SQL 偏好，以及“被 OpenAI 收购”对组织是否重要来选择。

**类型：** 学习
**语言：** Python（stdlib, toy sequential test simulator）
**前置知识：** 第 17 阶段 · 13（Observability）, 第 17 阶段 · 20（Progressive Deployment）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 evals（“can the model do the job”）与 A/B tests（“do users care”）。
- 枚举三个 testable axes（prompt、model、parameters），并为每个选择 metric。
- 解释 CUPED、sequential testing 和 Benjamini-Hochberg multiple-comparison corrections。
- 根据 warehouse-SQL posture 和 corporate acquisition stance，在 Statsig 与 GrowthBook 之间选择。

## The Problem / 问题

你手工调了 system prompt。感觉更好了。你上线。Conversion 变化落在噪声里。你怪 metric。或者你上线了新模型，conversion 没动：是模型退化了，还是变化太小检测不到？你不知道，因为上线前没有 A/B。

Evals 回答模型在 labeled set 上能不能做任务。它们不回答用户是否更喜欢输出。只有受控 online experiment 能回答，而且必须有足够 power、控制 non-determinism，并修正 multiple comparisons。

## The Concept / 概念

### Evals vs A/B tests / Evals 与 A/B tests

**Evals** — offline、labeled set、judge（rubric 或 LLM-as-judge 或 human）。回答：“在这个固定分布上，output 是否正确 / 有帮助 / 安全？”

**A/B test** — online、live users、randomized。回答：“新变体是否推动了真正重要的 user-level metric？”

两者都需要。Evals 在暴露前抓 regressions；A/B 在暴露后确认 product impact。

### What to test / 测什么

1. **Prompt engineering** — wording、system-prompt structure、examples。Metric：task success、user retention、cost/request。
2. **Model selection** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。Metric：accuracy（task）+ cost/request + latency P99。Multi-objective。
3. **Generation parameters** — temperature、top-p、max_tokens。Metric：task-specific（output diversity vs determinism）。

### CUPED — variance reduction / CUPED：方差降低

Controlled-experiments Using Pre-Experiment Data。在比较 post-period 前，用 pre-period variance 做回归消除。典型 variance reduction：30-70%。相当于免费提高 effective sample size。

实现：Statsig 和 GrowthBook 都支持。

### Sequential testing / Sequential testing

经典 A/B 假设固定 sample size。Sequential tests（“peek-and-decide”）在重复查看下控制 false-positive rate。Always-valid sequential procedures（mSPRT、Howard's confidence sequences）允许你在明确赢家出现时提前停止。

### Multiple-comparison corrections / 多重比较校正

同时跑 20 个 95% confidence 的 A/B tests，随机就会产生一个 false positive。Bonferroni correction 收紧每个 test 的 α；Benjamini-Hochberg 控制 false-discovery rate。GrowthBook 支持两者。

### SRM — sample ratio mismatch / SRM：样本比例不匹配

Assignment hash 把 users 随机到 variants。如果 50/50 split 得到 47/53，说明哪里坏了；SRM check 会标记。两个平台都支持。

### Statsig vs GrowthBook / Statsig 与 GrowthBook

**Statsig**:
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。Hosted、SaaS。
- Sequential testing、CUPED、held-out populations。
- All-in-one：feature flags + experimentation + observability。
- Best fit：团队想要 bundled product，且不介意 OpenAI ownership。

**GrowthBook**:
- Open-source（MIT）；warehouse-native（直接读取 Snowflake/BigQuery/Redshift）。
- 多个 engines：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH corrections。
- Self-host 或 managed cloud。
- Best fit：warehouse-SQL shop，data team 控制 metric layer，想要 OSS。

### Non-determinism complicates power / Non-determinism 会复杂化 power

同一 prompt 会产生不同 outputs。传统 power calculations 假设 IID observations。LLM non-determinism 使 effective sample size 低于 nominal。把所需 sample size 乘以约 1.3-1.5x，作为安全缓冲。

### Real case outcomes / 真实案例结果

- Chatbot reward model variant：+70% conversation length，+30% retention。
- Nextdoor subject lines：reward-function refinement 后 +1% CTR。
- Khan Academy Khanmigo：围绕 latency-vs-math-accuracy 取舍迭代。

### The anti-pattern: shipping on vibes / 反模式：靠感觉上线

每个 senior engineer 都能说出一个因为“感觉更好”而上线、却没有 A/B 的功能。多数都回退了团队没注意数月的产品指标。A/B 是强制函数。

### Numbers you should remember / 你应该记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：open-source MIT；Bayesian + Frequentist + Sequential。
- CUPED variance reduction：30-70%。
- LLM non-determinism → +30-50% sample-size buffer。

## Build It / 动手构建

用 `code/main.py` 模拟 fixed-sample 与 sequential boundaries，调整 baseline conversion、expected lift 和 variance buffer，计算 LLM feature A/B 的样本量。

## Use It / 应用它

`code/main.py` 模拟带 fixed 和 sequential boundaries 的 sequential A/B test。它展示 sequential 如何让你提前停止。

## Ship It / 交付它

本课产出 `outputs/skill-ab-plan.md`。给定 feature change、workload、baseline，它会选择 platform、gates 和 sample size。

## Exercises / 练习

1. 运行 `code/main.py`。baseline conversion 3%、expected lift 5% 时，80% power 需要多少 sample size？
2. 面向 healthcare-regulated on-prem customer，选择 Statsig 还是 GrowthBook？
3. 设计一个测试 GPT-4 vs GPT-3.5 在 cost-per-resolved-ticket 上表现的 A/B。primary metric、guardrail metric、secondary 分别是什么？
4. Canary 通过，但 A/B 显示 conversion -1.2%。要不要 ship？写出 escalation criteria。
5. 对 pre-period 能解释 post variance 60% 的数据应用 CUPED。计算 effective-sample-size boost。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Eval | “offline test” | 对 model capability 做 labeled-set evaluation |
| A/B test | “experiment” | 在线用户 randomized comparison |
| CUPED | “variance reduction” | 用 pre-period regression 降低方差 |
| Sequential test | “peek-ok test” | 允许 early stop 的 always-valid procedure |
| Multiple comparison | “the family error” | 多个 tests 会放大 false positives |
| Bonferroni | “tight correction” | 将 α 除以 tests 数 |
| Benjamini-Hochberg | “BH FDR” | 控制 false-discovery-rate，较不保守 |
| SRM | “bad split” | Sample ratio mismatch；assignment bug |
| Statsig | “OpenAI owned” | Commercial all-in-one，2025 年被收购 |
| GrowthBook | “the OSS one” | MIT warehouse-native platform |
| mSPRT | “sequential probability ratio test” | 经典 sequential procedure |

## Further Reading / 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
