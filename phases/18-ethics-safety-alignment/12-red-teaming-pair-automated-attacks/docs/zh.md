# Red-Teaming: PAIR and Automated Attacks / 红队测试：PAIR 与自动化攻击

> Chao, Robey, Dobriban, Hassani, Pappas, Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——Prompt Automatic Iterative Refinement——是 canonical automated black-box jailbreak。一个带 red-team system prompt 的 attacker LLM 会迭代提出 jailbreaks，攻击尝试和目标响应会积累到自己的 chat history 中，作为 in-context feedback。PAIR 通常在 20 queries 内成功，比 GCG（Zou et al. 的 token-level gradient search）高效几个数量级，并且不需要 white-box access。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 中的标准 baseline，与 GCG、AutoDAN、TAP、Persuasive Adversarial Prompt 并列。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites / 前置知识：** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 PAIR algorithm：attacker system prompt、iterative refinement、in-context feedback。
- 解释当 target 是 black-box 时，PAIR 为什么严格比 GCG 更高效。
- 说出另外四个 automated-attack baselines（GCG、AutoDAN、TAP、PAP），并说明每个的一个区分特征。
- 描述 JailbreakBench 与 HarmBench evaluation protocols，以及各自的 “attack success rate” 含义。

## The Problem / 问题

Red-teaming 曾经是手工活动。少量 expert testers 构造 adversarial prompts，并跟踪哪些有效。这不能 scale：attack success rate 需要统计样本，而 target 会随每次 model release 改变。PAIR 把 red-teaming operationalize 成针对 black-box target 的 optimization problem。

## The Concept / 概念

### PAIR algorithm / PAIR 算法

Inputs：
- Target LLM T（被攻击的模型）。
- Judge LLM J（判断 response 是否是 jailbreak）。
- Attacker LLM A（red-team optimizer）。
- Goal string G：“respond with [harmful instruction].”
- Budget K（通常 20 queries）。

Loop，对 k in 1..K：
1. A 接收 goal G 和目前为止的 (prompt, response) pairs history。
2. A 输出新 prompt p_k。
3. 将 p_k 提交给 T，收到 response r_k。
4. J 根据 goal 给 (p_k, r_k) 打分。
5. 如果 score >= threshold，停止，jailbreak found。
6. 否则把 (p_k, r_k) 追加到 A 的 history，继续。

经验结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat，attack success rate >50%；mean queries to success 在 10-20 区间。

### Why PAIR is efficient / 为什么 PAIR 高效

GCG（Zou et al. 2023）通过 gradient 搜索 adversarial token suffixes；它需要 white-box model access，并产生不可读的 suffixes。PAIR 是 black-box，并生成能跨模型迁移的 natural-language attacks。PAIR 的 in-context feedback 让 attacker 从每次 rejection 中学习；GCG 没有等价机制（每次 token update 都必须重新发现之前的进展）。

### Related automated attacks / 相关自动化攻击

- **GCG（Zou et al. 2023，arXiv:2307.15043）。** token-level gradient search for adversarial suffixes。White-box、transferable、输出不可读字符串。
- **AutoDAN（Liu et al. 2023）。** guided by hierarchical objective 的 evolutionary search over prompts。
- **TAP（Mehrotra et al. 2024）。** Tree-of-attacks with pruning，分支多个 PAIR-style rollouts。
- **PAP（Zeng et al. 2024）。** Persuasive Adversarial Prompts，把 human persuasion techniques 编码成 prompt templates。

### JailbreakBench and HarmBench / JailbreakBench 与 HarmBench

二者都在 2024 年标准化 evaluation：

- JailbreakBench（arXiv:2404.01318）。10 个 OpenAI-policy categories 下 100 个 harmful behaviors。Attack success rate（ASR）为主指标。需要 judge（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika et al. 2024）。7 类 510 个 behaviours，带 semantic 与 functional harm tests。比较 18 种 attacks 对 33 个 models 的效果。

ASR 通常在固定 query budget 下报告。比较 attacks 时必须匹配 budgets；200 queries 下 90% ASR 不能和 20 queries 下 85% ASR 直接比较。

### Reason it matters for 2026 deployments / 为什么对 2026 部署重要

每个 frontier lab 现在都会在发布前对 production models 跑 PAIR 和 TAP。ASR trajectories 会出现在 model cards（Lesson 26）和 safety-case appendices（Lesson 18）中。这个 attack 不奇特，它是标准基础设施。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lesson 12 是 automated-attack foundation。Lesson 13（Many-Shot Jailbreaking）是互补的 length-exploit。Lesson 14（ASCII Art / Visual）是 encoding attack。Lesson 15（Indirect Prompt Injection）是 2026 production attack surface。Lesson 16 覆盖 defensive-tooling counterparts（Llama Guard、Garak、PyRIT）。

## Build It / 动手构建

本课实现一个 mock PAIR loop：toy target、rule-based attacker、judge。你会看到 iterative feedback 如何在 keyword filter 上快速成功，而 semantic filter 为什么更难被同样策略击穿。

## Use It / 应用它

`code/main.py` 构建一个 toy PAIR loop。Target 是一个 mock classifier，会拒绝 “obvious” harmful prompts（keyword-filter）。Attacker 是 rule-based refiner，会尝试 paraphrase、roleplay-framing 和 encoding。Judge 给 response 打分。你会看到 attacker 在 keyword filter 上约 5-15 iterations 成功，而在 semantic filter 上失败。

## Ship It / 交付它

本课产出 `outputs/skill-attack-audit.md`。给定 red-team evaluation report，它会审计：运行了哪些 attacks（PAIR、GCG、TAP、AutoDAN、PAP）、每种 budget 是多少、使用哪个 judge、在哪个 harmful-behaviour set（JailbreakBench、HarmBench、internal）上评估。

## Exercises / 练习

1. 运行 `code/main.py`。测量三种 built-in attacker strategies 的 mean-queries-to-success。解释每种策略利用了 target-defense 的哪个 assumption。

2. 实现第四种 attacker strategy（例如翻译到另一种语言、base64 encoding）。报告它在 keyword-filter target 与 semantic-filter target 上新的 mean-queries-to-success。

3. 阅读 Chao et al. 2023 Figure 5（PAIR vs GCG comparison）。描述两个虽然 PAIR 更高效但仍更适合 GCG 的场景。

4. JailbreakBench 在固定 goal set 上报告 ASR。设计一个额外 metric，用于测量 attack diversity（successful prompts 的 variance）。解释 diversity 为什么对 defense evaluation 重要。

5. TAP（Mehrotra 2024）用 branching + pruning 扩展 PAIR。为 `code/main.py` sketch 一个 TAP-style extension，并描述 computational cost 与 success-rate trade-off。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| PAIR | “automated jailbreak” | Prompt Automatic Iterative Refinement；attacker-LLM + judge-LLM loop |
| GCG | “gradient jailbreak” | white-box token-level gradient search for adversarial suffixes |
| Attack success rate (ASR) | “% jailbreaks at k queries” | 主指标；必须同时报告 query budget 与 judge identity |
| Judge LLM | “the scorer” | 评估 response 是否满足 harmful goal 的 LLM |
| JailbreakBench | “the evaluation” | 带 tagged categories 的 standardized harmful-behaviour set |
| HarmBench | “the broader bench” | 510 behaviours，functional + semantic harm tests |
| TAP | “tree of attacks” | 带 branching + pruning 的 PAIR；compute 更高、ASR 更好 |

## Further Reading / 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR paper，NeurIPS 2023。
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG paper。
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — standardized evaluation。
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — broader evaluation。
