# Theory of Mind and Emergent Coordination / Theory of Mind 与涌现协同

> Li et al.（arXiv:2310.10701）显示，在一个合作文本游戏中，LLM Agent 会表现出 **emergent high-order Theory of Mind**（ToM）：推理另一个 Agent 对第三个 Agent belief 的 belief。但它们在 long-horizon planning 上因为 context management 和 hallucination 失败。Riedl（arXiv:2510.05174）测量 population 中的 higher-order synergy，发现 **只有** ToM-prompt condition 会产生 identity-linked differentiation 和 goal-directed complementarity；低容量 LLM 只显示 spurious emergence。也就是说，协同涌现是 prompt-conditional 且 model-dependent 的，不是免费的。本课实现一个最小 ToM-aware Agent，在有/无 ToM prompting 的合作任务上运行，并按 Riedl 2025 protocol 测量 coordination delta。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 07（Society of Mind and Debate）, 第 16 阶段 · 17（Generative Agents）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 zeroth-order、first-order、second-order Theory of Mind
- 解释 coordination illusion 与可测 coordination 的差别
- 构建显式 ToM state，并测量 duplication rate、completion rate 等协同指标
- 为“emergent coordination” claim 设计 control condition、统计检验和失败日志

## The Problem / 问题

多 Agent coordination 经常看起来很神奇：Agent 分工、预判彼此、避免重复。通常这种“涌现”是 prompt engineering 的产物：有人告诉 Agent “coordinate”。移除 prompt，协同就消失。

Riedl 2025 的发现更严格：在受控条件下，只有当 Agent 被提示去推理 **other agents' minds**（ToM）时，coordination 才会出现。没有 ToM prompt，即使强模型也只能产生经不起统计控制的 coordination pattern。这对生产很重要：团队上线的“多 Agent 协同”功能可能依赖 prompt 且脆弱。

本课把 ToM 当成具体能力（reasoning about beliefs about beliefs），实现一个最小 ToM-aware Agent，并测量真正 coordination 与 prompt dressing 的区别。

## The Concept / 概念

### What ToM means / ToM 是什么

发展心理学里，3 岁儿童认为别人的内心世界和自己一样。5 岁儿童理解别人会有不同 belief。7 岁儿童能推理 beliefs about beliefs（“她认为我以为球在杯子下面”）。这些是 zeroth、first、second-order ToM。

对 LLM Agent，ToM orders 对应：

- **Zeroth-order:** 没有他人模型。Agent 只按自己的 observations 行动。
- **First-order:** Agent 拥有每个其他 Agent 的 belief model。“Alice believes X.”
- **Second-order:** Agent 建模递归 belief。“Alice believes that Bob believes X.”

Li et al. 2023 发现，在 cooperative games 中，LLM Agent 会涌现 first- 和 second-order ToM，但在 long horizon 和不可靠 communication 下退化。

### The Sally-Anne test, in brief / Sally-Anne 测试

1985 年的 false-belief test：Sally 把 marble 放进篮子 A，然后离开。Anne 把它移到篮子 B。Sally 回来会去哪找？具备 first-order ToM 的孩子会说篮子 A（Sally 的 belief 与现实不同）。没有 ToM 的孩子会说篮子 B。

GPT-4 时代的 LLM 在直接提问的 Sally-Anne-style tests 上能通过。但当 narrative 很长、场景多次变化，或问题被间接表达时会失败。这就是 2026 年生产 LLM 的实际 ToM 状态。

### Riedl's coordination measurement / Riedl 的协同测量

Riedl（arXiv:2510.05174）构建 population-scale test：N 个 Agent，一个 cooperative objective，可变 prompt conditions。测量：

1. **Identity-linked differentiation.** Agent 是否随时间发展出稳定 role distinctions？
2. **Goal-directed complementarity.** Agent action 是否互补（不同子任务），而不是重复？
3. **Higher-order synergy.** 统计测量：群体是否达成任何子集都无法达成的效果。

结果：只有 ToM prompt condition 下，三个指标才都高于 baseline。没有 ToM prompt，中等容量模型的指标接近 chance。大模型没有显式 ToM prompt 也有一些 coordination，但效果小于显式 prompting。

### The coordination illusion / 协同幻觉

没有统计控制时，demo 里的 “emergent coordination” 往往反映：

- prompt engineering 内置了 coordination（system prompts 写着“work together”）。
- observer bias（我们看到想看到的 pattern）。
- post-hoc selection of successful runs。

声称“涌现协同”的生产系统如果没有可测信号，应该按营销看待。先测量，再声称。

### A minimal ToM-aware agent / 最小 ToM-aware Agent

结构：

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 属性就是 ToM state。first-order ToM 只保留一层。second-order 增加 `other_models[i][other_models_of_j]`，即“我认为 Agent i 认为 Agent j 相信什么”。

### Why long-horizon hurts / 为什么长程会伤害

Li et al. 记录：context limits 会让 Agent 忘记哪个 belief 属于谁。hallucination 会把 false beliefs 加进 other-agent models。两者会产生 “I thought he thought X” 错误，并随时间复合。

论文和 2024-2026 后续工作给出的缓解：

- **Explicit ToM state in the prompt.** 结构化格式：`{agent_id: belief_list}`。强制 retrieval 保留 identity-belief binding。
- **Shorter reasoning chains.** 每轮更少 ToM updates，减少 hallucination 复合。
- **External ToM store.** 在 LLM context 外维护 model；每轮只注入相关部分。

### Where ToM fails in production / 生产中 ToM 失败的位置

- **Adversarial settings.** 具备好 ToM 的 Agent 更容易被操控（对手可以建模它如何建模自己，再利用）。
- **Heterogeneous teams.** 模型不同后，一个 opponent 上有效的 ToM model 不泛化到另一个。
- **Ground-truth-dependent tasks.** ToM 关注 beliefs；如果正确性取决于事实，ToM 可能分散注意。

### The coordination you can actually measure / 可实际测量的协同

三个实用信号可以区分真实 coordination 和 prompt-dressed appearance：

1. **Complementarity over time.** 多轮任务里，Agent action 是否覆盖互不重叠的子任务？
2. **Anticipation.** Agent A 在 T+1 的行动是否依赖对 B 在 T+2 行动的预测，并且预测后来正确？
3. **Correction.** A 在 T 轮误读 B 的 belief 后，是否在 T+2 前纠正？

这些都能在带日志的多 Agent 系统中测量。它们是“coordination”叙事的实质版。

## Build It / 动手构建

`code/main.py` 实现：

- `ToMAgent` — 跟踪 own beliefs 和 per-other-agent belief models。
- 合作任务：三个 Agent 必须从三个 box 收集三个 token；每个 box 最多容纳一个 token。Agent 不能通信，只能从彼此 action 推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（一层 belief model 的 ToM）。
- 200 个随机 trials 上的测量：completion rate、duplication rate（两个 Agent 选择同一 box）、average turns to completion。

运行：

```
python3 code/main.py
```

预期输出：zeroth-order Agent 的重复努力约 35%，10 轮内完成约 60%。first-order ToM Agent 重复约 5%，完成约 95%。差值就是可测 coordination effect。

## Use It / 应用它

`outputs/skill-tom-auditor.md` 是一个 skill，用来审计多 Agent 系统对 “emergent coordination” 的声明。检查 prompt dressing、相对 control 的统计显著性，以及 measured complementarity。

## Ship It / 交付它

Coordination claims checklist：

- **Control condition.** 一个没有 coordination prompt 的系统版本。两者都测。
- **Statistical test.** system 与 control 在指标上的差异是否达到 `p < 0.05`？
- **Complementarity measure.** 随时间的 action-disjointness，而不只是最终成功。
- **Failure-case log.** 当 Agent miscoordinate 时，ToM state 长什么样？
- **Model-capacity disclosure.** 如果效果在小模型上消失，要说明。

## Exercises / 练习

1. 运行 `code/main.py`。确认 first-order ToM 把 duplication rate 降低约 7 倍。扩展到 5 Agent 和 5 boxes 时，差距还在吗？
2. 实现 second-order ToM（Agent A 建模 B 对 C 的 belief）。它相比 first-order 有提升吗？在哪类任务上？
3. 向 ToM state 注入 **hallucination**：每轮随机翻转一个 belief。这会让 first-order performance 降低多少？
4. 阅读 Li et al.（arXiv:2310.10701）。复现 “long-horizon degradation”：turns 从 10 增到 30 时，first-order ToM performance 如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在 simulation logs 上实现 higher-order synergy statistic。没有 ToM prompt condition 时效果存在吗？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Theory of Mind | “理解他人心智” | 建模另一个 Agent belief 的能力，按 order 分级（0、1、2+）。 |
| Sally-Anne test | “False-belief test” | 1985 发展心理学测试；LLM 能通过简单版本，复杂版本会失败。 |
| First-order ToM | “A believes X” | 建模另一个 Agent 对事实的 belief。 |
| Second-order ToM | “A believes B believes X” | 更深一层的递归建模。 |
| Identity-linked differentiation | “稳定角色” | Riedl 指标：角色持续存在，而不是随机变化。 |
| Goal-directed complementarity | “动作互补” | Agent 选择不同子任务，而不是同一个。 |
| Higher-order synergy | “群体超过任何子集” | Riedl 对真实 coordination 的统计测量。 |
| Coordination illusion | “看起来协同” | prompt-dressed appearance，没有可测信号。 |

## Further Reading / 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — cooperative games 中的 emergent ToM 与 long-horizon failure modes
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — population-scale measurement；ToM prompting 是承重条件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — ToM 概念的 1978 年起点
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne paper（1985）
