# Self-Refine and CRITIC: Iterative Output Improvement / Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine (Madaan et al., 2023) 让一个 LLM 扮演三个角色：generate、feedback、refine，并把它们放进循环。7 个任务平均提升 +20 绝对分。CRITIC (Gou et al., 2023) 通过外部工具验证加固 feedback 步骤。到 2026 年，这个模式以 “evaluator-optimizer”（Anthropic）或 guardrail loop（OpenAI Agents SDK）的形式出现在每个框架中。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 Self-Refine 的三个 prompts（generate、feedback、refine），并解释 history 为什么对 refine prompt 很重要。
- 解释 CRITIC 的关键洞察：没有外部 grounding 时，LLM 不擅长自我验证。
- 实现一个 stdlib Self-Refine loop，包含 history 和可选 external verifier。
- 把这个模式映射到 Anthropic 的 “evaluator-optimizer” workflow 和 OpenAI Agents SDK 的 output guardrails。

## The Problem / 问题

一个 Agent 给出了几乎正确的答案。可能是一行代码有语法错误，可能是 summary 太长，也可能是计划漏掉一个 edge case。你想要的是：Agent 批判自己的输出，然后修复它。

Self-Refine 证明了一个模型、没有训练数据、没有 RL 也能做到这一点。但有个前提：LLM 在 hard facts 上的自我验证很差。CRITIC 给这个修复命名：把 verify step 路由到外部工具，比如 search、code interpreter、calculator、test runner。

这两篇论文合起来定义了 2026 年迭代改进的默认形态：generate、verify（能外部验证就外部验证）、refine，直到 verifier 通过。

## The Concept / 概念

### Self-Refine (Madaan et al., NeurIPS 2023) / Self-Refine

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 会看到完整 history，也就是所有先前 outputs 和 critiques，因此不会重复犯错。论文做过 ablation：去掉 history，质量会明显下降。

标题结果：在数学、代码、acronym、对话等 7 个任务上，平均绝对提升 +20，包括 GPT-4。没有训练，没有外部工具，单模型完成。

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024) / CRITIC

Self-Refine 的弱点是 feedback step 由 LLM 自己给自己打分。对于事实性 claim，这不可靠：一个 hallucination 往往在生成它的模型看来也很可信。CRITIC 把 `feedback(task, output)` 替换成 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实 claim 的搜索引擎。
- 用于代码正确性的 code interpreter。
- 用于算术的 calculator。
- 领域特定 verifier（unit tests、type checkers、linters）。

verifier 产出基于工具结果的结构化 critique。refiner 再基于 critique 改写。

标题结果：CRITIC 在事实任务上超过 Self-Refine，因为 critique 有 grounding。对于没有 external verifiers 的任务（creative writing、formatting），CRITIC 退化为 Self-Refine。

### The stop condition / 停止条件

两种常见形状：

1. **Verifier passes。** 外部测试成功。只要可用就优先采用（unit tests、type checker、guardrail assertion）。
2. **No feedback issued。** 模型说 “the output is fine”。更便宜但不可靠；必须搭配 max-iteration cap。

2026 年默认组合是：“Stop if verifier passes OR model says fine AND iterations >= 2 OR iterations >= max_iterations。”

### Evaluator-Optimizer (Anthropic, 2024) / Evaluator-Optimizer

Anthropic 2024 年 12 月文章把它命名为五种 workflow patterns 之一。两个角色：

- Evaluator：给输出打分并产出 critique。
- Optimizer：基于 critique 修订输出。

循环直到 evaluator 通过。这就是 Anthropic 语境中的 Self-Refine / CRITIC。Anthropic 补充的关键工程细节是：evaluator 和 optimizer prompts 应该结构上明显不同，否则模型会自我背书。

### OpenAI Agents SDK output guardrails / OpenAI Agents SDK 输出 guardrails

OpenAI Agents SDK 把这个模式作为 “output guardrails” 提供。guardrail 是在 Agent 产生最终输出后运行的 validator。如果 guardrail 触发（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，Agent 可以 retry。Guardrails 可以调用工具（CRITIC-style），也可以是纯函数（Self-Refine-style）。

### 2026 pitfalls / 2026 年常见坑

- **Rubber-stamp loops。** 同一模型用同样 prompt 风格做生成和 critique，会收敛到 “looks good to me”。使用结构不同的 prompts，或让更小更便宜的模型做 critique。
- **Over-refinement。** 每次 refine 都增加延迟和 token。通常预算 1-3 次；之后升级到 human review。
- **CRITIC on trivial tasks。** 没有 external verifier 时，CRITIC 退化为 Self-Refine；不要为一个空 verifier 支付延迟。

## Build It / 动手构建

`code/main.py` 在 toy 任务上实现 Self-Refine 和 CRITIC：给定 topic，生成一个短 bullet list。verifier 检查格式（3 个 bullets，每条少于 60 字符）。CRITIC 增加一个外部 “fact verifier”，惩罚已知 hallucinations。

组件：

- `generate`：脚本化 producer。
- `feedback`：LLM-style self-critique。
- `verify_external`：CRITIC-style grounded verifier。
- `refine`：基于 history 改写输出。
- Stop condition：verifier passes 或最多 4 次 iterations。

运行：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行。CRITIC 能抓到 Self-Refine 漏掉的事实错误，因为 external verifier 有 self-critic 没有的 grounding。

## Use It / 应用它

Anthropic 的 evaluator-optimizer 就是这个模式的 Claude-friendly 说法。OpenAI Agents SDK 的 output guardrails 是 CRITIC-shaped（guardrails 可以调用工具）。LangGraph 提供的 reflection node 读起来像 Self-Refine。Google 的 Gemini 2.5 Computer Use 加了 per-step safety evaluator，它是 CRITIC 的变体：每个 action 在提交前都被验证。

## Ship It / 交付它

`outputs/skill-refine-loop.md` 会根据任务形状、verifier availability 和 iteration budget 配置 evaluator-optimizer loop。它会输出 generator、evaluator/verifier、optimizer 的 prompts，以及 stop policy。

## Exercises / 练习

1. 用 max_iterations=1 运行 toy。CRITIC 还有帮助吗？
2. 把 external verifier 换成有噪声的版本（随机 30% false positives）。循环会怎么做？这就是 2026 年多数 guardrail stacks 的现实。
3. 实现一个 “generator-critic on different models” 变体：大模型生成，小模型 critique。它能超过 same-model 吗？
4. 阅读 CRITIC Section 3 (arXiv:2305.11738 v4)。说出三类 verification tools，并各给一个例子。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier 角色。SDK 哪些地方做错了，哪些地方做对了？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | “LLM that fixes itself” | 单模型中的 Generate -> feedback -> refine loop，并带 history |
| CRITIC | “Tool-grounded verification” | 用外部 verifier（search、code、calc、tests）替换 feedback |
| Evaluator-Optimizer | “Anthropic workflow pattern” | evaluator 打分，optimizer 修订，循环到收敛 |
| Output guardrail | “Post-hoc check” | OpenAI Agents SDK 在 Agent 输出后运行的 validator |
| Verify step | “Critique phase” | 承重决策：是 grounded，还是 self-rated |
| Refine history | “What the model already tried” | 先前 outputs + critiques 会放进 refine prompt；去掉后质量崩 |
| Rubber-stamp loop | “Self-agreement failure” | 同风格 prompt 的 critique 返回 “looks good”；用结构不同的 prompts 修复 |
| Stop condition | “Convergence test” | Verifier passes OR no feedback AND iteration cap；不要只有单一条件 |

## Further Reading / 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 标准论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — tool-grounded verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer workflow pattern
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — output guardrails as CRITIC-shaped verifiers
