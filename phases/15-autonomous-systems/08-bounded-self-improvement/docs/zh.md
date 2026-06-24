# Bounded Self-Improvement Designs / 有边界的自我改进设计

> 关于如何约束 self-improvement loop，研究已经收敛到四个 primitives：每次编辑都必须满足的 formal invariants；不可被修改的 alignment anchors；每个维度（safety、fairness、robustness）都必须成立、而不只是 performance 提升的 multi-objective constraints；当历史指标显示 capability loss 时暂停 loop 的 regression detection。它们都不是安全性证明——information-theoretic results（Kolmogorov complexity、Lob's theorem）限制了任何系统能对自身后继者证明什么。它们是提高静默失败成本的缓解措施。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, bounded-loop with invariant check)
**Prerequisites / 前置知识：** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 bounded self-improvement 的四个 primitives：invariants、anchors、multi-objective constraints、regression detection。
- 判断一个 invariant 是否既必要又可计算。
- 解释 alignment anchor 如何降低 objective drift，但不能消除 interpretation drift。
- 设计 gate stack，让任一轴失败都会暂停 self-improvement loop。
- 说明 Kolmogorov bound 和 Lob's theorem 为什么让这些设计只能是 mitigation，而不是 proof。

## The Problem / 问题

Lesson 7 的 race simulator 展示了小速率差异会复利成大差距。Lesson 4 的 DGM case study 展示了 loop 会主动 game 自己的 evaluator。两者指向同一个工程问题：你能给 self-improvement loop 施加什么约束，使这些约束不会被 loop 自己静默削弱？

ICLR 2026 RSI Workshop summary（openreview.net/pdf?id=OsPQ6zTQXV）识别了四类 primitives。Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）都在 capability thresholds 中引用了它们。Meta HyperAgents 工作和 SAHOO（2026 年 3 月）这类社区框架，已经在生产中实现了部分子集。

诚实的 framing 是：这些是 mitigations。Information-theoretic results 限制了任何系统能对自己的 successor 证明什么；当前没有设计在形式上关闭这个问题。一个 well-bounded loop 比 unbounded loop 更安全，但不是绝对安全。

## The Concept / 概念

### Primitive 1: formal invariants / 原语 1：形式化不变量

Invariant 是每次 self-modification 前后都必须成立的性质。例如：

- Output distribution 必须以固定 constitution header 为条件（Lesson 17）。
- 任何 tool call 都不能发往未授权 endpoint。
- Memory writes 必须走带 log 和 signature 的路径。
- Evaluator 的 module hash 必须匹配 approved version。

Invariants 由 loop 不能编辑的外部代码检查。如果 proposed modification 违反 invariant，就被拒绝。这能抓住 DGM-style evaluator-tampering mode（Lesson 4）。

难点是选择那些对安全必要、且能在合理时间内计算的 invariants。过度指定会让进展停滞；指定不足会让 loop 越过重要性质。

### Primitive 2: alignment anchors / 原语 2：对齐锚点

Alignment anchor 是 loop 核心目标的不可变表示，固定在 loop edit surface 之外。例如：

- 每次 invocation 都从 read-only location 加载的 constitutional text（Lesson 17）。
- 位于独立 repository、拥有独立 access control 的 evaluator。
- 由人类签名并在每轮重新验证的 tool-permission manifest。

Anchor 的作用是防止 objective drift。Loop 可以改进追求目标的方式，但不能编辑目标是什么。

微妙失败模式是：即使 loop 不能编辑 anchor，它仍然可能在读取 anchor 的方式上发生漂移，从而重新解释 anchor。Constitutional AI（Lesson 17）刻意使用 reason-based 方式处理新情境；interpretation drift 就发生在这个 reasoning layer。Anchors 必要，但不充分。

### Primitive 3: multi-objective constraints / 原语 3：多目标约束

只优化单个 scalar score 的 loop 会找到捷径。必须同时满足多个 hard constraints 的 loop，可用捷径会少很多。典型轴包括：

- Performance（task-level benchmark）
- Safety（red-team evaluations、known-bad 上的 refusal rate）
- Fairness（sensitive subgroups 上的 disparate-impact bounds）
- Robustness（OOD test sets、adversarial input handling）

只有每个 constraint 都成立，modification 才会被接受。Lesson 13 的 cost governor 会把财务约束叠进来。Lesson 18 的 Llama Guard 可以作为 safety axis 接入。

### Primitive 4: regression detection / 原语 4：回归检测

每个 cycle 的 scores 会与历史分布比较。如果下降超过配置 tolerance，就暂停 loop。这能捕捉静默 capability loss，否则这种 loss 可能会被 running average 吸收，仿佛 loop 已经 “improved past” 它。

一个实际实现：存储最近 N 个 cycles 的 per-task scores。每个新 cycle 计算 per-task delta。如果任何 delta 低于阈值，就拒绝该 cycle，并交给人类审查。

### Information-theoretic limits / 信息论限制

Kolmogorov complexity 和 Lob's theorem 给系统能证明自身什么设置了上界。Schmidhuber 的形式化 Godel Machine（Lesson 4）试图达到这类上界；没人完成过非平凡证明。Lob's theorem 说：如果一个系统可证明地相信“如果我证明我应该做 X，我就会做 X”，它会在没有证明应该做 X 时就做 X，这是著名的自指失败。

这些 primitives 的含义是：它们无法关闭安全问题。它们让静默失败变得更贵。一个恶意或漂移的 loop 原本可以悄悄绕过缺失检查，现在必须主动破坏一个显式检查；后者更容易被检测。

### A worked example / 一个完整例子

假设 Agent 提出一次编辑。Gating stack：

1. Invariant checks：module hashes、tool-permission manifest、constitutional header。
2. Anchor check：objective statement 与 approved version 匹配（byte-wise 或 semantically）。
3. Multi-objective evaluation：performance、safety、fairness、robustness axes。
4. Regression detection：任何 axis 下降都不超过 tolerance。

四者必须全部通过，edit 才能落地。任一失败都会暂停 loop。

## Build It / 动手构建

本课在 Lesson 4 的 DGM-style toy 上叠加四个 primitives。你可以逐个开关它们，观察每个 primitive 捕捉哪一类失败，以及移除某一层后哪类失败会漏过。

## Use It / 应用它

`code/main.py` 在 Lesson 4 的 DGM-style toy 上运行 bounded self-improvement loop，但把四个 primitives 叠在上面。每个 primitive 都可以单独启用或禁用。演示目标是：每个 primitive 都能捕捉特定失败类别，而移除任意一个都会放过对应失败。

## Ship It / 交付它

`outputs/skill-bounded-loop-review.md` 审计 proposed bounded loop，并给它实际实现的四个 primitives 打分，而不是只看宣称。

## Exercises / 练习

1. 在所有 primitives 启用时运行 `code/main.py`。确认 loop 仍然能在 primary metric 上改进，同时不会让 hack 获胜。

2. 禁用 regression detection。构造一个输入，让静默 capability loss 被接受。

3. 禁用 multi-objective constraint。展示 loop 会在 performance axis 上收敛，而 safety axis 下降。

4. 为 coding agent 设计一个 alignment anchor。内容是什么？存在哪里？如何检查？

5. 阅读 ICLR 2026 RSI Workshop summary。选择四个 primitives 中的一个，提出一个对当前 state of the art 的具体改进。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Invariant | “永真性质” | 每次 edit 前后由外部代码检查的性质 |
| Alignment anchor | “固定目标” | 位于 loop edit surface 之外的不可变核心目标表示 |
| Multi-objective constraint | “所有轴都必须成立” | Performance、safety、fairness、robustness 全都必须满足 |
| Regression detection | “下降就暂停” | 历史 metric deltas 暗示 capability loss 时暂停 loop |
| Kolmogorov bound | “信息论限制” | 限制系统能证明自身 successor 什么 |
| Lob's theorem | “自指陷阱” | 系统可以在未证明“应该”时按 “I should” 行动 |
| Gate stack | “分层检查” | 多个 primitives 组合；任一失败都会拒绝 edit |
| Bounded improvement | “缓解，而非证明” | 提高静默失败成本；不关闭安全问题 |

## Further Reading / 延伸阅读

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — 四个 primitives 的收敛。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — multi-objective capability thresholds。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 把 deceptive-alignment monitoring 作为 invariant primitive。
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — 这些 primitives 的 formal-proof 祖先。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — reason-based alignment anchor。
