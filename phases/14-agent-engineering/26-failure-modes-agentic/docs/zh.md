# Failure Modes: Why Agents Break / 失败模式：Agent 为什么会坏

> MASFT（Berkeley, 2025）把 14 种 multi-agent failure modes 归为 3 类。Microsoft 的 Taxonomy 记录了已有 AI failures 如何在 agentic settings 中被放大。行业现场数据收敛到五种反复出现的模式：hallucinated actions、scope creep、cascading errors、context loss、tool misuse。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 05（Self-Refine and CRITIC）, 第 14 阶段 · 24（Observability）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 MASFT 的三类 failure categories，并在每类中至少举出四个具体 modes。
- 解释为什么 agentic failure 会放大已有 AI failure modes（bias、hallucination）。
- 描述五种行业反复出现的 modes 及其 mitigations。
- 实现一个 stdlib detector，为 agent traces 标注 failure-mode labels。

## The Problem / 问题

团队上线的 agents 往往在 90% traces 上表现正常。剩下 10% failure 不是随机噪声，而会落入少数反复出现的类别。只要你能命名它们，就能监控它们，并修复它们。

## The Concept / 概念

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy。14 种 failure modes 聚成 3 类。Inter-annotator Cohen's Kappa 为 0.88，说明这些类别可以被可靠地区分。

核心主张：failures 是 multi-agent systems 中的根本设计缺陷，不是靠更强 base models 就能消除的 LLM limitations。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 已有 AI failures（bias、hallucination、data leakage）会在 agentic settings 中被放大。
- autonomy 会带来新 failures：unintended action at scale、tool misuse、mission drift。
- 这份 whitepaper 是 agentic products 的 risk register。

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- Failures 来自 orchestration、internal state evolution 和 environment interaction。
- 不只是 “bad code” 或 “bad model output”。

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

两种主要表现：

1. **Instruction-following Deviation** — agent 没有遵守 system prompt。
2. **Long-range Contextual Misuse** — agent 忘记或误用较早 turns 中的 context。

Sub-intention errors：Omission（漏步骤）、Redundancy（重复步骤）、Disorder（步骤乱序）。

### The five industry-recurring modes / 五种行业反复出现的模式

Arize、Galileo、NimbleBrain 在 2024-2026 年的 field analyses 收敛到：

1. **Hallucinated actions.** Agent 调用不存在的工具，或编造 arguments。
2. **Scope creep.** Agent 把任务扩展到用户请求之外（创建额外 PRs、发送额外 emails）。
3. **Cascading errors.** 一次错误调用触发下游连锁影响。一个 phantom SKU hallucination 触发四次 API calls，最终变成 multi-system incident。
4. **Context loss.** Long-horizon tasks 忘记早期 turns 中的 constraints。
5. **Tool misuse.** 用错误 arguments 调用正确工具，或直接用了错误工具。

Cascading 是最致命的。Agents 常常分不清 “I failed” 和 “the task is impossible”，并且会在 400 errors 后 hallucinate 一个 success message 来闭环。

### Mitigation: gates at every step / 缓解：每一步都设 gate

在 reasoning chain 的每一步设置 automated verification gates，针对 environment state 检查 factual grounding。具体包括：

- Per-step safety classifier（Lesson 21）。
- Tool-call argument validation（Lesson 06）。
- 将 retrieved content 与 known facts cross-check（Lesson 05, CRITIC）。
- 通过重新探测状态来检测 success hallucination（文件真的创建了吗？）。

### Where failure monitoring goes wrong / Failure monitoring 容易出错的地方

- **Tagging only crashes.** 大多数 agent failures 会产生看起来合法的输出。需要 content-level checks。
- **No baseline.** Drift detection 需要 last-known-good；没有它，你无法判断 “this is getting worse”。
- **Over-alerting.** 每个 failure 都发 page。应 cluster 并 rate-limit。

## Build It / 动手构建

`code/main.py` 实现一个 stdlib failure-mode tagger：

- 一个覆盖五种 modes 的 synthetic trace dataset。
- 每个 mode 的 detector functions（基于 tool calls、outputs、repeat actions 的 signature patterns）。
- 一个 tagger，给每条 trace 打 label，并报告 mode distribution。

运行：

```
python3 code/main.py
```

输出：per-trace labels + aggregate distribution，这是 Phoenix trace clustering 暴露内容的廉价复现。

## Use It / 应用它

- **Phoenix** 用于生产 drift clustering（Lesson 24）。
- **Langfuse** 用于 session replay + annotation。
- **Custom** 用于 observability platform 无法检测的 domain-specific signatures。

## Ship It / 交付它

`outputs/skill-failure-detector.md` 会生成面向你 domain 的 failure-mode detectors，并接入 trace store。

## Exercises / 练习

1. 增加一个 “success hallucination” detector：agent 返回成功，但目标状态没有变化。
2. 标注你构建过的产品中的 100 条真实 traces。哪种 mode 占主导？修复它的成本是多少？
3. 实现一个 “cascade radius” metric：给定第 N 步的 failure，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种 failure modes。选出三个适用于你产品的模式，并编写 detectors。
5. 把一个 detector 接入 CI job：如果 >=5% traces 标记到某种 mode，就 fail build。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MASFT | “Multi-agent failure taxonomy” | Berkeley 的 14-mode categorization |
| Cascading error | “Ripple failure” | 一个早期错误沿 N 个步骤传播 |
| Context loss | “Forgot the constraint” | long-horizon turn 丢掉早期事实 |
| Tool misuse | “Wrong tool / wrong args” | 调用有效，但 invocation 错了 |
| Success hallucination | “Faked completion” | Agent 在 400 后声称成功；状态未变 |
| Scope creep | “Overreach” | Agent 做了超出请求范围的事 |
| Instruction-following deviation | “Disobedience” | 忽略 system prompt 或 user constraint |
| Sub-intention errors | “Plan bugs” | plan execution 中的 omission、redundancy、disorder |

## Further Reading / 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure modes, 3 categories
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risk register
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering in practice
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when simpler patterns avoid modes entirely
