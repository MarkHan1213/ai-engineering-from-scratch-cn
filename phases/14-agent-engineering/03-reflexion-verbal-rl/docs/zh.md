# Reflexion: Verbal Reinforcement Learning / Reflexion：语言化强化学习

> 基于梯度的 RL 要修复一个失败模式，通常需要上千次试验和 GPU 集群。Reflexion (Shinn et al., NeurIPS 2023) 用自然语言做这件事：每次试验失败后，Agent 写一条反思，把它存入 episodic memory，并在下一次试验时把这段记忆放进上下文。Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md learnings、pro-workflow 的 learn-rule，背后都是这个模式。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 Reflexion 的三个组件：Actor、Evaluator、Self-Reflector，并说明 episodic memory 的作用。
- 实现一个 stdlib Reflexion loop，包含 binary evaluator、reflection buffer 和全新的 retry。
- 根据任务选择 scalar、heuristic 或 self-evaluated feedback source。
- 解释为什么 verbal reinforcement 能抓住基于梯度的 RL 需要上千次试验才能修复的错误。

## The Problem / 问题

一个 Agent 任务失败了。在标准 RL 中，你会继续跑上千次试验、计算梯度、更新权重。这昂贵、缓慢，而且多数生产 Agent 没有为每个失败做训练的预算。

Reflexion (Shinn et al., arXiv:2303.11366) 提出另一个问题：如果 Agent 只是思考一下自己为什么失败，然后带着这条思考再试一次，会怎样？没有权重更新。没有梯度。只有跨试验保存的自然语言。

结果是：在 ALFWorld 上，它超过 ReAct 和其他非微调 baseline；在 HotpotQA 上，它优于 ReAct；在代码生成（HumanEval/MBPP）上，它达到当时 SOTA。全程没有一次梯度更新。

## The Concept / 概念

### The three components / 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次 trial 由 Actor 运行。Evaluator 给它评分。如果分数低，Self-Reflector 产出一条 reflection，例如“I picked the wrong tool because I misread the question as asking about X when it was asking about Y”。这条 reflection 放入 episodic memory。下一次 trial 重新开始，但能看到这条 reflection。

### Three evaluator types / 三类 evaluator

1. **Scalar**：外部 binary signal。ALFWorld 成功或失败。HumanEval tests 通过或失败。最简单，信号最强。
2. **Heuristic**：预定义失败特征。“如果 Agent 连续两次产生同一 action，标记为 stuck。”“如果 trajectory 超过 50 步，标记为低效。”
3. **Self-evaluated**：LLM 给自己的 trajectory 打分。没有 ground truth 时需要它。信号较弱；最好和 tool-grounded verification 搭配（Lesson 05 — CRITIC）。

2026 年默认做法通常是混合：有 scalar 就用 scalar，没有时用 self-eval，再用 heuristics 当安全护栏。

### Why this generalizes / 为什么它可以泛化

Reflexion 与其说是新算法，不如说是一个被命名的模式。几乎每个生产里的 “self-healing” Agent 都在跑某种变体：

- Letta 的 sleep-time compute（Lesson 08）：另一个 Agent 反思过去对话，并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / “save memory” 模式：把 reflection 捕获成 learnings，放到未来 sessions 前面。
- pro-workflow 的 `/learn-rule` command：把纠正捕获成显式规则。
- LangGraph 的 reflection nodes：一个节点评分输出，如果需要则路由到 refine。

它们都来自同一个洞察：自然语言已经足够丰富，可以把“我从失败里学到了什么”跨 run 传递。

### When it works and when it does not / 何时有效，何时无效

Reflexion 适用于：

- 有清晰失败信号（test failure、tool error、wrong answer）。
- 任务类别可复现（同一类问题未来还会出现）。
- 轨迹仍有改进空间（action budget 足够）。

Reflexion 不适用于：

- Agent 第一次就已经成功。
- 失败来自外部（network down、tool broken），“网络挂了”的反思不会帮助未来 run。
- reflection 变成迷信，把一次性 flaky run 编成一个持久叙事。

2026 年常见坑是 memory rot。Reflections 会积累；其中一些过期或错误；episodic buffer 越长，rerun 越慢。缓解方式包括 periodic compaction（Lesson 06）、reflection TTL，或单独的 sleep-time cleanup agent（Letta）。

```figure
react-trace
```

## Build It / 动手构建

`code/main.py` 在一个 toy puzzle 上实现 Reflexion：生成一个 3 元素列表，使其和等于 target。Actor 产出候选列表；Evaluator 检查总和；Self-Reflector 写一行失败诊断。这条 reflection 会进入下一次 trial 的 episodic memory。

组件：

- `Actor`：脚本化策略，看到 reflections 后会变好。
- `Evaluator.binary()`：对 target sum 做 pass/fail。
- `SelfReflector`：生成一行失败诊断。
- `EpisodicMemory`：带 TTL 语义的 bounded list。

运行：

```
python3 code/main.py
```

trace 展示三次 trial。Trial 1 失败并存入 reflection；trial 2 看到 reflection 后有改进但仍失败；trial 3 成功。和 baseline（没有 reflection）比较，它会一直卡在 trial 1 的答案上。

## Use It / 应用它

LangGraph 把 reflection 作为节点模式提供。Claude Code 的 `/memory` command 和 pro-workflow 的 `/learn-rule` 把 episodic buffer 外置为 markdown 文件。Letta 的 sleep-time compute 在空闲时间运行 Self-Reflector，使主 Agent 保持低延迟。OpenAI Agents SDK 不直接提供 Reflexion；你可以用自定义 Guardrail 按分数拒绝 trajectory，并用跨 run 存活的 memory `Session` 来实现。

## Ship It / 交付它

`outputs/skill-reflexion-buffer.md` 创建并维护一个 episodic buffer，包含 reflection capture、TTL 和 deduplication。给定任务类别和一次失败，它会产出真正能帮助下一次 trial 的 reflection，而不是泛泛的 “be more careful”。

## Exercises / 练习

1. 从 binary evaluator 改成返回距离度量的 scalar evaluator（离 target 多远）。它收敛得更快吗？
2. 给 reflections 增加 10 次 trial 的 TTL。更老的 reflections 在那之后是帮忙还是添乱？
3. 实现 heuristic evaluator：如果相同 action 重复出现，就标记 trial stuck。它和 Self-Reflector 如何交互？
4. 用一个会忽略 reflections 的 adversarial Actor 运行 Reflexion。最小需要怎样的 reflection prompt engineering 才能逼 Actor 注意到它们？
5. 阅读 Reflexion 论文关于 AlfWorld 的 Section 4。概念性复现 130% success-rate improvement：相对 vanilla ReAct 的关键差异是什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “Self-correction” | Shinn et al. 2023：Actor、Evaluator、Self-Reflector 加 episodic memory |
| Verbal reinforcement | “Learning without gradients” | 把自然语言 reflection 加到下一次 trial prompt 前面 |
| Episodic memory | “Per-task reflections” | 某一任务类别的先前 reflections bounded buffer |
| Scalar evaluator | “Binary success signal” | 来自 ground truth 的 pass/fail 或 numeric score |
| Heuristic evaluator | “Pattern-based detector” | 预定义失败特征，例如 stuck-loop、too-many-steps |
| Self-evaluator | “LLM-as-judge on own trace” | 没有 ground truth 时的低信号 fallback，应搭配 tool-grounded verification |
| Memory rot | “Stale reflections” | episodic buffer 填满过期条目；用 compaction / TTL 修复 |
| Sleep-time reflection | “Async self-reflection” | 在 hot path 之外运行 Self-Reflector，让主 Agent 保持快速 |

## Further Reading / 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 标准论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 生产中的异步 reflection
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 把 episodic buffer 当作 context 的一部分管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection node pattern
