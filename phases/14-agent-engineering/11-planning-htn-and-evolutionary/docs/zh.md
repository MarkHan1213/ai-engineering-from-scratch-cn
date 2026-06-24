# Planning with HTN and Evolutionary Search / 用 HTN 与 Evolutionary Search 做规划

> Symbolic planning 处理“计划必须可证明正确”的场景。Evolutionary code search 处理“fitness function 可机器检查”的场景。ChatHTN (2025) 和 AlphaEvolve (2025) 展示了 LLM 与这两类系统结合后分别能解锁什么。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 Hierarchical Task Networks：tasks、methods、operators、preconditions、effects。
- 描述 ChatHTN 的 hybrid loop：symbolic search 加 LLM fallback decomposition。
- 解释 AlphaEvolve 的 evolutionary loop，以及它为什么只在有 programmatic evaluator 时成立。
- 用 stdlib 实现一个 toy HTN planner 和一个 toy evolutionary search。

## The Problem / 问题

ReWOO（Lesson 02）、Plan-and-Execute 和 ReAct 覆盖了多数 Agent planning。但有两类场景它们处理得不好：

1. **Plans with provable correctness。** 调度、航线规划、合规 workflow：计划必须由构造保证 sound。一个表达流畅但偶尔幻觉步骤的 LLM plan 不可接受。
2. **Optimizations with a machine-checkable fitness function。** 矩阵乘法、调度启发式、compiler passes：目标不是“正确计划”，而是“最优计划”。

HTN planning 和 AlphaEvolve 解决的是两个不同问题。两者都把 LLM 当放大器，而不是替代品。

## The Concept / 概念

### Hierarchical Task Networks / 分层任务网络

一个 HTN 包含：

- **Tasks**：compound（需要分解）和 primitive（可直接执行）。
- **Methods**：把 compound task 分解为 subtasks 的方式，并带 preconditions。
- **Operators**：带 preconditions 和 effects 的 primitive actions。
- **State**：facts 集合。

Planning：给定 goal task 和 initial state，找到一个 primitive operators 序列，使得每一步 preconditions 按顺序满足。

HTN 早于 LLM 很多年，但仍是可证明正确计划的参考模型。

### ChatHTN (Gopalakrishnan et al., 2025) / ChatHTN

ChatHTN (arXiv:2505.11814) 把 symbolic HTN 和 LLM queries 交错：

1. 先尝试用现有 methods 分解当前 compound task。
2. 如果没有 method 适用，就问 LLM：“在 state `s` 下你会如何分解 `task`？”
3. 把 LLM response 翻译为候选 subtasks。
4. 根据 operator schema 校验；拒绝 invalid decompositions。
5. 递归。

论文中心论点是：生成的每个 plan 都可证明 sound，因为 LLM suggestions 只作为候选 decompositions 进入，从不直接编辑计划。symbolic layer 持有正确性；LLM 扩展 method library。

Online method learning（OpenReview `gwYEDY9j2x`, 2025 follow-up）增加了一个 learner，用 regression 泛化 LLM-produced decompositions，可将 LLM query 频率最多降低 75%。

### AlphaEvolve (Novikov et al., 2025) / AlphaEvolve

AlphaEvolve (arXiv:2506.13131, DeepMind, June 2025) 是另一种东西：由 Gemini 2.0 Flash / Pro ensemble 编排的 evolutionary code search。

循环：

1. 从 seed program + programmatic evaluator（返回 fitness score）开始。
2. LLM ensemble 提出 mutations。
3. 用 evaluator 运行 mutations。
4. 保留最优者，再继续 mutate。

公开成果：

- 56 年来首次改进 Strassen 的 4x4 complex matrix multiplication（48 次 scalar multiplications）。
- 通过 Borg scheduling heuristic 找回 0.7% Google compute。
- 在 frontier workload 上让 FlashAttention 加速 32%。

硬约束：fitness function 必须可机器检查。对 prose answers 做 evolutionary search 不会收敛。

### When to use which / 什么时候用哪种

| Problem class | Use | Why |
|---------------|-----|-----|
| Scheduling with hard constraints | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, no formal guarantees |
| Code improvement with tests | AlphaEvolve | Tests are the evaluator |
| Policy-bound automation | HTN | Preconditions encode policy |

### Where this pattern goes wrong / 这个模式在哪里会出错

- **HTN without operators。** 没有 precondition / effect schemas，soundness claim 就崩了。ChatHTN 的 “LLM suggests decomposition” 依赖 schema 拒绝 invalid moves。
- **AlphaEvolve without a real evaluator。** “问 LLM 代码是否更好” 不是 fitness function。evaluator 必须 deterministic 且 fast。
- **Over-engineering。** 多数 Agent tasks 不需要这两者。先用 ReAct 或 ReWOO。

## Build It / 动手构建

`code/main.py` 实现两个 toy：

- 一个 stdlib HTN planner，包含 operators、methods、preconditions、effects，以及在没有 method 匹配 compound task 时启动的 `LLMFallback`。“LLM” 是脚本化 decomposer，因此 planner 可离线运行。
- 一个对 arithmetic programs 做 stdlib evolutionary search 的例子：增长 expressions，使其在 test set 上最小化 `|f(x) - target|`。Evaluator 是 deterministic 的。

运行：

```
python3 code/main.py
```

trace 会展示 HTN planner 如何分解 compound task（中途带 LLM fallback），以及 evolutionary loop 如何收敛到 target expression。

## Use It / 应用它

- **HTN planners**：`pyhop`、`SHOP3`，或为 domain-specific policy enforcement 自建。
- **ChatHTN**：研究代码；模式（symbolic + LLM fallback）可以干净移植到任何 HTN planner。
- **AlphaEvolve**：DeepMind 论文；模式（ensemble + evaluator）可复现。OpenEvolve 等开源分支正在出现。
- **Agent frameworks**：目前没有一等 HTN 或 AlphaEvolve。把它做成 subagent 或 background worker。

## Ship It / 交付它

`outputs/skill-hybrid-planner.md` 生成 hybrid planner scaffold（HTN 或 evolutionary），并明确限定 LLM 的角色。

## Exercises / 练习

1. 给 HTN planner 增加 backtracking：当 operator 的 postcondition 在 runtime 失败时，rollback 并尝试下一个 method。
2. 给 ChatHTN 增加 LLM-method cache：当 LLM 在 state pattern `P` 下分解 task `T` 时，保存结果。下一次先检查 method library。
3. 把 evolutionary search evaluator 换成真实 test suite。进化一个通过 20 个 test cases 的 sort function；报告收敛所需 generations。
4. 阅读 AlphaEvolve 的 evaluator design notes。为你关心的领域设计 evaluator（SQL query optimization、test-suite minimization、deployment YAML）。
5. 组合两者：用 HTN 把 compound task 分成 subtasks，再在每个 subtask 的 primitive operator 上用 evolutionary search。它在哪些地方发光，哪些地方过度工程？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| HTN | “Hierarchical planner” | 带 operators、preconditions、effects 的 task decomposition |
| Method | “Decomposition rule” | 把 compound task 拆成 subtasks 的规则 |
| Operator | “Primitive action” | 带 precondition 和 effect 的具体步骤 |
| ChatHTN | “LLM + HTN” | symbolic planner 在没有 method 匹配时询问 LLM |
| AlphaEvolve | “Evolutionary code search” | LLM ensemble mutate code，deterministic evaluator 选择 |
| Fitness function | “Evaluator” | 对 outputs 给出 deterministic、machine-checkable score |
| Online method learning | “Cached LLM decomposition” | 存储并泛化 LLM plans，减少 query cost |

## Further Reading / 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带 LLM mutations 的 evolutionary code search
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 什么时候使用 planner，什么时候用简单循环
