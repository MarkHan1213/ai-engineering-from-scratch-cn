# Multi-Agent Debate and Collaboration / 多 Agent 辩论与协作

> Du et al.（ICML 2024，“Society of Minds”）运行 N 个 model instances，让它们独立提出答案，再经过 R 轮相互 critique 后收敛。它能提升 factuality、rule-following 和 reasoning。Sparse topology 在 token cost 上优于 full mesh。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 12（Workflow Patterns）, 第 14 阶段 · 05（Self-Refine and CRITIC）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 debate protocol：N 个 proposers、R 轮、收敛到共享答案。
- 描述为什么 debate 能改善 factuality、rule-following 和 reasoning。
- 解释 sparse topology：并不是每个 debater 都需要看到所有其他 debater。
- 基于 scripted LLM 实现一个 stdlib debate，包含 full-mesh 与 sparse variants；衡量 token cost 与 accuracy。

## The Problem / 问题

Self-Refine（Lesson 05）是一个模型批评自己，容易陷入 groupthink。CRITIC（Lesson 05）把 critique grounding 到外部工具，但外部工具并不总是可用。Debate 引入第三种模式：多个实例、cross-critique、通过分歧逐步收敛。

## The Concept / 概念

### Society of Minds (Du et al., ICML 2024)

- N 个 model instances 独立回答同一个问题。
- 经过 R 轮，每个模型阅读其他模型的 proposals 并 critique。
- 模型基于 critiques 更新自己的答案。
- R 轮之后，返回 convergent answer。

原始实验因为成本使用 N=3、R=2。在难题上（MMLU、GSM8K、Chess Move Validity、biography generation），更多 agents 和更多 rounds 会提升 accuracy。

跨模型组合优于单模型 debate：ChatGPT + Bard 一起 > 任一单独模型。

### Sparse topology / 稀疏拓扑

“Improving Multi-Agent Debate with Sparse Communication Topology”（arXiv:2406.11776, 2024-2025）表明，full-mesh debate 并不总是最优。Sparse topologies（star、ring、hub-and-spoke）可以用更低 token cost 达到相近 accuracy。每个 debater 只看到一部分 peers。

影响：

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals，每个读取 4 个 peers = 60 critique ops。
- Star N=5, R=3（一个 hub + 4 个 spokes）= 15 proposals，spokes 只读取 hub = 12 critique ops。

### When debate helps / Debate 何时有用

- **Factuality.** N 个独立 proposals 互相 cross-check，可减少 hallucination。
- **Rule-following.** 在 chess move validity 中，一个模型漏掉的规则，其他模型可能抓住。
- **Open-ended reasoning.** 多种 framing 会把答案逐渐压缩到正确方向。

### When debate hurts / Debate 何时伤害系统

- **Latency-sensitive UX.** N × R 轮串行推理是你可能承受不起的延迟。
- **Cost-sensitive scale.** 每个问题消耗 N × R tokens。
- **Simple factual lookups.** 一次 lookup 比五个 agent debate 便宜得多。

### 2026 practical instantiations / 2026 年实践形态

- **Anthropic orchestrator-workers**（Lesson 12）— 带 synthesis step 的 debate 变体。
- **LangGraph supervisor**（Lesson 13）— central router + specialist agents，可以把 debate 实现为一个 node。
- **OpenAI Agents SDK**（Lesson 16）— agents 之间来回 handoff，进行 iterative critique。
- **Multi-agent evals** — 把 debate 与 evaluator-optimizer 组合，用作 eval signal。

### Where this pattern goes wrong / 这种模式容易出错的地方

- **Convergence collapse.** 所有 agents 收敛到第一个错误答案。缓解方式：要求 disagreement rounds。
- **Hub failure.** 在 star topology 中，坏 hub 会污染所有人。应轮换 hub 或使用多个 hubs。
- **Prompt homogenization.** 所有 agents 使用同一个 prompt，会产出同样的答案。使用多样化 prompts 和/或 models。

## Build It / 动手构建

`code/main.py` 实现 stdlib debate：

- `Debater` class（scripted LLM，带 per-debater opinion drift）。
- `FullMeshDebate` 和 `SparseDebate` runners。
- 三个问题：一个 factual、一个 rule-based、一个 reasoning。
- Metrics：convergent answer、rounds to convergence、total critique ops。

运行：

```
python3 code/main.py
```

输出：每种 protocol 的 accuracy 和 cost；sparse 在 2/3 个问题上以更低成本匹配 full mesh。

## Use It / 应用它

- **Anthropic orchestrator-workers** 适合简单的 2-3-worker debates。
- **LangGraph** 适合带 checkpointing 的 stateful multi-round debate。
- **Custom** 适合研究场景或专门的 correctness guarantees。

## Ship It / 交付它

`outputs/skill-debate.md` 会搭出一个 multi-agent debate，支持配置 topology、N、R 和 convergence rule。

## Exercises / 练习

1. 实现一个 “forced disagreement” 规则：第 1 轮中，每个 debater 必须给出不同 proposal。测量它对 convergence speed 的影响。
2. 增加 confidence-weighted aggregation：debaters 返回 (answer, confidence)；aggregator 按 confidence 加权。它有帮助吗？
3. 把其中一个 “agent” 替换为观点不同的另一个 scripted LLM。heterogeneity 会提升 accuracy 吗？
4. 在你的 3 个问题上测量 full mesh vs sparse 的 token cost。绘制 cost vs accuracy。
5. 阅读 Society of Minds paper。把 toy port 到 N=5、R=3。什么会坏掉？什么会变好？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Debate | “Multi-agent critique” | N 个 proposers，R 轮 cross-critique，然后收敛 |
| Full mesh | “Everyone reads everyone” | 每个 debater 每轮都读取所有 peers |
| Sparse topology | “Limited peer view” | debaters 只读取一部分 peers |
| Hub-and-spoke | “Star topology” | 一个 central debater，N-1 个 spokes 只读取 hub |
| Convergence | “Agreement” | debaters 收敛到共享答案 |
| Society of Minds | “Du et al. debate paper” | ICML 2024 multi-agent debate method |

## Further Reading / 延伸阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology results
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers as a debate variant
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
