# Voting, Self-Consistency, and Debate Topology / 投票、自一致性与辩论拓扑

> 最便宜的聚合：采样 N 个独立 Agent，多数投票。Wang et al. 2022 self-consistency 用一个模型采样 N 次做到了这件事。多 Agent 把它扩展为 **heterogeneous** agents，以逃离 monoculture：不同模型、不同 prompt、不同 temperature、不同上下文。超越 majority vote 后，debate topology 很重要：MultiAgentBench（arXiv:2503.01935, ACL 2025）评估 star / chain / tree / graph coordination，发现 **graph 最适合 research**，并且超过约 4 个 Agent 后出现 “coordination tax”。AgentVerse（ICLR 2024）记录两种 emergent patterns：volunteer behaviors 和 conformity behaviors；conformity 既是特性（找到共识）也是风险（groupthink，Lesson 24）。本课绘制拓扑空间，实现每种变体，并测量 coordination tax。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 07（Society of Mind and Debate）, 第 16 阶段 · 14（Consensus and BFT）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 将 self-consistency 视为单模型投票基线，并解释其相关错误限制
- 比较 star、chain、tree、graph 四种 debate topology 的信息流
- 解释 heterogeneity、adversarial voice 和 bounded rounds 对准确率的影响
- 用成本、latency 和 accuracy 共同评估多 Agent ensemble，而不是只看最终正确率

## The Problem / 问题

Debate 可以提升准确率（Du et al., arXiv:2305.14325）。它也可能降低准确率。debate 是否有帮助取决于四个结构性选择：

1. 谁和谁说话（topology）。
2. 多少轮（Du 2023：rounds 和 agents 都独立贡献）。
3. Agent 是否异质（不同 base models 打破 monoculture）。
4. 是否存在 adversarial voice（steel-manning vs straw-manning）。

很多团队把“运行 5 个 Agent 然后投票”直接加到任务上，结果比单 Agent 更差。失败不是随机的，而是跟 topology 和 heterogeneity 绑定。本课是 topology map。

## The Concept / 概念

### Self-consistency, the single-model baseline / Self-consistency：单模型基线

Wang et al. 2022（"Self-Consistency Improves Chain of Thought Reasoning"）让同一个模型在 temperature > 0 下采样 N 次，并对 reasoning-path answers 做 majority vote。GSM8K 上，N=40 samples 相比单次 greedy decode 有显著提升。Self-consistency 是 multi-agent voting 的单 Agent 前身。

限制：self-consistency 使用一个 base model。错误天然相关。如果模型有系统性偏差，N 个样本都会共享。

### Multi-agent vote, the heterogeneous extension / 多 Agent 投票：异质扩展

把 N 个样本替换成 N 个 *不同* Agent。不同 base models（Claude、GPT、Llama）、不同 prompts、不同 tool access。收益是错误去相关；成本是不同 Agent 费用不同，协调也有开销。

2026 年异质辩论的一个常见名称是 **A-HMAD**：Adversarial Heterogeneous Multi-Agent Debate。不是 universally adopted，但论文用它表示“不同模型辩论，从而减少 monoculture collapse 造成的相关错误”。

### The four topologies / 四种拓扑

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star：一个 hub，其他 Agent 只和 hub 说话。等价于没有 back-channel 的 supervisor-worker。

Chain：线性，每个 Agent 看前一个输出。类似 pipeline。

Tree：层级，用在 hierarchical agent systems（Lesson 06）。

Graph：任意互联。包括 fully-connected clique 和任意 DAG。

### The coordination tax (MultiAgentBench) / 协调税

MultiAgentBench（MARBLE, ACL 2025, arXiv:2503.01935）在包含 research、coding、planning 的任务集上 benchmark star、chain、tree、graph。关键结果：

- **Graph** topology 在 research tasks 上最好。信息可以任意流动，Agent 能互相批评。
- **Star** 在 fast-answer factual tasks 上最好。hub 过滤并整合。
- **Chain** 在 stepwise pipelines（分阶段 refinement）上最好。
- **Coordination tax** 在 graph topology 超过约 4 个 Agent 后出现。wall-clock 和 token cost 增长快于质量。

4-Agent ceiling 是经验值，不是根本限制。它反映 2026 LLM context capacity：每个 Agent 的上下文被 peer outputs 填满，新增 Agent 的边际价值在所有人都能看见所有人后下降。

### Multi-Agent Debate Strategies / MAD 策略

arXiv:2311.17371 是 2023 年 MAD strategy survey。被后续工作复现的关键发现：那些 *结构上类似* self-consistency 的 MAD variants（独立采样 + 聚合），在相同预算下经常输给 self-consistency。MAD 最有帮助的场景是 Agent 真正异质，并且辩论有 adversarial structure（有一个 Agent 反对）。

### AgentVerse emergent patterns / AgentVerse 涌现模式

AgentVerse（ICLR 2024）记录了即使没有显式设计，多 Agent debate 也会涌现两类行为：

- **Volunteer.** Agent 主动提供帮助（“I can take the next step”）。有用之处：把工作分配给最有能力处理某子任务的 Agent。
- **Conformity.** Agent 调整自己的立场以匹配 critic，即使 critic 是错的。这是 debate 版 sycophancy（Lesson 14）。

Conformity 是 “debate-until-agreement” 会奖励强势发言者的原因。bounded rounds 加 separate judge 是缓解办法。

### Heterogeneity: the actual knob that moves accuracy / 真正推动准确率的是异质性

2024-2026 实践文献里的模式：把 N 个 Agent 中一个换成不同 base model，通常比把 N 增加 1 带来更大准确率提升。直觉是 monoculture：每增加一个独立错误源，都比再增加一个相关样本更有价值。

极限上，heterogeneity 胜过 numerosity。在有清晰 ground truth 的任务上，三个不同模型通常胜过一个模型的五个副本。

### Jury methods / Jury 方法

Sibyl framework（在 Minsky-LLM literature 中被引用）形式化了一个 "jury"：少量 specialized agents 在每个阶段通过投票 refinement 答案。不同于普通 majority vote，jury 有角色：一个 Agent cross-examines，一个提供 context，一个给 plausibility 打分。jury 是 plain vote（便宜但 monoculture-prone）和 full MAD（昂贵且 conformity-prone）之间的中间点。

### When vote-with-debate dominates / 什么时候投票+辩论占优

- 问题有 ground truth（fact、math、code behavior）。vote convergence 有意义。
- Agent 能访问不同 sources 或 tools（heterogeneity 可用）。
- 轮数有边界（通常 2-3）且有 separate judge 或 verifier。
- 预算允许 3-5 个 Agent。Graph topology 超过 5-7 个后 coordination tax 主导。

### When vote-with-debate hurts / 什么时候会伤害效果

- 问题偏 opinion。Agent 收敛到看起来最自信的答案，而不是最正确的答案。
- 所有 Agent 共享 base model。monoculture 让 consensus 失去意义。
- 轮数无边界。conformity 每次都会赢。
- 任务很简单。一个单 Agent 加 self-consistency N=5 更便宜且同样准确。

## Build It / 动手构建

`code/main.py` 实现：

- `run_star(agents, hub, question)` — hub 轮询每个 worker 并聚合。
- `run_chain(agents, question)` — 顺序 refinement。
- `run_tree(root, children, question)` — depth-2 aggregation 的 hierarchy。
- `run_graph(agents, question, rounds)` — all-to-all debate，bounded rounds。
- 脚本化 heterogeneity dial：每个 Agent 有一个 `error_bias`，表示其系统性错误倾向。
- measurement harness：在 N=3、5、7 下运行每个 topology，并报告 (accuracy, total_tokens, wallclock_simulated)。

运行：

```
python3 code/main.py
```

预期输出：topology × N → (accuracy, tokens, latency) 表。research-style tasks 上 graph 在 N=3-5 时胜出；fast-factual tasks 上 star 胜出；N=7 的 graph 展示 coordination tax（latency 增长快于 accuracy）。

## Use It / 应用它

`outputs/skill-topology-picker.md` 是一个 skill，读取 task description 并推荐 topology（star / chain / tree / graph）、N（Agent 数）、heterogeneity profile（使用哪些 base models）和 round bound。

## Ship It / 交付它

对任何 ensemble：

- 从 **self-consistency at N=5** using one strong base model 开始。它是便宜基线。
- 如果准确率重要，再升级到 **heterogeneous voting at N=3**。测量 delta。
- 只有当任务具备结构（research、multi-step）且 bounded rounds 可行时，才升级到 **debate topology**。
- 始终记录 minority cluster。当 minority 持续正确，这是 diversity signal。
- 和 accuracy 一起 benchmark wall-clock 与 tokens。“10x 成本换更高准确率”是业务决策，不是纯技术结论。

## Exercises / 练习

1. 运行 `code/main.py`。画出 graph topology 的 coordination-tax curve：accuracy vs N，tokens vs N。曲线在哪个 N 拐弯？
2. 实现 A-HMAD：三个 Agent 有刻意不同的 biases。在 Lesson 14 的 monoculture attack 上，同偏差 baseline 与 A-HMAD 如何比较？
3. 给 graph topology 增加一个不投票、只评分 final consensus 的 "judge" role。这会改变 emergent conformity behavior 吗？
4. 阅读 AgentVerse paper（ICLR 2024）。识别你的实现最强地表现出哪种 emergent behavior。能否通过 prompt change 引出相反行为？
5. 阅读 MultiAgentBench（arXiv:2503.01935）Section 4（topology experiments）。用你的 harness 在论文中的一个任务上复现 “graph-wins-research”。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Self-consistency | “采样 N 次后投票” | Wang 2022。单模型、N 个 temperature>0 samples，对 reasoning paths majority vote。 |
| Heterogeneity | “不同模型” | 不同 base models 或 prompt families 的 ensemble。打破 monoculture。 |
| MAD | “Multi-agent debate” | Agent 多轮交换 critique 的通用术语。见 Du 2023。 |
| A-HMAD | “Adversarial Heterogeneous MAD” | 强调不同模型 + adversarial structure 的 MAD 变体。 |
| Topology | “谁和谁说话” | star、chain、tree、graph。决定信息流。 |
| Coordination tax | “收益递减” | graph 中超过约 4 个 Agent 后，成本增长快于质量。 |
| Volunteer behavior | “主动帮忙” | AgentVerse 涌现模式：Agent 主动提出承担一步。 |
| Conformity behavior | “压力下同意” | AgentVerse 涌现模式：Agent 向 critic 对齐。 |
| Jury | “小型专门评审团” | Sibyl-style ensemble，带 examiner、context、scorer 等角色。 |

## Further Reading / 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — 单模型基线
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — agents 与 rounds 都独立贡献
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark，显示 graph 适合 research、chain 适合 pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD-strategy survey；相同预算下 MAD 常输给 self-consistency
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer 与 conformity emergent patterns
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — reference benchmark implementation
