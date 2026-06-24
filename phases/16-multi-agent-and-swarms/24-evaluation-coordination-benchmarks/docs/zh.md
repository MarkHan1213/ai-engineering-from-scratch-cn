# Evaluation and Coordination Benchmarks / 评测与协同基准

> 2025-2026 年五个 benchmark 覆盖了多 Agent 评测空间。**MultiAgentBench / MARBLE**（ACL 2025, arXiv:2503.01935）用 milestone KPIs 评测 star/chain/tree/graph topologies；**graph 最适合 research**，cognitive planning 带来约 3% milestone achievement。**COMMA** 评测 multimodal asymmetric-information coordination；包括 GPT-4o 在内的 SOTA 模型都很难超过 random baseline。**MedAgentBoard**（arXiv:2505.12371）覆盖四类医学任务，并常发现 multi-agent 不压倒 single-LLM。**AgentArch**（arXiv:2509.10769）benchmark enterprise agent architectures，组合 tool-use + memory + orchestration。**SWE-bench Pro**（[arXiv:2509.16941](https://arxiv.org/abs/2509.16941)）有 1865 个问题，覆盖 41 个 repos，包括 business apps、B2B services 和 developer tools；frontier models 在 Pro 上约 23%，而 Verified 上 70%+，这是 contamination 的现实提醒。Claude Opus 4.7（2026 年 4 月）据称在 Pro 上 **64.3%**，并使用 explicit agent-teams coordination（暂无 Anthropic primary source，要视为 preliminary）；Verdent（agent scaffold）在 Verified 上达到 **76.1% pass@1**（[Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。**AAAI 2026 Bridge Program WMAC**（https://multiagents.org/2026/）是 2026 年社区焦点。本课基于 MARBLE metrics，运行 topology-vs-metric sweep，并钉住规则：“只通过 SWE-bench Verified 不能证明泛化”。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 15（Voting and Debate Topology）, 第 16 阶段 · 23（Failure Modes）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 识别 2026 年五个关键多 Agent benchmark 及各自测量对象
- 区分 SWE-bench Verified 与 SWE-bench Pro，并解释 contamination 风险
- 使用 benchmark-claims checklist 审视论文或产品 headline result
- 为生产系统设计内部 benchmark，并同时报告 cost、latency、random baseline delta

## The Problem / 问题

当一篇论文说“我们的多 Agent 系统更好”时，真正的问题是：比谁更好、在哪个任务上、怎么测？2023-2024 年多 Agent evaluation 很混乱：每个人选自己的 metrics、自己的 baselines、自己的 task sets。2025-2026 年的 benchmark 开始施加结构。

没有共享 benchmark，就无法有意义地比较两个多 Agent 系统。更糟的是，没有 hold-out benchmark，frontier models 会被污染。SWE-bench Verified 到 2025 年中已部分进入训练语料，frontier scores inflated；Pro 被设计成 uncontaminated reality check。

本课列出 2026 年五个 canonical benchmark，说明每个测什么，并训练你怀疑地阅读 benchmark claims。

## The Concept / 概念

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935。评测四种 coordination topologies（star、chain、tree、graph），任务包含 research、coding、planning。milestone-based KPIs 追踪 partial progress，而不是只看最终 success。

测量结果：

- **Graph** topology 在 research scenarios 最好；支持 any-to-any critique。
- **Chain** 最适合 stepwise-refinement coding。
- **Star** 最适合 fast-factual consolidation。
- **Coordination tax** 在 graph 超过约 4 个 Agent 后出现。
- **Cognitive planning** 在各 topology 上增加约 3% milestone achievement。

适用：需要 apples-to-apples 比较 coordination topologies。MARBLE repo（https://github.com/ulab-uiuc/MARBLE）提供 evaluator。

### COMMA — multimodal asymmetric information

覆盖 Agent 拥有不同 observation modalities，且必须在不能完全共享信息的情况下协调的任务。报告结果很不舒服：包括 GPT-4o 在内的 frontier models 在 COMMA 的 agent-agent collaboration 上很难超过 **random baseline**。信号是：多 Agent modality coordination 训练和评测都不足。LLM 对单模态合作还可以，多模态协同会崩。

适用：系统包含 multimodal 或 asymmetric-information coordination。COMMA 的 null result 提醒你先测量再声称。

### MedAgentBoard — domain stress test

arXiv:2505.12371。四类医学任务：diagnosis、treatment planning、report generation、patient communication。比较 multi-agent、single-LLM 和 conventional rule-based systems。

发现：multi-agent 在多数类别并不压倒 single-LLM。多 Agent 优势很窄；当子任务清晰可分（diagnosis + treatment）时，task decomposition 有帮助；当 coordination overhead 超过 specialization gain（report generation）时会伤害。

适用：你的领域有清晰 single-LLM baseline。若 MedAgentBoard 教训可泛化，许多 proposed multi-agent systems 都是过度工程。

### AgentArch — enterprise architectures

arXiv:2509.10769。企业场景，tool use、memory 和 orchestration 叠在一起。benchmark 隔离每层贡献：加 tools 帮多少？加 memory 帮多少？加 multi-agent orchestration 帮多少？

适用：你在设计 enterprise agent stack，需要为每一层证明价值。AgentArch 帮你避免购买无法测量价值的 feature。

### SWE-bench Pro — the reality check

arXiv:2509.16941。1865 个 problems，覆盖 41 个 repositories，包括 business apps、B2B services 和 developer tools。设计目标是与后续训练 cutoffs **不污染**。frontier models 在 Pro 上约 23%，在 Verified 上 70%+。差距就是 contamination signal。

2026 年 4 月分数：

- Claude Opus 4.7 on Pro：**64.3%**（据称使用 explicit agent-teams coordination；暂无 Anthropic primary source，因此视为 preliminary）。
- Verdent（agent scaffold）on Verified：**76.1% pass@1**（[technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- Pro 上无 agent scaffolding 的 frontier raw scores：约 23-35%（[SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)）。

结论：“我们打过 SWE-bench Verified”已经不再是能力证据。Pro 是当前 gating test。Agent-team scaffolding 在 Pro 上带来可测增益（约 30-40 点 delta），这是 2026 年多 Agent coordination 最强的经验证据之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination（https://multiagents.org/2026/）。这是 2026 年 multi-agent AI research 的社区焦点。accepted papers 和 workshop proceedings 是评估新方法的核心场地；生产决策上，WMAC-accepted claims 优先于 arXiv preprints。

### Read benchmark claims skeptically — the 2026 checklist / 2026 benchmark claim 审查清单

有人声称多 Agent 结果时，问：

1. **Which benchmark, which split?** SWE-bench Verified vs Pro 差别很大。错 split 上的数字没有价值。
2. **Contamination check.** benchmark 是否在 model training cutoff 之后发布？如果不是，谨慎看待。
3. **Baseline comparison.** 与 single-LLM baseline、random、prior multi-agent work 比。不是“与自己系统的未调优版本比”。
4. **Statistical significance.** N trials、p-value、confidence interval。frontier models 方差大，单次 run 会误导。
5. **Task diversity.** 单任务还是多任务？生产关心 generalization。
6. **Cost disclosure.** 每任务 tokens、wall-clock。20x 成本换 90% 是业务决策，不是单纯能力 claim。

### What none of the benchmarks measure well / 当前 benchmark 测不好的东西

- **Long-horizon coordination.** 数天 wall-clock 互动。现有 benchmark 都偏短。
- **Adversarial resilience.** 一个 Agent malicious 或 compromised 时会发生什么？
- **Drift under deployment.** benchmark 静态；生产分布会漂移。
- **Cost-normalized performance.** 多数 benchmark 报 raw accuracy，不报 accuracy-per-dollar。

为你真正关心的轴构建内部 benchmark，通常是正确动作。

## Build It / 动手构建

`code/main.py` 是一个非交互 walk-through：

- 在 toy task 上模拟 3 个多 Agent 系统。
- 为每个系统计算 MARBLE-style milestone metrics。
- 通过从 “training” set 中 withholding tasks 做 contamination check。
- 显式与 random baseline 比较。
- 打印 benchmark-claims scorecard。

运行：

```bash
python3 code/main.py
```

预期输出：system scorecard，包含 raw accuracy、milestone achievement、cost-per-task、vs-random baseline delta，以及 contamination-check note。

## Use It / 应用它

`outputs/skill-benchmark-reader.md` 读取任意 multi-agent benchmark claim，并应用审查 checklist。输出 grade 和 caveats。

## Ship It / 交付它

生产 evaluation discipline：

- **Build an internal benchmark** 反映真实生产分布。public benchmarks 提供参考，但不能替代。
- **Include a random baseline** in every comparison。如果 coordination task 上不能大幅超过 random，任务可能定义有问题。
- **Report cost alongside accuracy.** token cost 和 wall-clock。ops teams 两者都需要。
- **Rebuild the benchmark quarterly.** 生产分布漂移，旧 benchmark 会误导。
- **Avoid published-benchmark overfitting.** 如果团队只为 SWE-bench Pro 数字优化，会损害生产。

## Exercises / 练习

1. 运行 `code/main.py`。找出三个模拟系统中 cost-per-milestone 最好的一个。它是否也是 raw-accuracy 最高的？
2. 阅读 MultiAgentBench（arXiv:2503.01935）。对你自己的任务领域，判断 MARBLE 会推荐四种 topology 中哪一种。根据论文结果说明理由。
3. 阅读 SWE-bench Pro paper。它具体靠什么抵抗 contamination？同样技术能否用于你关心的其他 benchmark？
4. 阅读 COMMA 的 multimodal coordination 发现。设计一个可加入你内部 benchmark 的简单 multimodal coordination task。什么结果算有用信号？
5. 对近期一篇 multi-agent paper 的 headline result 应用 benchmark-claims checklist。你会给什么 grade？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MARBLE | “MultiAgentBench” | ACL 2025；star/chain/tree/graph topologies，带 milestone KPIs。 |
| COMMA | “Multimodal benchmark” | multimodal asymmetric-info coordination；frontier models 对 random 表现吃力。 |
| MedAgentBoard | “Domain stress test” | 四类医学任务；常发现 multi-agent 不压倒 single-LLM。 |
| AgentArch | “Enterprise benchmark” | tools + memory + orchestration 分层。 |
| SWE-bench Pro | “Contamination-resistant” | 1865 problems、41 repos；Pro 约 23% vs Verified 70%+，是 contamination signal。 |
| Milestone achievement | “Partial credit” | 奖励过程进展，而不是只看最终成功的 benchmark。 |
| Contamination | “Benchmark 泄漏进训练” | benchmark 发布后进入训练语料，导致分数膨胀。 |
| WMAC | “AAAI 2026 Bridge Program” | Workshop on Multi-Agent Coordination；社区焦点。 |

## Further Reading / 延伸阅读

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 带 milestone KPIs 的 topology benchmark
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — 参考实现
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domain stress test；multi-agent 常不占优
- [AgentArch](https://arxiv.org/abs/2509.10769) — enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — frontier models 在 Verified 和 Pro 上的分数
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026 社区焦点
