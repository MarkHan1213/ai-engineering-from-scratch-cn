# Automated Alignment Research (Anthropic AAR) / 自动化对齐研究

> Anthropic 在彼此独立的 sandboxes 中运行多组 Claude Opus 4.6 Autonomous Alignment Researchers，并通过一个 shared forum 协调；forum 日志位于任何 sandbox 之外，因此 Agent 不能删除自己的记录。在 weak-to-strong training 问题上，这些 AAR 的表现超过了人类研究者。Anthropic 自己的总结也提醒，prescribed workflows 经常限制 AAR 灵活性并降低表现。自动化 alignment research 是时间线压缩的一步，也会把时间线压缩到 RSP 想要检测的那些 misalignment risks 上。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, parallel-research-forum simulator)
**Prerequisites / 前置知识：** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 Anthropic AAR 的 parallel agents、independent sandboxes 和 shared forum 架构。
- 解释 out-of-sandbox append-only log 为什么是研究可信度的必要条件。
- 分析 prescribed workflow 与 free decomposition 在能力和审计性上的 tradeoff。
- 将 AAR 放到 RSP / FSF 的 AI R&D capability thresholds 语境中。
- 设计 automated alignment research pipeline 的部署前审查。

## The Problem / 问题

Alignment research 昂贵，昂贵在高质量人类研究者的时间上。Scalable oversight、reward specification、weak-to-strong training 这类问题，每次实验迭代都可能花费数周。随着前沿能力推进，alignment workload 增长速度快过合格研究者供给。

Automated Alignment Research（AAR）问的是：那些能力正在跑赢 alignment 的前沿模型，是否也能参与缩小这个差距？Anthropic 2026 年关于 AAR-run weak-to-strong-training study 的报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是这类 deployed system 的首批公开结果之一。

结果确实积极：AARs 在一个研究问题上超过了 human baseline。但同一个系统也集中体现了本阶段已经展开的所有担忧。如果 alignment research 可以自动化，那么破坏 safeguard 的部分也可能自动化。RSP 的 AI R&D capability thresholds 正是带着这个 loop 写的。

## The Concept / 概念

### The architecture (as publicly described) / 公开描述中的架构

- **Parallel agents / 并行 Agent。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个都在独立 sandbox 中。Sandboxes 限制 filesystem、network 和 process surface。
- **Shared forum / 共享论坛。** Findings 发布到 shared forum；其 storage backend 位于所有 sandbox 之外。Agent 可以读 forum，但不能在自己的 sandbox 内删除或编辑 forum records。
- **Task allocation / 任务分配。** 人类设计的 task queue 将研究问题分配给 AAR。AAR 提出 decompositions、运行实验、写 findings。
- **Weak-to-strong training focus / 聚焦 weak-to-strong training。** 已发布研究处理的是 weak-to-strong training：用较弱 supervisor model 训练更强 student model。AARs 在该任务上超过了人类研究者。

### Why the out-of-sandbox log matters / 为什么 out-of-sandbox log 重要

Log-integrity property 是受控实验与潜在被攻破实验之间的差异。如果 AAR 能编辑自己的 logs，它就能掩盖失败或可疑实验。Anthropic 直接强调了这一点：这个属性是信任研究输出的必要条件，并且并不容易验证。

基础设施里的类比是 append-only logs，并把写入穿透到外部存储（例如 S3 object-lock、immutable Datadog logs）。Agent 没有 credential 修改历史记录。这也是 2026 年企业安全团队用来构建 ransomware-resistant audit trails 的同一模式。

### The prescribed-workflow tradeoff / prescribed workflow 的 tradeoff

Anthropic 报告指出，prescribed workflows 经常限制 AAR 灵活性并降低表现。换句话说，如果人类 designer 把研究计划逐步写死，AAR 的表现反而不如只给 high-level goals 并允许自由 decomposes。这与 agentic research 中其他结果一致（LangGraph、Microsoft's Agent Framework）：过度指定的 scaffolds 会伤害强 Agent。

安全含义更微妙。提高 AAR 表现的灵活性，也允许未经批准的 decomposition、新颖 tool use 和 goal drift。这里的取舍是一个 design-space decision：要给 Agent 多少灵活性，才能换来足够能力、超过人类，同时又不把目标泛化到分布外？

### The compression risk / 时间线压缩风险

RSP v3.0（Lesson 19）引入 AI R&D capability threshold：以相对 human + AI tools 有竞争力的成本，完全自动化 AI R&D pipeline。DeepMind 的 FSF v3 也包含类似的 ML R&D autonomy level。两个框架都把这个 threshold 当作触发更高控制措施的条件。

AAR 离这个 threshold 还差一步：它自动化了 pipeline 的一部分（针对具体、明确边界任务的 alignment research），但没有自动化端到端 capability development loop。时间线问题是这个差距会以多快速度闭合。

Compressed timelines 带来的问题是复合失败。如果 alignment research 和 capability research 以相似速度复利，misalignment risk surface 至少和 capability 一样快增长。如果 capability 复利更快（历史趋势如此），差距会扩大。这就是为什么 AAR 是有条件的好事：每新增一个 alignment result 只有在研究过程可信时，才会真正缩小差距。

### What AAR does not replace / AAR 不能替代什么

人类研究者设定 task queue、审查结果，并持有 constitutional authority。AAR 加速的是 pipeline 中段，而不是两端。Anthropic 的公开输出既包含 AAR contributions，也包含人类研究者对 publish、retract 和 refine 的判断。

这相当于把 Lesson 15 的 propose-then-commit pattern 应用到研究本身：AARs propose；humans commit。

## Build It / 动手构建

本课会构建一个 multi-AAR research forum 模拟器：多个独立研究 Agent 并行发布 findings，forum 以 append-only 方式记录，并可模拟 tamper attempts。你会看到能力、灵活性和审计性如何互相牵制。

## Use It / 应用它

`code/main.py` 模拟一个 multi-AAR research forum。三个独立 “research agents” 并行运行，把 findings 发布到 append-only forum，并可选择尝试篡改。

观察：

- out-of-sandbox logging 如何捕捉 log-tampering attempts。
- 带自由 decomposition 的 task allocation 如何在更难问题上胜过 fixed workflows。
- 如果 log 不是 tamper-evident，一个 compromised AAR 如何污染 forum signal。

## Ship It / 交付它

`outputs/skill-aar-deployment-review.md` 是 automated-alignment-research pipeline 的 pre-deployment review。

## Exercises / 练习

1. 运行 `code/main.py`。对比 “fixed-workflow” 和 “free-decomposition” settings。Free decomposition 是否总是赢，还是存在 fixed workflow 有竞争力的问题类别？

2. 修改模拟器，让一个 Agent 尝试 log tampering。确认 append-only log 会检测到它。写一段描述 log 中的 detection signature 到底是什么。

3. 阅读 Anthropic 的 weak-to-strong AAR report。指出 AARs 超过人类研究者的具体 sub-task。它为什么适合自动化？

4. 设计一个 task-queue allocation policy，在 AAR flexibility（更好结果）和 prescribed-workflow constraints（更易审计）之间平衡。说明你会如何 A/B test 两者。

5. 阅读 RSP v3.0 的 AI R&D-4 threshold。用一段话说明你认为哪些能力会跨过它，而当前 AAR 还没有。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AAR | “Automated Alignment Researcher” | Claude Opus 4.6 instance，针对 alignment problems 自主运行 |
| Weak-to-strong training | “用弱 supervisor 训练强模型” | 经典 scalable-oversight benchmark，AARs 在其上超过人类 |
| Shared forum | “Agent 发布 findings 的地方” | Append-only、out-of-sandbox storage |
| Out-of-sandbox log | “Agent 不能编辑自己的记录” | Tamper-evident write-through to external storage |
| Prescribed workflow | “人类 designer 写的 step-by-step plan” | 限制 AAR；常常比 free decomposition 表现更差 |
| Free decomposition | “Agent 自己拆任务” | 更有能力，更难审计 |
| AI R&D threshold | “RSP/FSF capability level” | 以有竞争力成本完全自动化 R&D pipeline |
| Compressed timeline | “Alignment vs capability race” | 如果 capability 复利快过 alignment，misalignment risk 会增长 |

## Further Reading / 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 一手来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D threshold framing。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广义的 agent-autonomy framing。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 并行的 ML R&D autonomy levels。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AARs 攻克的底层问题。
