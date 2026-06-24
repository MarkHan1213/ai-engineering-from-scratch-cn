# The Autonomous Coding Agent Landscape (2026) / 2026 自主编码 Agent 版图

> SWE-bench Verified 在不到三年里从 4% 走到 80.9%。同一个 Claude Sonnet 4.5，在 SWE-agent v1 上是 43.2%，在 Cline autonomous 上是 59.8%——围绕模型的 scaffolding 现在和模型本身一样重要。OpenHands（原 OpenDevin）是最活跃的 MIT-licensed platform，它的 CodeAct loop 会在 sandbox 中直接执行 Python actions，而不是 JSON tool calls。headline numbers 掩盖了一个方法论问题：SWE-bench Verified 的 500 个任务里有 161 个只需要 1-2 行修改，而同样的 frontier models 在 SWE-bench Pro（10+ 行任务）上只有 23-59%。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites / 前置知识：** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释为什么“哪个 coding agent 最好”不如“在我的任务分布上端到端可靠性是多少”重要。
- 区分 base model 能力与 scaffold 带来的 reliability gain。
- 对比 CodeAct 和 JSON tool calls 的组合能力、审计性与 blast radius。
- 识别 SWE-bench Verified saturation 和 easy-tail 对 leaderboard 的影响。
- 为自己的 bug backlog 构造 Pro-like evaluation subset。

## The Problem / 问题

“哪个 coding agent 最好”是错误问题。正确问题是：在匹配我工作的 task distribution 上，用我将在生产中运行的 scaffolding，我能得到什么端到端可靠性？

2022 到 2026 年间，领域学到了一件事：scaffolding——retrieval layer、planner、sandbox、edit-verify loop、feedback format——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分是 43.2%；同一模型放进 Cline 的 autonomous scaffold 是 59.8%。相差 16.6 个百分点，权重相同。Base model 是组件；loop 才是产品。

配套问题是 benchmark saturation 会隐藏回归。SWE-bench Verified 接近饱和，而 easy-task tail（500 个任务中 161 个只需 ≤2 行修改）会把 top scores 拉高。真实世界质量更适合用 SWE-bench Pro（10+ 行改动）这类分布测量；同样的领先系统在那里仍只有 23-59%。

## The Concept / 概念

### SWE-bench, one paragraph / 一段话理解 SWE-bench

SWE-bench（Jimenez et al.）取真实 GitHub issues 和 ground-truth patches，让 Agent 产出能通过 test suite 的 patch。SWE-bench Verified（OpenAI, 2024）是人工策展的 500-task subset，移除了歧义和损坏任务。SWE-bench Pro 是更难的后继版本——任务要求 10+ 行修改，当前 frontier agents 在其上只有 23-59%。

### What the 2022 → 2026 curve actually shows / 2022 → 2026 曲线实际说明什么

- **2022**：research models 在 raw SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin-style scaffolding 约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推进到 40-55% 区间。
- **2026**：Claude Sonnet 4.5 和 frontier competitors 在 SWE-bench Verified 上达到 70-80%+。Epoch AI leaderboard 实时跟踪这一变化。

斜率来自三个复合来源：更好的 base models、更好的 scaffolding（CodeAct、reflection、verifier loops），以及更好的 benchmarks（Verified 移除噪声）。

### CodeAct vs JSON tool calls / CodeAct 与 JSON tool calls

OpenHands（All-Hands-AI, arXiv:2407.16741，原 OpenDevin）押注一个特定架构：模型不输出由 host 解码并执行的 JSON tool calls，而是输出 Python code，由 Jupyter-style kernel 在 sandbox 中运行。Agent 可以在一个 action 里遍历文件、串联 tools，并捕捉自己的 exceptions。

Trade-off：

- **JSON tool calls**：每个 action 都是一轮；容易审计；组合性有限；默认更安全，因为每次调用都会经过显式 validator。
- **CodeAct**：一个 action 可以是一整个 program；组合性强；需要 hardened sandbox（OpenHands 使用 Docker isolation）；失败模式包括 sandbox runtime 允许的一切。

两种架构都在生产中使用。CodeAct 在 open platforms 中占主导（OpenHands、smolagents）。JSON tool calls 仍在 managed services 中占主导（Anthropic Managed Agents、OpenAI Assistants），因为 provider 控制 executor。

### Scaffolds in the 2026 landscape / 2026 版图中的 scaffolds

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的 open platform；event-stream replayable |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 首个端到端 SWE-bench scaffold |
| Aider | Apache-2 | local repo 中 edit-via-diff | 最小 scaffold，regression stability 强 |
| Cline | Apache-2 | 带 tool policy 的 VS Code agent | Sonnet 4.5 上得分最高的 open scaffold |
| Devin (Cognition) | Proprietary | Managed VM + planner | 首个 “AI software engineer” 产品类别 |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 详细覆盖 agent loop |

### Why scaffolding dominates / 为什么 scaffolding 主导结果

一次 coding run 是 long-horizon trajectory（Lesson 1）。可靠性会跨步骤复合。Scaffolding 主要在三个地方买来分数：

1. **Retrieval / 检索**：找到该读哪些文件是静默瓶颈。SWE-agent 的 ACI、OpenHands 的 file-index、Aider 的 repo-map 都在解决它。
2. **Verifier loop / 验证循环**：运行 tests、阅读 stack traces、重新尝试，在 SWE-bench 上能带来 10+ 点差异。
3. **Failure containment / 失败收容**：能在错误时 rollback 的 sandbox 会阻止损害复合。同一模型有无 verifier loop，看起来像两个不同产品。

### Benchmark saturation and the real distribution / benchmark 饱和与真实分布

OpenHands 作者和 Epoch AI 都指出，SWE-bench Verified 有一个 easy tail：500 个任务里有 161 个只需 1-2 行修改。高分部分来自这条尾部。SWE-bench Pro 限定 10+ 行改动，即便 frontier systems 也只有 23-59%。你的生产分布几乎一定更接近 Pro，而不是 Verified。

选择 Agent 的含义是：跑你自己 bug backlog 的 Pro-like subset。真正重要的分数，是代表你实际交付任务的那部分任务上的分数。

## Build It / 动手构建

本课用 deterministic stub model 对比两个 toy scaffold：一个每轮只执行一个 JSON tool-call action，另一个每次可以执行小段 Python 的 CodeAct scaffold。这样可以把 scaffold 差异从 model quality 中隔离出来。

## Use It / 应用它

`code/main.py` 在固定 mini-task distribution 上比较两个 toy agent scaffolds：

1. 一个 **JSON tool-call** scaffold，每轮执行一个 action。
2. 一个 **CodeAct** scaffold，每个 action 可以发出一小段 Python snippet。

两者都使用 stub “model”（deterministic rules），因此比较隔离了 scaffold 与 model quality。输出会展示 CodeAct scaffold 用更少 turns 解决更多任务，但代价是更大的 per-action blast radius。

## Ship It / 交付它

`outputs/skill-scaffold-audit.md` 帮你在采用一个 proposed coding-agent scaffold 前做审计：retrieval quality、verifier presence、sandbox isolation，以及 benchmark-to-distribution fit。

## Exercises / 练习

1. 运行 `code/main.py`。两个 scaffold 在同一 task set 上分别需要多少 turns？每个的 per-action blast radius 是什么？

2. 阅读 OpenHands paper（arXiv:2407.16741）。论文认为 CodeAct 在 complex tasks 上优于 JSON tool calls。指出论文承认的一个 failure mode，并写一句话说明它什么时候会在生产中占主导。

3. 从你的 bug backlog 中选一个需要跨两个文件修改 10+ 行的任务。估算 frontier model 在（a）JSON tool calls 和（b）CodeAct 下的端到端成功概率。解释差距。

4. SWE-bench Verified 有 161 个 single-file、1-2 行任务。构造一个排除它们的分数。Leaderboard 会如何洗牌？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释用于移除 ambiguous tasks 的具体方法，并说出一种 curation 会漏掉的类别。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| SWE-bench | “Coding benchmark” | 带 ground-truth patches 和 test suites 的真实 GitHub issues |
| SWE-bench Verified | “Cleaned subset” | 500 个人工策展任务，仍有 easy-tail |
| SWE-bench Pro | “Harder subset” | 10+ 行修改；frontier 只有 23-59% |
| CodeAct | “Code-as-action” | Agent 发出 Python；Jupyter-style kernel 在 sandbox 中执行 |
| JSON tool call | “Function calling” | 每个 action 是执行前会被验证的 structured JSON payload |
| Scaffold | “Agent framework” | 围绕 base model 的 retrieval + planner + executor + verifier loop |
| ACI (Agent-Computer Interface) | “SWE-agent 的格式” | 为 LLM ergonomics 设计的 command set，不是 human shells |
| Verifier loop | “Test-and-retry” | 跑 tests、读输出、修 patch；最大的非模型可靠性增益 |

## Further Reading / 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始 benchmark 与方法。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset 的构建方式。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构与 event-stream design。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding-agent reliability framing。
