# Eval-Driven Agent Development / 评测驱动的 Agent 开发

> Anthropic 的建议是：“start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when needed.” Evaluation 不是最后一步。它是驱动 Phase 14 中其他所有选择的外层循环。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段全部内容
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出三层 evaluation：static benchmarks、custom offline、online production，以及每层用途。
- 解释 evaluator-optimizer tight loop。
- 描述 2026 年最佳实践：evals 与 code 放在一起，在 CI 中运行，并 gate PRs。
- 把 Phase 14 的每一课都连接到它生成的 eval case。

## The Problem / 问题

Agents 往往能通过 demo，却会以 demo 无法预测的方式在生产中失败。Benchmarks 回答的是 “这个模型整体能力够不够？” 而不是 “这个 agent 是否在为我的产品提交正确 patch？” 答案是：持续运行三层 evaluation，并把每个 guardrail 和 learned rule 映射到一个 eval case。

## The Concept / 概念

### Three evaluation layers / 三层评测

1. **Static benchmarks** — 代码使用 SWE-bench Verified（Lesson 19），浏览器 / 桌面使用 WebArena/OSWorld（Lesson 20），generalist 使用 GAIA（Lesson 19），tool use 使用 BFCL V4（Lesson 06）。用于 cross-model comparison 和 regression gating。Contamination 真实存在：SWE-bench+ 发现 32.67% solution leakage。始终报告 Verified / +-audited scores。

2. **Custom offline evals** — 你的产品形状：
   - LLM-as-judge（Langfuse、Phoenix、Opik — Lesson 24）。
   - Execution-based（运行 patch，检查 tests）。
   - Trajectory-based（把 action sequences 与 gold 对比；OSWorld-Human 显示 top agents 是 gold 的 1.4-2.7x）。

3. **Online evals** — 生产：
   - Session replays（Langfuse）。
   - Guardrail-triggered alerts（Lesson 16, 21）。
   - Per-step cost / latency tracking（Lesson 23 OTel spans）。

### Evaluator-optimizer (Anthropic)

紧密循环：

1. Proposer 生成 output。
2. Evaluator 评判。
3. Refine，直到 evaluator 通过。

这是泛化后的 Self-Refine（Lesson 05）。任何你在意的 agent flow 都可以包一层 evaluator-optimizer 来提高可靠性。

### 2026 best practice / 2026 年最佳实践

- Evals 与 code 放在一起。
- 每个 PR 都在 CI 中运行。
- 用 eval scores gate merge（例如 “no regression > 5% vs main”）。
- 每个 guardrail 都映射到 eval case。
- 每条 learned rule（Reflexion、pro-workflow learn-rule）都映射到 failure case。

### Tying Phase 14 together / 串起 Phase 14

Phase 14 的每一课都会生成 eval cases：

| Lesson | Eval case it generates |
|--------|------------------------|
| 01 Agent Loop | Budget-exhausted, infinite-loop guard |
| 02 ReWOO | Planner replans correctly when a tool fails |
| 03 Reflexion | Learned reflections apply on retry |
| 05 Self-Refine/CRITIC | Judge passes refined output |
| 06 Tool Use | Argument coercion works; unknown tools rejected |
| 07-10 Memory | Retrieval citations match sources; stale facts invalidate |
| 12 Workflow Patterns | Each pattern produces correct output |
| 13 LangGraph | Resume reproduces state exactly |
| 14 AutoGen Actors | DLQ catches crashed handlers |
| 16 OpenAI Agents SDK | Guardrail trips on the right inputs |
| 17 Claude Agent SDK | Subagent results return to orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score, WebArena success rate, OSWorld efficiency |
| 21 Computer Use | Per-step safety catches injected DOM |
| 23 OTel | Spans emit required attributes |
| 26 Failure Modes | Detectors tag known failures |
| 27 Prompt Injection | PVE refuses poisoned retrievals |
| 28 Orchestration | Supervisor routes to the right specialist |
| 29 Runtime Shapes | DLQ handles N% failure |

如果你的 eval suite 覆盖了这些 case，就覆盖了 Phase 14。

### Where eval-driven development fails / Eval-driven development 容易失败的地方

- **No baseline.** 没有 last-known-good 的 evals 读不出意义。存储 baselines。
- **LLM-judge without grounding.** Judges 也会 hallucinate。CRITIC pattern（Lesson 05）— judge 要 grounding 到 external tools。
- **Over-fitting to evals.** 为 eval 优化偏离了生产有用性。轮换 cases。
- **Flaky evals.** 非确定性 cases 会造成 false alarms。固定 seeds，snapshot state。

## Build It / 动手构建

`code/main.py` 是一个 stdlib eval harness：

- 带 categories（benchmark、custom、online）的 case registry。
- 一个 scripted agent under test。
- Evaluator-optimizer loop：propose、judge、refine，直到 pass 或达到 max rounds。
- CI gate：aggregate pass rate + regression against baseline。

运行：

```
python3 code/main.py
```

输出：per-case pass/fail、regression flag、CI gate verdict。

## Use It / 应用它

- 在与你 agent code 相同的 repo 中编写 eval cases。
- 每个 PR 都通过 CI 运行它们。
- 遇到 regression 就 fail build。
- 随时间跟踪 pass rate。
- 每个 production failure 都补一个新 case。

## Ship It / 交付它

`outputs/skill-eval-suite.md` 会为一个 agent product 构建三层 eval suite，包含 CI gates 和 regression tracking。

## Exercises / 练习

1. 拿一个你的 production failure。写一个能复现它的 eval case。你的 agent 现在能通过吗？
2. 为你的 domain 构建一个三维度（factual、tone、scope）的 LLM-judge rubric。给 50 个 sessions 打分。
3. 把 eval suite 接入 CI。遇到 >=5% regression 时 fail build。
4. 增加 trajectory-efficiency metric：agent 走了多少步，相比 gold trajectory 如何？
5. 把 Phase 14 的每一课映射到你的 suite 中的 eval case。有没有缺失？那就是要补的 gap。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Static benchmark | “Off-the-shelf eval” | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | “Domain eval” | 面向产品形态的 LLM-as-judge / exec / trajectory |
| Online eval | “Production eval” | Session replay, guardrail alerts, cost/latency tracking |
| Evaluator-optimizer | “Propose-judge-refine” | 迭代直到 judge 通过 |
| CI gate | “Merge blocker” | eval regression 时 fail build |
| Baseline | “Last-known-good” | 用于检测 regression 的 reference score |
| Trajectory efficiency | “Steps over gold” | Agent step count 除以 human expert minimum |

## Further Reading / 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "start simple, optimize with evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — the curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-use benchmark
- [Langfuse docs](https://langfuse.com/) — evals + session replay in practice
