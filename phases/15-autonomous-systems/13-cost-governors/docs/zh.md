# Action Budgets, Iteration Caps, and Cost Governors / 动作预算、迭代上限与成本治理

> 一个中型电商 Agent 团队启用 “order-tracking” skill 后，月度 LLM 成本从 $1,200 跳到 $4,800。这不是计价 bug，而是 Agent 找到了一个新 loop，并持续在里面花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把这类防御 codify 成一组层：每请求 `max_tokens`、每任务 token 和 dollar budgets、每日/月 caps、iteration caps、tiered model routing、prompt caching、context windowing、昂贵动作上的 HITL checkpoints、budget breach 上的 kill switches。Anthropic 的 Claude Code Agent SDK 用不同名称交付同一批 primitives。Financial velocity limits——例如 10 分钟内超过 $50 就切断访问——能比月度 caps 更快抓住 loop。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, layered cost-governor simulator)
**Prerequisites / 前置知识：** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 Denial of Wallet 为什么是 autonomous agents 独有的生产失败模式。
- 设计覆盖 per-request、per-task、rolling window 和 velocity 的 cost-governor stack。
- 区分 runaway loop、slow leak、bad release 和 legitimate surge 对应的告警窗口。
- 将 Claude Code 的 `max_turns`、`max_budget_usd`、tool allowlists 与 permission modes 组合使用。
- 为新 tool surface 设计 per-tool cap 和增长告警。

## The Problem / 问题

Autonomous agents 的每一轮都在花真钱。聊天机器人的坏输出是坏回答；Agent 的坏循环是一张账单。行业文档里把这种失败叫作 “Denial of Wallet”：Agent 不停 reasoning、不停 tool-calling、不停计费，而没有任何东西阻止它，因为系统从未为此设计。

修复方法不是一个数字，而是在不同时间尺度和粒度上叠加的 limit stack：per-request、per-task、per-hour、per-day、per-month。设计良好的 stack 能在几分钟内抓住 runaway loop，在几小时内抓住 slow leak，在一天内抓住 bad release。当 Agent 是 long-horizon 且自主时，同一个 stack 才能让预算真正存在。

这是工程课：数学很简单，团队失守的是纪律。下面的 limits 都出现在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK docs 中。

## The Concept / 概念

### The cost-governor stack / 成本治理栈

1. **每请求 `max_tokens`。** 简单。防止单次调用发出无界 completion。
2. **Per-task token budget。** 整个 run 不超过 N tokens。达到 cap 就 hard stop。
3. **Per-task dollar budget。** 与 token budget 相同，只是用货币计量。Claude Code 中是 `max_budget_usd`。
4. **Per-tool call cap。** 最多 N 次 `WebFetch` calls、N 次 `shell_exec` calls 等。
5. **Iteration cap (`max_turns`)。** Agent loop 总 iterations；防止无限 reasoning loops。
6. **Per-minute / per-hour / per-day / per-month cap。** Rolling windows。在不同时间尺度抓 leaks。
7. **Financial velocity limit。** 例如“10 分钟内花费超过 $50 就切断访问”。在 monthly caps 触发前抓住 loop-based burn。
8. **Tiered model routing。** 默认使用更小模型；只有 classifier 判断任务值得时才升级到大模型。
9. **Prompt caching。** System prompt 和 stable context 存在 provider cache 中；重复发送的 token cost 接近零。
10. **Context windowing。** 通过 compaction / summarization 把 active context 保持在阈值以下；直接降低 token cost。
11. **HITL checkpoints on expensive actions。** 对已知昂贵动作（长 tool call、大下载、昂贵模型升级）要求人类确认。
12. **Kill switch on budget breach。** 任一 cap 触发时 session abort。记录触发 cap；重新启用必须走独立路径。

### Why the stack, not one cap / 为什么要栈，而不是一个 cap

单一月度 cap 只能在钱包已经空了之后抓住 runaway agent。单一 per-request cap 抓不住 session 级问题。不同失败模式需要不同时间尺度：

- **Runaway loop**（Agent 卡在 5 秒 retry 中）：由 velocity limit 抓住。
- **Slow leak**（Agent 每任务做了约 2x 预期工作）：由 daily cap 抓住。
- **Bad release**（新版本使用 5x tokens）：由 weekly / monthly cap 抓住。
- **Legitimate surge**（真实需求，而不是 bug）：由 hour / day cap 抓住，并带清晰 log。

### Claude Code's budget surface / Claude Code 的预算控制接口

Claude Code Agent SDK 在公开 docs 中暴露：

- `max_turns` — iteration cap。
- `max_budget_usd` — dollar cap；breach 时 session abort。
- `allowed_tools` / `disallowed_tools` — tool allowlist 和 denylist。
- tool use 前的 hook points，可用于自定义 cost-accounting。

把它和 permission-mode ladder（Lesson 10）组合。没有 `max_budget_usd` 的 `autoMode` session 是无治理的 autonomy。Anthropic 明确把 Auto Mode 描述为需要 budget controls；classifier 与 cost 正交。

### EU AI Act, OWASP Agentic Top 10 / EU AI Act 与 OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 覆盖 OWASP Agentic Top 10，以及 EU AI Act Article 14（human oversight）要求。对于欧盟生产系统，logging 和 cap enforcement 不是可选项。

### The observed $1,200 → $4,800 case / $1,200 → $4,800 案例

Microsoft docs 中的真实案例：某电商 Agent 增加新 tool 后，月度成本翻了三倍。这个 tool 让 Agent 在每个 session 中轮询订单状态。没有 loop detection。没有 per-tool cap。没有 week-over-week growth alert。修复方式是 per-tool cap 加 daily-growth alert。这个模式可复用：每个新 tool surface 都是新的潜在 loop；每个新 tool 都需要自己的 cap 和 alert。

## Build It / 动手构建

本课构建一个 layered cost-governor simulator：同一条 Agent trajectory 在没有治理、只有月度 cap、以及带 velocity / tool / turn / budget 分层限制时会产生不同结果。

## Use It / 应用它

`code/main.py` 模拟一个带和不带 layered cost-governor stack 的 Agent run。模拟 Agent 在若干 turns 后漂移进 polling loop；layered stack 会在 velocity window 内抓住它，而单一 monthly cap 要等数天后才会触发。

## Ship It / 交付它

`outputs/skill-agent-budget-audit.md` 审计 proposed agent deployment 的 cost-governor stack，并标记缺失层。

## Exercises / 练习

1. 运行 `code/main.py`。确认 polling-loop trajectory 上，velocity limit 在 iteration cap 之前触发。然后禁用 velocity limit，测量 Agent 在 iteration cap 抓住前“花掉”多少。

2. 为 browser agent（Lesson 11）设计一组 per-tool caps。哪个 tool 需要最严格 cap？哪个 tool 可以无界运行而没有风险？

3. 阅读 Microsoft Agent Governance Toolkit docs。列出 toolkit 命名的每种 cap type。把每一种映射到 failure mode（runaway loop、slow leak、bad release、surge）。

4. 给一个真实 overnight unattended run 定价（例如 “triage 50 issues in a repo”）。把 `max_budget_usd` 设为你的点估计的 2x。解释为什么是 2x。

5. Claude Code 的 `max_budget_usd` 按 session aggregate cost 触发。设计一个你会在外部强制执行的 complementary velocity limit。什么触发 cut-off？重新启用是什么样？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Denial of Wallet | “失控账单” | Agent loop 持续产生 spend，却没有 cap 阻止 |
| max_tokens | “Per-request cap” | 单次 completion 大小上限 |
| max_turns | “Iteration cap” | session 中 Agent loop iterations 上限 |
| max_budget_usd | “Dollar kill switch” | Session cost cap；breach 时 abort |
| Velocity limit | “Rate cap” | 短窗口内 spend 限制（例如 $50 / 10 min） |
| Tiered routing | “小模型优先” | 默认便宜模型；只有 classifier 判断值得才升级 |
| Prompt caching | “Cached system prompt” | Provider-side cache 把重发 token cost 降到接近零 |
| HITL checkpoint | “人类审批门” | 昂贵动作前需要人类确认 |

## Further Reading / 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、tool allowlists。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoints。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider-side cost controls。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — caching mechanics。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon agents 的 cost profile。
