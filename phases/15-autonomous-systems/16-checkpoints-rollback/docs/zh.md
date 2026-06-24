# Checkpoints and Rollback / Checkpoint 与回滚

> 每个 graph-state transition 都会持久化。Worker 崩溃时，它的 lease 过期，另一个 worker 从最新 checkpoint 接手。Cloudflare Durable Objects 可以跨数小时或数周保存状态。Propose-then-commit（Lesson 15）为每个 action 定义 rollback plan。Post-action verification 闭合循环。EU AI Act Article 14 要求 high-risk systems 必须有 effective human oversight——实践中，这意味着 checkpoints 必须可查询，rollbacks 必须演练过，audit trail 必须跨 deploy 存活。尖锐失败模式是：没有 idempotency keys 和 precondition checks 时，transient failure 后的 retry 会 double-execute 一个已经批准的 action。Post-action verification 才能抓住它。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites / 前置知识：** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 checkpoint、lease recovery、idempotency、precondition 和 post-action verify 的分工。
- 识别 crash-during-commit 中最常见的 double-execute 事故路径。
- 为 consequential action 设计 rollback plan，并区分 in-band、compensating、out-of-band。
- 判断 checkpoint backend 是否能支撑 audit trail persistence。
- 设计 rehearsed-rollback test，验证真实 workflow 的失败恢复路径。

## The Problem / 问题

Durable execution（Lesson 12）让 crash 后的 Agent 可以恢复。Propose-then-commit（Lesson 15）让已批准动作可审计。本课把两者合起来：当一个已批准动作部分执行、崩溃、再恢复时，会发生什么？Rollback 什么时候运行？基于哪个状态运行？

真实系统有不同接线方式：

- **LangGraph** 会把每个 graph-state transition checkpoint 到 PostgreSQL。Worker crash 后，lease 释放，另一个 worker 从最新 checkpoint 恢复。Workflows 在 `interrupt()` 上暂停，而暂停本身也会持久化。
- **Cloudflare Durable Objects** 跨数小时或数周保存 per-key state。把计算和已批准 action 的 storage 放在一起。
- **Microsoft Agent Framework** 在 workflow API 中暴露 `Checkpoint` primitives；replay 加 idempotency 覆盖 retry。

无论实现差异如何，真正有效的组合都是：idempotency key（防 double-execute）+ precondition check（状态仍与批准时一致）+ post-action verify（side effect 真的发生）+ verify-fail 时 rollback。

## The Concept / 概念

### Every transition persists / 每个 transition 都持久化

Graph-state transition 是 workflow 从一个命名状态移动到另一个命名状态的任何步骤。Naive implementations 只在特定 commit points 持久化；生产实现会持久化每个 transition。成本（额外几次写入）相对可靠性收益很小：replay 可以落在任何位置，lease recovery 也更精确。

### Lease recovery / Lease 恢复

Worker 崩溃时，workflow 不会丢失；lease（一个短期 claim，表示该 worker 正在执行该 run）只是过期。另一个 worker 接手最新 checkpoint 并恢复。Lease mechanism 让生产系统可以在 rolling deploys 中不丢失 in-flight work。

### Idempotency plus preconditions / Idempotency 加 preconditions

只有 idempotency 还不够。考虑一个 workflow 被批准去“在 balance > $1000 时，从 A 向 B 转账 $100”。Workflow commit 后中途崩溃并恢复。如果只检查 idempotency key，那么恢复后 execution 只会运行一次（这点正确）。但如果在 crash 和 resume 之间，A 的 balance 被另一个 workflow 降到 $500 呢？Idempotency check 仍然通过；precondition 不通过。没有 precondition check，我们就会制造透支。

每个 consequential action 都需要两者：

- **Idempotency key**：防止 double-execute。
- **Precondition check**：确认状态仍与批准时一致。

### Post-action verification / 动作后验证

“Tool returned 200” 不是 verification。真正的 verification 会重新读取目标状态，确认 side effect 确实发生。模式包括：

- Database update：`UPDATE ... RETURNING *`，再断言返回 row 匹配 intended state。
- Email send：提交后检查 sent-folder 中的 message ID。
- File write：读回文件并 hash。
- API call：对目标 resource 做 follow-up `GET`。

如果 verify 失败，workflow 进入已知坏状态。Rollback 介入。

### Rollback plans / 回滚计划

Propose-then-commit（Lesson 15）中的每个 consequential action 都带 rollback plan。类型：

- **In-band rollback**：直接反转 side effect（`DELETE` after `INSERT`、发送后 `Send-correction-email`）。
- **Compensating transaction**：用一个新 action 抵消原动作（标准 SAGA pattern）。
- **Out-of-band rollback**：alert 人类、暂停 workflow、保留坏状态供调查。

No-op rollback（“we cannot undo this”）必须在 proposal 中明示。没有 rollback 的 actions 在 commit 时需要更强 HITL（Lesson 15 challenge-and-response）。

### EU AI Act Article 14 operational reading / EU AI Act 第 14 条的操作化解读

Article 14 要求 high-risk systems 具备 “effective human oversight”。在操作层，implementers 通常把它解读为：

- Checkpoints 可由 auditor 查询。
- Rollbacks 经过演练（至少端到端测试一次）。
- Audit trail 跨 deploy 存活（checkpoint backend 不能是 ephemeral）。
- Failed verifications 触发 alert，而不是静默写日志。

一个 workflow 如果在 mid-commit 崩溃、恢复并完成 side effect，却没有 verify + rollback path，就无法通过 Article 14 测试。

### The sharp failure mode: the double-execute / 尖锐失败模式：重复执行

这个领域最常见的生产事故：

1. Action approved，idempotency key k。
2. Commit starts，executes，returns 200。
3. Workflow 在持久化 “committed” status 前崩溃。
4. Workflow 恢复；看到 “approved but not committed”；重新执行。
5. Side effect 触发两次。

缓解方式：在 execution 前持久化一个 “in-flight” intent，带 idempotency key 执行，然后只有 post-action verification 成功后才标记 “committed”。如果 action 触发了但 status write 失败，你知道要 verify，并在必要时重放。如果 status write 成功但 action 失败，你会 verify，并通过 recovery path 精确触发一次。

## Build It / 动手构建

本课构建一个带 checkpoint、idempotency、precondition、verify 和 rollback 的 workflow。Driver 会模拟 clean run、crash retry、precondition fail 和 verify fail 四类路径。

## Use It / 应用它

`code/main.py` 实现一个 checkpointed workflow，包含 idempotency、preconditions、verify 和 rollback。Driver 模拟四个场景：clean run、crash 后 retry（idempotency 捕捉）、precondition fail（workflow 不触发 action 即 abort）、verify fail（rollback 触发）。

## Ship It / 交付它

`outputs/skill-rollback-rehearsal.md` 为 proposed workflow 设计 rollback-rehearsal test，并审计 checkpoint backend 的 audit-trail persistence。

## Exercises / 练习

1. 运行 `code/main.py`。验证四个场景。对于 crash-during-commit case，确认 action 在 retries 中只触发一次。

2. 修改 “mark as done first, then do it” pattern，让 status write 在 action 之后触发。重跑 crash scenario。测量重复 actions 触发了多少次。

3. 为一个具体生产动作（例如 “post to a Slack channel”）设计 rollback plan。分类为 in-band、compensating 或 out-of-band，并说明理由。

4. 选择一个你熟悉的 workflow。识别每个 state transition。为每个标记 durability requirement（persist / do not persist）。统计你当前没有持久化的 transition。

5. Rehearsed-rollback test：设计一个端到端测试，运行真实 workflow、让它崩溃，并确认 rollback path 触发。测试断言什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Checkpoint | “Save point” | 每个 graph-state transition 持久化到 durable store |
| Lease | “Worker claim” | 短期 claim，表示 worker 正在执行 run；crash 时过期 |
| Precondition | “State gate” | 断言状态仍与 approved action 一致 |
| Post-action verify | “Re-read check” | 确认 side effect 真的发生在 target system 中 |
| In-band rollback | “Direct undo” | 用逆操作反转 side effect |
| Compensating transaction | “SAGA undo” | 用新 action 中和原动作 |
| Mark-as-done-first | “Status write order” | commit 返回前先持久化 committed status |
| Article 14 | “EU AI Act human oversight” | 操作化含义：queryable checkpoints、rehearsed rollbacks、auditable trail |

## Further Reading / 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint primitives 和 lease recovery。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects as a state substrate。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon workflows 的 reliability framing。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的 workflow shape。
