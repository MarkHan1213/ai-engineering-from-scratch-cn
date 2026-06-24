# Human-in-the-Loop: Propose-Then-Commit / 人在回路：先提案，再提交

> 2026 年关于 HITL 的共识很具体。它不是“Agent 问一下，用户点 Approve”。它是 propose-then-commit：proposed action 会带 idempotency key 持久化到 durable store；展示给 reviewer 时附带 intent、data lineage、permissions touched、blast radius 和 rollback plan；只有 positive acknowledgement 后才 commit；执行后还要 verify，确认 side effect 真的发生。LangGraph 的 `interrupt()` 加 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现了同一形状。典型失败模式是 rubber-stamp approval：“Approve?” 被不经审查地点掉。文档化缓解方式是带显式 checklist 的 challenge-and-response。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites / 前置知识：** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 propose-then-commit state machine 的四步：propose、surface、commit、verify。
- 解释 idempotency key 如何防止 retry 后重复执行已批准动作。
- 区分 consequential actions、reversible actions、reads/inspections 的 HITL 要求。
- 设计 challenge-and-response checklist 来降低 rubber-stamp approvals。
- 为 HITL workflow 补齐 metadata、durability、verification 和 rollback hooks。

## The Problem / 问题

Agent 要采取一个动作。用户必须决定：批准还是不批准。如果这个决定是瞬时的，它很可能不是 review。如果这个决定是结构化的，它会更慢，但更可信。工程问题是：如何让结构化 review 成为阻力最小的路径。

2023 时代的 HITL pattern 是同步 prompt：“Agent wants to send email to X with body Y — approve?” 用户点击 Approve。大家感觉系统安全了。实践中，这个表面很容易被 rubber-stamp：用户快速批准，approval 对真实审查预测力很低；当 Agent 出错时，audit trail 里是一长串用户自己也想不起的 approvals。

2026 模式——propose-then-commit——把 HITL 移到 durable substrate 上，附加 structured metadata，并要求 positive commit。每个 managed agent SDK 都有一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名字不同，形状相同。

## The Concept / 概念

### The propose-then-commit state machine / propose-then-commit 状态机

1. **Propose / 提案。** Agent 产出 proposed action。它被持久化到 durable store（PostgreSQL、Redis、Durable Object）。包含：
   - intent（Agent 为什么这么做）
   - data lineage（哪个来源导致这个 proposal）
   - permissions touched（触及哪些 scopes / files / endpoints）
   - blast radius（最坏情况是什么）
   - rollback plan（如果 committed，如何撤销）
   - idempotency key（每个 proposal 唯一；重复提交返回同一 record）
2. **Surface / 展示。** Reviewer 看到带全部 metadata 的 proposal。Reviewer 是人，而不是 Agent 审查自己。
3. **Commit / 提交。** Positive acknowledgement。动作执行。
4. **Verify / 验证。** 执行后读回 side effect 并确认。如果 verify step 失败，系统进入已知坏状态并触发 alerting。

### The idempotency key / 幂等键

没有 idempotency key，transient failure 后的 retry 可能让已批准动作执行两次。具体例子：用户批准“从 A 向 B 转账 $100”。网络抖动。Workflow 重试。用户只批准一次，但转账执行两次。Idempotency key 把 approval 绑定到一个唯一 side effect；第二次执行是 no-op。

这与 Stripe 和 AWS APIs 使用的 idempotency pattern 相同。Microsoft Agent Framework docs 明确把它复用到 Agent approvals 中。

### Durability: why approvals outlast processes / Durability：为什么 approval 能跨进程存活

Approval waiting room 是 Agent 不拥有的一段状态。Workflow 暂停（Lesson 12）。Approval 到达时，workflow 从精确位置恢复。这就是 LangGraph 把 `interrupt()` 与 PostgreSQL checkpointing 配对，而不是只用 in-memory state 的原因——两天后的 approval 仍然能找到完整 workflow。

### Rubber-stamp approvals and the challenge-and-response mitigation / rubber-stamp approval 与 challenge-and-response

HITL 的默认 UI（“Approve” / “Reject” 按钮）会产生快速 approval，但不产生真实 review。文档化缓解方式：challenge-and-response checklist，要求 reviewer 对具体问题给出 positive answers 后，Approve 按钮才可用。具体形状：

- “Do you understand what resource this touches? [ ]”
- “Have you verified the blast radius is acceptable? [ ]”
- “Do you have a rollback plan if this fails? [ ]”

这不是为了官僚流程而官僚，而是 forcing function。无法勾选的 reviewer 要么要求澄清（escalation），要么拒绝（safe default）。Anthropic agent-safety research 明确引用 checklist-driven HITL 作为 rubber-stamp approval patterns 的缓解措施。

### What counts as consequential / 什么算 consequential

不是所有动作都需要 propose-then-commit。2026 指引：

- **Consequential actions（总是 HITL）**：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **Reversible actions（有时 HITL）**：本地文件编辑、staging-env 变更、带清晰 rollback 的可逆写入。
- **Reads and inspections（不需要 HITL）**：读取文件、列出资源、调用 read-only API。

### Post-action verification / 动作后验证

“commit 跑了”不等于“side effect 发生了”。Network partition 和 race conditions 会让 workflow 以为自己成功，但 backend 没有持久化。Verify step 会在 commit 后重新读取目标 resource 来确认。这与 database transactions 的 `RETURNING` clause、或 AWS `PutObject` 后 `GetObject` 是同一模式。

### EU AI Act Article 14 / EU AI Act 第 14 条

Article 14 要求欧盟 high-risk AI systems 具备 effective human oversight。“Effective” 不是装饰性用语。监管语言明确排除 rubber-stamp patterns。Microsoft Agent Governance Toolkit compliance docs 中，带 challenge-and-response 的 propose-then-commit 是能经受 Article 14 审查的形状。

## Build It / 动手构建

本课实现一个 propose-then-commit state machine：proposal 持久化、idempotency key 去重、approval 后 commit、commit 后 verify，并对比 rubber-stamp 与 challenge-and-response flow。

## Use It / 应用它

`code/main.py` 用 stdlib Python 实现 propose-then-commit state machine。Durable store 是 JSON file。Idempotency key 是 `(thread_id, action_signature)` 的 hash。Driver 模拟三个 case：clean approval flow、transient failure 后 retry（不能 double-execute）、rubber-stamp default 与 challenge-and-response flow 对比。

## Ship It / 交付它

`outputs/skill-hitl-design.md` 审查 proposed HITL workflow 是否具备 propose-then-commit shape，并标记缺失的 metadata、idempotency、verification 或 challenge-and-response layers。

## Exercises / 练习

1. 运行 `code/main.py`。确认 approved proposal 的 retry 使用 durable record 且不会重新执行。然后把 idempotency key 改成包含 timestamp，展示 retry 会 double-execute。

2. 给 proposal record 增加 `rollback` field。模拟 verify step 失败的 execution。展示 rollback 自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` docs。找出 API 包含而 toy engine 缺失的一个 metadata field。加上它，并解释它防护什么。

4. 为一个具体 action（例如 “post to a public Twitter account”）设计 challenge-and-response checklist。Reviewer 必须回答哪三个问题？为什么是这三个？

5. 选择一个同步 “Approve?” prompt 已经足够、不需要 durable store 的场景。解释原因，并说明你接受的是哪一类风险。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Propose-then-commit | “Two-phase approval” | Persisted proposal + positive commit + verify |
| Idempotency key | “Retry-safe token” | 每个 proposal 唯一；第二次 execution no-ops |
| Data lineage | “来源是什么” | 导致 proposal 的具体 source content |
| Blast radius | “最坏情况” | 动作出错时的影响范围 |
| Rubber-stamp | “快速批准” | 未经真实 review 就点击 “Approve” |
| Challenge-and-response | “强制 checklist” | Reviewer 必须正向确认具体问题 |
| RequestInfoEvent | “MS Agent Framework primitive” | 带 structured metadata 的 durable HITL request |
| `interrupt()` / `waitForApproval()` | “Framework primitives” | LangGraph / Cloudflare 的同形状等价物 |

## Further Reading / 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、durable approvals。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为 long-horizon risk 的缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — high-risk systems 的监管 baseline。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — oversight 的 constitutional framing。
